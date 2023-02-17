from abc import ABC, abstractmethod
import warnings
import os.path
import re
import glob
import sqlite3

import pandas as pd
import h5py
import cv2
import numpy as np
from imgstore.stores.utils.mixins.extract import _extract_store_metadata
from flyhostel.data.yolov7 import letterbox


class HDF5VideoMaker(ABC):

    _basedir = None
    _number_of_animals = None
    _index=None
    video_writer=None

    @abstractmethod
    def init_video_writer(self, basedir, frame_size, first_chunk=0, chunksize=None):
        return


    @staticmethod
    def fetch_frame_time(cur, frame_number):
        return


    @staticmethod
    def list_episode_images(basedir, chunk):
        """
        List all episode_images_X.hdf5 files sorted by episode number (increasing)

        Such files should have the following naming scheme: episode_images_X.hdf5
        where X is the episode number, with or without zero padding
        """
        segmentation_data = os.path.join(basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "segmentation_data")
        pattern=os.path.join(segmentation_data, "episode_images*.hdf5")
        episode_images = sorted(
            glob.glob(pattern),
            key=lambda f: int(os.path.splitext(f)[0].split("_")[-1])
        )
        return episode_images


    def _make_single_video(self, chunks, output, frame_size, resolution, background_color=255, **kwargs):
        width, height = frame_size
        basedir  = self._basedir
        if output is None:
            output = os.path.join(basedir, "flyhostel", "single_animal")

        with sqlite3.connect(self._index, check_same_thread=False) as conn:
            cur = conn.cursor()
            target_fn = None

            for chunk in chunks:

                written_images=0
                count_NULL=0
                episode_images=self.list_episode_images(basedir, chunk)
                assert episode_images, f"{len(episode_images)} hdf5 files found"

                start_next_chunk = False

                with HDF5ImagesReader("flyhostel", episode_images, number_of_animals=self._number_of_animals, width=width, height=height, resolution=resolution, background_color=background_color, chunks=[chunk]) as hdf5_reader:

                    while True:

                        data = hdf5_reader.read(target_fn, self._number_of_animals)
                        if data is None:
                            break

                        frame_number, img = data
                        if img is None:
                            break

                        if self.video_writer is None:
                            resolution_full=(resolution[0] * self._number_of_animals, resolution[1])
                            fn = self.init_video_writer(basedir=output, frame_size=resolution_full, **kwargs)
                            print(f"Working on chunk {chunk}. Initialized {fn}. start_next_chunk = {start_next_chunk}")
                            assert img.shape == resolution_full[::-1]
                            assert str(chunk).zfill(6) in fn

                        frame_time = self.fetch_frame_time(cur, frame_number)
                        assert img.shape == resolution_full[::-1], f"{img.shape} != {resolution_full[::-1]}"
                        capfn=self.video_writer._capfn
                        fn = self.video_writer.add_image(
                            img, frame_number, frame_time, annotate=False,
                            start_next_chunk=start_next_chunk
                        )

                        written_images+=1
                        target_fn=frame_number+1
                        if fn is not None:
                            print(f"Working on chunk {chunk}. Initialized {fn}. start_next_chunk = {start_next_chunk}, chunks={chunks}")

                with open("status.txt", "a", encoding="utf8") as filehandle:
                    filehandle.write(f"Chunk {chunk}:{count_NULL}:{written_images}\n")


        self.video_writer.close()
        return capfn



class HDF5ImagesReader:


    _EXTENSION=".hdf5"


    def __init__(
        self,
        consumer, hdf5_files, chunks, width=None, height=None, resolution=None, background_color=255,
        img_size=640, stride=32, frequency=1, number_of_animals=None
        ):

        """
            consumer (str): Either flyhostel or yolov7. If in flyhostel consumer,
               frames are read and returned one frame number at a time,
               with a constant resolution


            hdf5_files (list): hdf5 datasets containing the bounding box of all animals on all frames
            of the recording. Each bounding box is stored under a key with the following pattern:

                    frameNumber-blobIndex[-modified]

                if a key has the suffix modified, the bounding box has been produced with a method
                other than the basic pixel intensity segmentation from idtrackerai (prepocessing step)
                Typically with some AI-based method

            chunks (list): Chunks of the recording to be processed
            width (int): Target width of the bounding box.
                If smaller than that, it is padded with the background_color on the right side
            height (int): Target height of the bounding box.
                If smaller than that, it is padded with the background_color on the bottom side
            resolution (tuple): Width and height of the output video.
                If not matching width and height, then the boxes are resized to the passed resolution
            background_color (int): 0-255 where 0 is black and 255 is white. Color that will be used to pad the boxes, if needed
            img_size (int): YOLOv7 letterbox algorithm specific. Always leave at 640. No effect if consumer != yolov7
            stride (int): YOLOv7 letterbox algorithm specific. Always leave at 32. No effect if consumer != yolov7
            frequency (int): Frames to be sampled per second of recording.
                If less than framerate, then some frames are skipped

        """
        self.consumer=consumer

        if self.consumer == "flyhostel":
            assert width is not None
            assert height is not None
            assert resolution is not None
            assert background_color is not None
            self.width = width
            self.height = height
            self.background_color = background_color
            self.resolution = resolution
            self._NULL = np.ones((self.height, self.width), np.uint8) * self.background_color
        else:
            self.width = self.height = self.background_color = self.resolution = None
            self._NULL = np.ones((height, width), np.uint8) * self.background_color

        self._files = hdf5_files
        self._file_idx = -1
        self._key=None
        self._key_counter = 0
        self._file = None
        self._tqdm=None
        self._update_filehandler()
        self._finished = False

        self._chunk = chunks[0]
        self.metadata=None
        self._experiment_metadata=None
        self._number_of_animals=number_of_animals
        self.img_size=img_size
        self.stride=stride
        self.frequency=frequency
        self.mode="image"

    @property
    def rect(self):
        if self.consumer == "flyhostel":
            return True
        else:
            return False


    @property
    def count(self):
        key=self.key
        if key is None:
            key=self._keys[self._key_counter]
        return self._parse_frame_number_from_key(key)

    @classmethod
    def from_sources(cls, metadata, chunks=None, **kwargs):

        pattern=os.path.join(os.path.dirname(metadata), "idtrackerai", "session_*", "segmentation_data", f"episode_images*{cls._EXTENSION}")
        sources_=glob.glob(pattern)
        s_chunk = [int(re.search("/session_([0-9]*)/", p).group(1)) for p in sources_]
        episode = [int(os.path.splitext(p)[0].split("_")[-1]) for p in sources_]

        df=pd.DataFrame({"source": sources_, "chunk": s_chunk, "episode": episode})
        if chunks is not None:
            df=df.loc[df["chunk"].isin(chunks)]
        else:
            chunks = sorted(list(set(df["chunk"].values.tolist())))
        df.sort_values(["chunk", "episode"], inplace=True)
        sources=df["source"].values.tolist()

        number_of_sources=len(sources)
        if number_of_sources > 0:
            print(f"{number_of_sources} sources detected")
        else:
            raise Exception(f"{number_of_sources} sources detected")

        experiment_metadata=_extract_store_metadata(metadata)
        number_of_animals=int(os.path.basename(os.path.dirname(os.path.dirname(metadata))).split("X")[0])
        reader=cls(hdf5_files=sources, number_of_animals=number_of_animals, chunks=chunks, **kwargs)
        reader.metadata = os.path.realpath(metadata)
        reader._experiment_metadata = experiment_metadata

        return reader

    @property
    def source(self):
        if self._file_idx >= len(self._files):
            return None
        else:
            return self._files[self._file_idx]

    @property
    def chunksize(self):
        return int(self._experiment_metadata["chunksize"])


    @property
    def framerate(self):
        return int(self._experiment_metadata["framerate"])

    @property
    def skip_n(self):
        # if frequency = 50 and framerate =150 -> 3
        return max(int(self.framerate / self.frequency) - 1, 1)

    @property
    def key(self):
        try:
            k=self._keys[self._key_counter]
            return k
        except Exception as error:
            print(error)
            end = self._update_filehandler()
            if end is None: return None
            else:
                return self.key

    @key.setter
    def key(self, value):
        frame_number = self._parse_frame_number_from_key(value)
        if frame_number > self._interval[-1]:
            end = self._update_filehandler()
            if end:
                self._key = None
        #assert self._interval[0] <= frame_number <= self._interval[1], f"""
        #Key value {value} invalid.
        #frame number must be within {self._interval[0]} and {self._interval[1]}
        #"""
        self._key = value
        try:
            self._key_counter = self._keys.index(value)
        except ValueError:
            warnings.warn(f"key {value} not found in current file, skipping to next")
            # the curernt key is not present in the current file
            end = self._update_filehandler()
            if end is None:
                self._key = None
            else:
                self._key_counter = self._keys.index(value)

    @property
    def frame_number(self):
        return self._parse_frame_number_from_key(self.key)


    def __enter__(self):
        return self


    def __exit__(self, type, value, traceback):
        self.close()


    def close(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):

        if self.consumer == "flyhostel":
            raise NotImplementedError
            # return self._next_flyhostel()

        if self.consumer == "yolov7":
            return self._next_yolov7()


    def _move_through_h5py(self):

        img0=[]
        keys=[]
        if self.key is None or self.source is None:
            print("STOP")
            raise StopIteration

        for i in range(5):
            frame_number=self._parse_frame_number_from_key(self.key)
            output = self._read_complex(frame_number, self._number_of_animals, stack=False)

            print(output["key"])
            img0.extend(output["img"])
            keys.extend(output["key"])

            print(f"Skipping {self.skip_n}")
            self.key = f"{self._parse_frame_number_from_key(self.key)+self.skip_n}-0"


        if len(img0) == 0:
            print("Error: early end detected")
            raise StopIteration

        print(keys)
        return img0, keys, output["source"]


    def _next_yolov7(self):

        """
        Returns:

            paths (list): Path to output .png for each img in img0
            img (np.ndarray): Array combining all images in img0 sent to YOLOv7 AI for processing
            img0 (list): Batches of images. Every element of the list is a batch of images i.e. another list (optionally of length 1)
            None
        """

        img0, keys, source = self._move_through_h5py()
        img0 = [cv2.cvtColor(img_, cv2.COLOR_GRAY2BGR) for img_ in img0]

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        try:
            chunk = int(re.search("session_([0-9]{6})/segmentation_data/episode_images", source).group(1))
        except TypeError as error:
            print(f"Cannot find chunk for file {source}")
            raise error

        paths = []
        for k in keys:
            parsed = k.split("-")
            if len(parsed) == 2:
                frame_number, blob_index = parsed

            # this can be if the blob is modified (it's suffixed with -modified)
            elif len(parsed) == 3:
                frame_number, blob_index, _ = parsed
            frame_idx = int(frame_number) % (chunk * self.chunksize)
            paths.append(f"{frame_number}_{chunk}-{frame_idx}-{blob_index}.png")
        print(f"Paths {paths}")
        return paths, img, img0, None


    @staticmethod
    def edit_image(img, width, height, background_color):
        img=cv2.copyMakeBorder(img, 0, max(0, height-img.shape[0]), 0, max(0, width-img.shape[1]), cv2.BORDER_CONSTANT, None, background_color)

        if img.shape[0] > height:
            #logger.debug(f"Chunk {chunk} - frame_number {frame_number}. Cropping {img.shape[0]-height} pixels along Y dim")
            top = (img.shape[0] // 2 - height // 2)
            img=img[top:(top+height), :]
        if img.shape[1] > width:
            #logger.debug(f"Chunk {chunk} - frame_number {frame_number}. Cropping {img.shape[1]-width} pixels along X dim")
            left = (img.shape[1] // 2 - width // 2)
            img=img[:, left:(left+width)]

        img=cv2.copyMakeBorder(img, 0, max(0, height-img.shape[0]), 0, max(0, width-img.shape[1]), cv2.BORDER_CONSTANT, None, background_color)
        assert img.shape[0] == height, f"{img.shape[0]} != {height}"
        assert img.shape[1] == width, f"{img.shape[1]} != {width}"
        return img


    @staticmethod
    def _resize_to_resolution(img, resolution):
        if img.shape != resolution[::-1]:
            img = cv2.resize(img, resolution[::-1], cv2.INTER_AREA)
        return img


    def _read(self):
        key=self.key
        # this means we went past the last of the last file
        if key is None:
            return None, None, None

        img_ = self._file[self.key][:]
        if self.consumer == "flyhostel":
            assert self.width is not None
            assert self.height is not None
            assert self.background_color is not None
            img_ = self.edit_image(img_, self.width, self.height, self.background_color)
            assert self.resolution is not None
            img_ = self._resize_to_resolution(img_, self.resolution)
        self._key_counter+=1
        self._check_end_of_file()
        return img_, key, self.source


    def _fetch(self, frame_number, in_frame_index):
        modified_key = f"{frame_number}-{in_frame_index}-modified"
        # NOTE
        # This assumes the modified version of the key
        # will exist in the next position
        try:
            if self._key_counter < (len(self._keys)-1) and self._keys[self._key_counter+1] == modified_key:
                self._key_counter+=1
        except Exception as error:
            raise error
            #print(self._key_counter, len(self._keys))
            #print(error)

        return self._read()


    def _read_complex(self, frame_number, number_of_animals=None, stack=True):
        """Load bounding box of all animals under the requested frame number

        Args:
            frame_number (int): Frame number of the recording requested
            number_of_animals (int): Expected number of animals in the frame (should be constant througout the recording)
            stack (bool, optional): Whether to return a single array with all animals stacked horizontally (True) or not (False). Defaults to True.

        Returns:
            output: Dictionary with keys frame_number, img, keys and source
                key is a list of the keys contained in the image
                a horizontally stacked imge or an array of images is available under img
                depending of whether stack was True or False
        """

        # no more data left in any file
        if self._finished:
            if stack:
                img = None
            else:
                img=[]

            return {"frame_number": frame_number, "img": img, "key": [None], "source": self.source}

        if number_of_animals is None:
            number_of_animals=self._number_of_animals

        if number_of_animals is None:
            raise ValueError("Please pass a number of animals")

        arrs = []
        keys=[]
        for i in range(number_of_animals):

            frame_number_ = self.frame_number
            if frame_number is None:
                frame_number=frame_number_
                img_, key, source = self._fetch(frame_number, i)
            else:
                # the frame number requested is less than the current key
                # which means there is no DATA and we should add NULL
                if frame_number < frame_number_:
                    img_ = self._NULL.copy()
                    key = self.key
                    source=self._files[self._file_idx]
                # the frame number requested
                # is the same as the current key
                elif frame_number == frame_number_:
                    img_, key, source = self._fetch(frame_number, i)
                # the frame number requested
                # is more than the current key
                # which means there are too many entries in a frame
                # and we need to move forward until they agree again
                else:
                    count=0
                    while frame_number > frame_number_:
                        self._key_counter+=1
                        self._check_end_of_file()
                        frame_number_ = self.frame_number
                        count+=1

                    warnings.warn(f"Skipped {count} keys to deliver frame number {frame_number}")
                    img_, key, source = self._fetch(frame_number, i)

            if key is not None:
                arrs.append(img_)
                keys.append(key)

        if key is not None:
            assert len(arrs) == number_of_animals
        if stack:
            img = np.hstack(arrs)
        else:
            img= arrs

        return {"frame_number": frame_number, "img": img, "key": keys, "source": source}


    def read(self, *args, **kwargs):
        output = self._read_complex(*args, **kwargs)
        return output["frame_number"], output["img"]


    def _check_end_of_file(self):

        if len(self._keys) == self._key_counter:
            if (self._file_idx+1) == len(self._files):
                self._finished=True
            else:
                self._update_filehandler()

    def _update_filehandler(self):
        self._file_idx +=1
        if self._file is not None:
            self._file.close()

        if self._tqdm is not None:
            self._tqdm.update(1)

        if self._file_idx >= len(self._files):
            return None

        print(f"Opening {self.source}")
        self._file = h5py.File(self.source, "r")
        self._keys = sorted(list(self._file.keys()), key=lambda k: int(k.split("-")[0]))

        self._interval = (
            self._parse_frame_number_from_key(self._keys[0]),
            self._parse_frame_number_from_key(self._keys[-1]),
        )

        self._key_counter=0
        return self._file

    @staticmethod
    def _parse_frame_number_from_key(key):
        return int(key.split("-")[0])

