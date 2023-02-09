import warnings
import os.path
import re
import glob

import h5py
import cv2
import numpy as np
from tqdm.auto import tqdm
from imgstore.stores.utils.mixins.extract import _extract_store_metadata
from .yolov7 import letterbox

class HDF5ImagesReader:


    _EXTENSION=".hdf5"


    def __init__(self, mode, hdf5_files, number_of_animals, width, height, resolution, background_color, chunks):
        
        self._files = hdf5_files
        self._file_idx = -1
        self._key=None
        self._key_counter = 0
        self._file = None
        self._tqdm=None
        self._update_filehandler()
        self._finished = False
        self.width = width
        self.height = height
        self.background_color = background_color
        self.resolution = resolution
        self._NULL = np.ones((self.height, self.width), np.uint8) * self.background_color
        self._chunk = chunks[0]
        self.metadata=None
        self._experiment_metadata=None
        self._number_of_animals=number_of_animals
        self.mode=mode
        
        
    @classmethod
    def from_sources(cls, metadata, chunks=None, **kwargs):

       pattern=os.path.join(os.path.dirname(metadata), "idtrackerai", "session_*", "segmentation_data", "episode_images*.hdf5")
       sources=glob.glob(pattern)
       sources = sorted(sources, key=lambda f: int(os.path.splitext(f)[0].split("_")[-1]))

       sources = []

       for source in sources:
           assert os.path.exists(source) and source.endswith(cls._EXTENSION)
           if chunks:
               for chunk in chunks:
                   if re.search(f"session_{str(chunk).zfill(6)}", source):
                       sources.append(source)
                       break
           else:
               sources.append(source)
       print(f"{len(sources)} sources detected")

       experiment_metadata=_extract_store_metadata(metadata)
       number_of_animals=int(os.path.basename(os.path.dirname(os.path.dirname(metadata))).split("X")[0])
       reader=cls(hdf5_files=sources, number_of_animals=number_of_animals, chunks=chunks, **kwargs)
       reader.metadata = os.path.realpath(metadata)
       reader._experiment_metadata = experiment_metadata

       return reader


    @property
    def chunksize(self):
        return int(self._experiment_metadata["chunksize"])


    @property
    def framerate(self):
        return int(self._experiment_metadata["framerate"])

    @property
    def key(self):
        try:
            k=self.keys[self._key_counter]
            return k
        except Exception as error:
            end = self._update_filehandler()
            if end is None: return None
            else:
                return self.key
            
    @key.setter
    def key(self, value):
        frame_number = self._parse_frame_number_from_key(value)        
        
        assert self._interval[0] <= frame_number <= self._interval[1], f"""
        Key value {value} invalid.
        frame number must be within {self._interval[0]} and {self._interval[1]}
        """
        self._key = value        

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
        
        if self.mode == "flyhostel":
            raise NotImplementedError
            # return self._next_flyhostel()
        
        if self.mode == "yolov7":
            return self._next_yolov7()
        
        
    def move_through_h5py(self):

        img0=[]
        keys=[]
        source=self.source

        with h5py.File(source, "r") as file:
            for i in range(5):
                self._key_n += self.sampling_rate
                frame_number=self._parse_frame_number_from_key(self.key)
                output = self._read_complex(frame_number, self._number_of_animals, stack=False)
                keys=output["key"]
                img0.extend(output["img"])
                
                if keys[0] is None:
                    raise StopIteration
 
                keys.extend(keys)
                

        if len(img0) == 0:
            print("Error: early end detected")
            raise StopIteration
        return img0, keys, output["source"]

    
    def _next_yolov7(self):
            
        img0, keys, source = self._move_through_h5py()

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        chunk = int(re.search("session_([0-9]{6})/segmentation_data/episode_images", source).group(1))

        paths = []
        for k in keys:
            frame_number, blob_index = k.split("-")
            frame_idx = int(frame_number) % (chunk * self.chunksize)
            paths.append(f"{frame_number}_{chunk}-{frame_idx}-{blob_index}.png")
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
        if key is None:
            return None, None, None

        source=self._file
        img_ = self._file[self.key][:]
        img_ = self.edit_image(img_, self.width, self.height, self.background_color)
        img_ = self._resize_to_resolution(img_, self.resolution)
        self._key_counter+=1
        self._check_end_of_file() 
        return img_, key, source

    def _read_complex(self, frame_number, number_of_animals, stack=True):

        if self._finished:
            return None

        arrs = []
        keys = []
        for i in range(number_of_animals):

            frame_number_ = self.frame_number
            if frame_number is None:
                frame_number=frame_number_
                img_, key, source = self._read()

            else:
                if frame_number < frame_number_:
                    img_=self._NULL.copy()
                elif frame_number == frame_number_:
                    img_, key, source = self._read()

                else:
                    count=0
                    while frame_number > frame_number_:
                        self._key_counter+=1
                        self._check_end_of_file() 
                        frame_number_ = self.frame_number
                        count+=1

                    warnings.warn(f"Skipped {count} keys to deliver frame number {frame_number}")
                    img_, key, source = self._read()
                    
            if key is None:
                break

            arrs.append(img_)
            keys.append(key)
       
        assert len(arrs) == number_of_animals
        if stack:
            img = np.hstack(arrs)
        else:
            img = arrs

        return {"frame_number": frame_number, "img": img, "key": keys, "source": source}


    def read(self, *args, **kwargs):
        output = self._read_complex(*args, **kwargs)
        return output["frame_number"], output["img"]


    def _check_end_of_file(self):
        
        if len(self._keys) == self._key_counter:
            if (self._file_idx+1) == len(self._files):
                self._finished=True
                return None
            else:
                self._update_filehandler()

    def _update_filehandler(self):
        self._file_idx +=1
        if self._file is not None:
            self._file.close()
            return None
        
        if self._tqdm is not None: self._tqdm.update(1)
        self._file = h5py.File(self._files[self._file_idx], "r")
        self._keys = list(self._file.keys())

        self._interval = (
            self._parse_frame_number_from_key(self._keys[0]),
            self._parse_frame_number_from_key(self._keys[-1]),
        )
        
        self._key_counter=0
        return self._file
        
    @staticmethod
    def _parse_frame_number_from_key(key):
        return int(key.split("-")[0])
        
