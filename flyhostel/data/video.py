import os.path
import sqlite3
import json
import glob
import logging

import cv2
import numpy as np
import joblib
import imgstore
ENCODER_FORMAT_GPU="h264_nvenc/mp4"
ENCODER_FORMAT_CPU="mp4v/mp4"

def get_machine_id():
    with open("/etc/machine-id", "r", encoding="utf8") as filehandle:
         return filehandle.read()

from .hdf5 import HDF5ImagesReader

logger = logging.getLogger(__name__)

def validate_video(path):

    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    assert ret, f"Validation for {path} failed"
    assert isinstance(frame, np.ndarray), f"Validation for {path} failed"
    assert frame.shape[0] > 0 and frame.shape[1] > 0, f"Validation for {path} failed"


class SingleVideoMaker:

    def __init__(self, flyhostel_dataset, value=None):

        self._flyhostel_dataset = flyhostel_dataset
        flyhostel, x_number_of_animals, date, hour = os.path.splitext(os.path.basename(flyhostel_dataset))[0].split("_")
        self._basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], flyhostel, x_number_of_animals, f"{date}_{hour}")
        self._index = os.path.join(self._basedir, "index.db")

        self.background_color = 255
        print(f"Reading {self._flyhostel_dataset}")

        with sqlite3.connect(self._flyhostel_dataset, check_same_thread=False) as conn:

            cur = conn.cursor()
            cmd = "SELECT COUNT(frame_number) FROM ROI_0;"
            cur.execute(cmd)
            count = int(cur.fetchone()[0])
            if count == 0:
                raise ValueError(f"{self._flyhostel_dataset} is empty")

            cmd = "SELECT MIN(frame_number), MAX(frame_number) FROM ROI_0;"
            cur.execute(cmd)
            self.min_frame_number, self.max_frame_number = cur.fetchone()


            cmd = 'SELECT value FROM METADATA WHERE field = "idtrackerai_conf";'
            cur.execute(cmd)
            conf = cur.fetchone()[0]
            self._number_of_animals = int(json.loads(conf)["_number_of_animals"]["value"])

            cmd = 'SELECT value FROM METADATA WHERE field = "framerate";'
            cur.execute(cmd)
            self.framerate=int(float(cur.fetchone()[0]))

            cmd = 'SELECT value FROM METADATA WHERE field = "chunksize";'
            cur.execute(cmd)
            self.chunksize=int(float(cur.fetchone()[0]))


        if value is None:
            self._value = (self.min_frame_number, self.max_frame_number)

        else:
            assert value[0] >= self.min_frame_number
            assert value[1] <= self.max_frame_number
            self._value = value

        self._video_object_list={}
        self.video_writer = None


    def fetch_angle(self, frame_number, blob_index):

        with sqlite3.connect(self._flyhostel_dataset, check_same_thread=False) as conn:
            cur = conn.cursor()
            cmd = "SELECT angle FROM ORIENTATION WHERE frame_number = ? AND in_frame_index = ?"
            cur.execute(cmd, frame_number, blob_index)
            angle = float(cur.fetchone()[0])

        return angle

    def init_video_writer(self, basedir, frame_size, first_chunk=0, chunksize=None):

        if chunksize is None:
            chunksize= self.chunksize
        print(f"chunksize = {chunksize}")

        self.video_writer = imgstore.new_for_format(
            mode="w",
            fmt=ENCODER_FORMAT_CPU,
            framerate=self.framerate,
            basedir=basedir,
            imgshape=frame_size[::-1],
            chunksize=chunksize,
            imgdtype=np.uint8,
            first_chunk=first_chunk,
        )
        print(f"{basedir}:resolution={frame_size[::-1]}:framerate={self.framerate}")
        return self.video_writer._capfn

    def frame_number2chunk(self, frame_number):
        assert frame_number is not None

        with sqlite3.connect(self._index, check_same_thread=False) as conn:

            cur = conn.cursor()
            cmd = f"SELECT chunk FROM frames WHERE frame_number = {frame_number};"
            cur.execute(cmd)
            chunk = cur.fetchone()[0]
            return chunk


    def make_single_video_multi_process(self, n_jobs=-2, chunks=None, **kwargs):

        if chunks is None:
            chunks=list(range(self.frame_number2chunk(self._value[0]), self.frame_number2chunk(self._value[1])+1))
        nproc=len(os.sched_getaffinity(0))

        if n_jobs>0:
            jobs = n_jobs
        else:
            jobs = nproc + n_jobs

        # partition_size = math.ceil(len(chunks) / jobs)
        # I cannot have the same joblib.Process continue the imgstore on to the new chunk
        # Instead, each Process needs to produce one chunk only and exit
        partition_size = 1
        n_blocks=len(chunks)
        chunk_partition_ = [chunks[partition_size*i:((partition_size*i)+partition_size)] for i in range(n_blocks)]

        chunk_partition = []
        for partition in chunk_partition_:
            if len(partition)>0:
                chunk_partition.append(partition)

        print("Chunk partition:")
        for partition in chunk_partition:
            print(partition)

        joblib.Parallel(n_jobs=jobs)(
            joblib.delayed(self._make_single_video)(
                chunk_partition[i], first_chunk=chunk_partition[i][0], **kwargs
            )
            for i in range(len(chunk_partition))
        )


    def make_single_video_single_process(self, chunks=None, **kwargs):
        if chunks is None:
            chunks=list(range(self.frame_number2chunk(self._value[0]), self.frame_number2chunk(self._value[1])+1))

        self._make_single_video(chunks=chunks, first_chunk=chunks[0], **kwargs)




    @staticmethod
    def list_episode_images(basedir, chunk):
        """
        List all episode_images_X.hdf5 files sorted by episode number (increasing)

        Such files should have the following naming scheme: episode_images_X.hdf5
        where X is the episode number, with or without zero padding
        """
        segmentation_data = os.path.join(basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "segmentation_data")
        print(segmentation_data)
        pattern=os.path.join(segmentation_data, "episode_images*.hdf5")
        episode_images = sorted(glob.glob(pattern), key=lambda f: int(os.path.splitext(f)[0].split("_")[-1]))
        return episode_images


    @staticmethod
    def fetch_frame_time(cur, frame_number):
        cur.execute(f"SELECT frame_time FROM frames WHERE frame_number = {frame_number}")
        frame_time = int(cur.fetchone()[0])
        return frame_time


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
                        fn = self.video_writer.add_image(img, frame_number, frame_time, annotate=False, start_next_chunk=start_next_chunk)
                        written_images+=1
                        target_fn=frame_number+1
                        if fn is not None:
                            print(f"Working on chunk {chunk}. Initialized {fn}. start_next_chunk = {start_next_chunk}, chunks={chunks}")

                with open("status.txt", "a", encoding="utf8") as filehandle:
                    filehandle.write(f"Chunk {chunk}:{count_NULL}:{written_images}\n")


        self.video_writer.close()
        print(f"Validating {capfn}")
        validate_video(capfn)



    @staticmethod
    def rotate_image(img, angle):

        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated
