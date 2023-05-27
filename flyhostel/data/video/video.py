import os.path
import sqlite3
import json
import logging
import warnings
import cv2
import numpy as np
import joblib
import imgstore

# from flyhostel.data.hdf5 import HDF5VideoMaker
from .maker import MP4VideoMaker

ENCODER_FORMAT_GPU="h264_nvenc/mp4"
ENCODER_FORMAT_CPU="mp4v/mp4"

def get_machine_id():
    """Read machine-id"""
    with open("/etc/machine-id", "r", encoding="utf8") as filehandle:
        return filehandle.read()


logger = logging.getLogger(__name__)

def validate_video(path):

    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    if not ret:
        warnings.warn(f"Validation for {path} failed. ret not True")

    if not isinstance(frame, np.ndarray):
        warnings.warn(f"Validation for {path} failed. output is not an array")
    if not (frame.shape[0] > 0 and frame.shape[1] > 0):
        warnings.warn(f"Validation for {path} failed. frame has size 0 in one of its dimensions")


class SingleVideoMaker(MP4VideoMaker):

    def __init__(self, flyhostel_dataset, identifiers, stacked=False, value=None):
        """
        
        identifiers (list): Local identity of the animals whose video you want to create.
            A list whose only element is -1 is interpreted as all identifiers available in this dataset

        """

        self._flyhostel_dataset = flyhostel_dataset
        self._basedir = "."
        self._index_db = os.path.join(self._basedir, "index.db")
        self._identifiers = identifiers
        self._stacked=stacked

        self.background_color = 255
        print(f"Reading {self._flyhostel_dataset}")

        with sqlite3.connect(self._flyhostel_dataset, check_same_thread=False) as conn:

            cur = conn.cursor()
            cmd = "SELECT COUNT(frame_number) FROM ROI_0;"
            # NOTE
            # This does not check if the local_identity desired is available
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


        if len(self._identifiers) == 0 or (len(self._identifiers) == 1 and self._identifiers[0] == -1):
            if self._number_of_animals == 1:
                self._identifiers = [0]
            else:
                self._identifiers = list(range(1, self._number_of_animals+1))


        if value is None:
            self._value = (self.min_frame_number, self.max_frame_number)

        else:
            assert value[0] >= self.min_frame_number
            assert value[1] <= self.max_frame_number
            self._value = value

        self._video_object_list={}
        self.video_writer = None

    @property
    def number_of_animals(self):
        return self._number_of_animals


    def fetch_angle(self, frame_number, blob_index):

        with sqlite3.connect(self._flyhostel_dataset, check_same_thread=False) as conn:
            cur = conn.cursor()
            cmd = "SELECT angle FROM ORIENTATION WHERE frame_number = ? AND in_frame_index = ?"
            cur.execute(cmd, frame_number, blob_index)
            angle = float(cur.fetchone()[0])

        return angle

    def init_video_writer(self, basedir, frame_size, identifier, chunk, first_chunk=0, chunksize=None):

        if chunksize is None:
            chunksize= self.chunksize
        print(f"chunksize = {chunksize}")

        self.video_writer[identifier] = imgstore.new_for_format(
            mode="w",
            fmt=ENCODER_FORMAT_CPU,
            framerate=self.framerate,
            basedir=basedir,
            imgshape=frame_size[::-1],
            chunksize=chunksize,
            imgdtype=np.uint8,
            first_chunk=first_chunk,
        )
        print(f"{basedir}:resolution={frame_size}:framerate={self.framerate}")

        txt_file = os.path.join(basedir, f"{str(chunk).zfill(6)}.txt")
        cached_images=0
        if os.path.exists(txt_file):
            with open(txt_file, "r") as filehandle:
                try:
                    cached_images=int(filehandle.readline().strip("\n"))
                except ValueError:
                    cached_images=0

                if cached_images == self.chunksize:
                    return None, cached_images
                elif cached_images != self.chunksize:
                    cached_images=0
                
        self.txt_file[identifier]=txt_file

        return self.video_writer[identifier]._capfn, cached_images

    def frame_number2chunk(self, frame_number):
        assert frame_number is not None

        with sqlite3.connect(self._index_db, check_same_thread=False) as conn:

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
            joblib.delayed(self._make_and_validate_single_video)(
                chunk_partition[i], first_chunk=chunk_partition[i][0], **kwargs
            )
            for i in range(len(chunk_partition))
        )


    def make_single_video_single_process(self, chunks=None, **kwargs):

        if chunks is None:
            chunks=list(range(self.frame_number2chunk(self._value[0]), self.frame_number2chunk(self._value[1])+1))

        self._make_and_validate_single_video(chunks=chunks, first_chunk=chunks[0], **kwargs)

    def _make_and_validate_single_video(self, *args, **kwargs):

        filename=self._make_single_video(*args, **kwargs)
        if filename is not None:
            print(f"Validating {filename}")
            validate_video(filename)


    @staticmethod
    def fetch_frame_time(cur, frame_number):
        cur.execute(f"SELECT frame_time FROM frames WHERE frame_number = {frame_number}")
        frame_time = int(cur.fetchone()[0])
        return frame_time


    @staticmethod
    def rotate_image(img, angle):

        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated

