import os.path
import sqlite3
import json
import glob
import math
import warnings
import logging

import cv2
import h5py
import numpy as np
import joblib
from tqdm.auto import tqdm
import imgstore
ENCODER_FORMAT_GPU="h264_nvenc/mp4"
ENCODER_FORMAT_CPU="mp4v/mp4"

from .hdf5_images import HDF5ImagesReader

logger = logging.getLogger(__name__)

class SingleVideoMaker:

    def __init__(self, flyhostel_dataset, value=None):

        self._flyhostel_dataset = flyhostel_dataset
        self._basedir = os.path.dirname(flyhostel_dataset)
        self._index = os.path.join(self._basedir, "index.db")

        self.background_color = 255
        print(f"Reading {self._flyhostel_dataset}")

        with sqlite3.connect(self._flyhostel_dataset, check_same_thread=False) as conn:

            cur = conn.cursor()
            cmd = "SELECT COUNT(frame_number) FROM ROI_0;"
            cur.execute(cmd)
            count = int(cur.fetchone()[0])
            if count == 0:
                raise Exception(f"{self._flyhostel_dataset} is empty")

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

    def init_video_writer(self, basedir, frameSize, first_chunk=0):

        # self.video_writer = cv2cuda.VideoWriter(
        #     os.path.join(folder, os.path.splitext(os.path.basename(self._flyhostel_dataset))[0], +".mp4"),
        #     apiPreference="FFMPEG",
        #     fourcc="h264_nvenc",
        #     fps=self.framerate,
        #     frameSize=frameSize,
        #     isColor=False,
        # )
        self.video_writer = imgstore.new_for_format(
            mode="w",
            fmt=ENCODER_FORMAT_CPU,
            framerate=self.framerate,
            basedir=basedir,
            imgshape=frameSize[::-1],
            chunksize=self.chunksize,
            imgdtype=np.uint8,
            first_chunk=first_chunk,
        )

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
            
        partition_size = math.ceil(len(chunks) / jobs)

        chunk_partition_ = [chunks[partition_size*i:((partition_size*i)+partition_size)] for i in range(jobs)]
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

        pattern=os.path.join(basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "segmentation_data", "episode_images*.hdf5")
        episode_images = sorted(glob.glob(pattern), key=lambda f: int(os.path.splitext(f)[0].split("_")[-1]))
        return episode_images


    @staticmethod
    def fetch_frame_time(cur, frame_number):
        cur.execute(f"SELECT frame_time FROM frames WHERE frame_number = {frame_number}")
        frame_time = int(cur.fetchone()[0])
        return frame_time


    def _make_single_video(self, chunks, basedir, output, frameSize, resolution, background_color=255, **kwargs):
        width, height = frameSize


        with sqlite3.connect(self._index, check_same_thread=False) as conn:
            cur = conn.cursor()
            target_fn = None

            for chunk in chunks:
            
                written_images=0
                count_white_imgs=0
                episode_images=self.list_episode_images(basedir, chunk)
                
                with HDF5ImagesReader(episode_images, width=width, height=height, resolution=resolution, background_color=background_color, chunk=chunk) as hdf5_reader:
                
                    # print(f"{len(keys)} keys found for chunk {chunk}")
                    while True:

                        data = hdf5_reader.read(target_fn, self._number_of_animals)
                        if data is None:
                            break
                        else:
                            frame_number, img = data

                        if self.video_writer is None:
                            resolution=(resolution[0] * self._number_of_animals, resolution[1])
                            self.init_video_writer(basedir=output, frameSize=resolution, **kwargs)
                            assert img.shape == resolution[::-1]

                        frame_time = self.fetch_frame_time(cur, frame_number)
                        self.video_writer.add_image(img, frame_number, frame_time, annotate=False)
                        written_images+=1
                        target_fn=frame_number+1

                #print(f"Written images: {written_images}")
                #print(f"White images: {count_white_imgs}")

    @staticmethod
    def rotate_image(img, angle):

        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated
