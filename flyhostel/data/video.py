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


    def make_single_video_multi_process(self, n_jobs=-2, **kwargs):

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


    def make_single_video_single_process(self, **kwargs):
        chunks=range(self.frame_number2chunk(self._value[0]), self.frame_number2chunk(self._value[1])+1)
        self._make_single_video(chunks=chunks, **kwargs)


    def _make_single_video(self, chunks, basedir, output, frameSize, resolution, **kwargs):
        width, height = frameSize

        white_img = np.ones((height, width), np.uint8) * 255
        
        with sqlite3.connect(self._index, check_same_thread=False) as conn:
            cur = conn.cursor()
            target_fn = None

            for chunk in chunks:
                episode_images = sorted(glob.glob(os.path.join(basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "segmentation_data", "episode_images*")), key=lambda f: int(os.path.splitext(f)[0].split("_")[-1]))
                # print(f"{len(episode_images)} hdf5 files found for chunk {chunk}")
                for episode_image in tqdm(episode_images, desc=f"Producing single animal video for {os.path.basename(self._flyhostel_dataset)}. Chunk {chunk}"):
                    key_counter=0
                    missing=False
                    with h5py.File(episode_image, "r") as file:
                        keys = list(file.keys())
                        # print(f"{len(keys)} keys found for chunk {chunk}")
                        while key_counter < len(keys):
                            imgs=[]
                            for animal in range(self._number_of_animals):                            
                                key=keys[key_counter]
                                frame_number, blob_index = key.split("-")
                                
                                frame_number = int(frame_number)
                                blob_index = int(blob_index)

                                if target_fn is None:
                                    target_fn = frame_number

                                if target_fn < frame_number:
                                    #import ipdb; ipdb.set_trace()
                                    frame_number, last_blob_index=keys[key_counter-1].split("-")
                                    frame_number = int(frame_number)
                                    blob_index = int(blob_index)
                                    warnings.warn(f"Missing blobs ({last_blob_index} < {self._number_of_animals}). Too few animals in frame_number {target_fn}")
                                    missing=True
                                    break


                                while target_fn > frame_number:
                                    warnings.warn(f"Skipping key {key}. Too many animals in frame_number {frame_number}")
                                    key_counter+=1
                                    if key_counter == len(keys): return
                                    key=keys[key_counter]
                                    frame_number, blob_index = key.split("-")
                                    
                                    frame_number = int(frame_number)
                                    blob_index = int(blob_index)

                                if blob_index >= self._number_of_animals or blob_index != animal:
                                    warnings.warn(f"More blobs than animals in frame_number {frame_number}")

                                img_ = file[key][:]
                                #assert img_.shape[0] <= height, f"{img_.shape[0]} > {height}"
                                #assert img_.shape[1] <= width, f"{img_.shape[1]} > {width}"

                                if self.video_writer is None:
                                    resolution=(resolution[0] * self._number_of_animals, resolution[1])
                                    self.init_video_writer(basedir=output, frameSize=resolution, **kwargs)
                                
                                # angle=self.fetch_angle(frame_number, blob_index)
                                # img=self.rotate_image(img, angle)
                                
                                img_=cv2.copyMakeBorder(img_, 0, max(0, height-img_.shape[0]), 0, max(0, width-img_.shape[1]), cv2.BORDER_CONSTANT, None, self.background_color)

                                # if resize:
                                #     if img_.shape[0] > height or img_.shape[1] > width:
                                #         longest=np.argmax(img_.shape[:2])
                                #         target_shape=[height, width]
                                #         target_shape[1-longest]=int(target_shape[longest] * img_.shape[1-longest] / img_.shape[longest])
                                #         target_shape=tuple(target_shape)
                                #         logger.debug(f"Chunk {chunk} - frame_number {frame_number}. Resizing {img_.shape} to {target_shape}")
                                #         img_ = cv2.resize(img_, target_shape, cv2.INTER_AREA)
                                # else:
                                if img_.shape[0] > height:
                                    warnings.warn(f"Chunk {chunk} - frame_number {frame_number}. Cropping {img_.shape[0]-height} pixels along Y dim")
                                    top = (img_.shape[0] // 2 - height // 2)
                                    img_=img_[top:(top+height), :]
                                if img_.shape[1] > width:
                                    warnings.warn(f"Chunk {chunk} - frame_number {frame_number}. Cropping {img_.shape[1]-width} pixels along X dim")
                                    left = (img_.shape[1] // 2 - width // 2)
                                    img_=img_[:, left:(left+width)]

                                img_=cv2.copyMakeBorder(img_, 0, max(0, height-img_.shape[0]), 0, max(0, width-img_.shape[1]), cv2.BORDER_CONSTANT, None, self.background_color)
                                assert img_.shape[0] == height, f"{img_.shape[0]} != {height}"
                                assert img_.shape[1] == width, f"{img_.shape[1]} != {width}"

                                key_counter+=1
                                imgs.append(img_)

                            if len(imgs) < self._number_of_animals:
                                if missing:
                                    for _ in range(self._number_of_animals - len(imgs)):
                                        imgs.append(white_img.copy())
                                        
                                    missing=False
                            
                                else:
                                    raise Exception(f"Missing blobs in frame_number {frame_number}. Stopping")

                            img = np.hstack(imgs)
                            target_fn=frame_number+1

                            cur.execute(f"SELECT frame_time FROM frames WHERE frame_number = {frame_number}")
                            frame_time = int(cur.fetchone()[0])
                            
                            if img.shape != resolution[::-1]:
                                img = cv2.resize(img, resolution[::-1], cv2.INTER_AREA)

                            self.video_writer.add_image(img, frame_number, frame_time, annotate=False)


    @staticmethod
    def rotate_image(img, angle):

        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated
