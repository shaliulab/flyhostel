import os.path
import sqlite3
import json
import glob
import warnings

import cv2
import h5py
import numpy as np
from tqdm.auto import tqdm
import imgstore
ENCODER_FORMAT_GPU="h264_nvenc/mp4"


class SingleVideoMaker:

    def __init__(self, flyhostel_dataset, value=None):

        self._flyhostel_dataset = flyhostel_dataset
        self._basedir = os.path.dirname(flyhostel_dataset)
        self._index = os.path.join(self._basedir, "index.db")

        self.background_color = 255

        with sqlite3.connect(flyhostel_dataset, check_same_thread=False) as conn:

            cur = conn.cursor()
            cmd = "SELECT MIN(frame_number), MAX(frame_number) FROM ROI_0;"
            cur.execute(cmd)
            self.min_frame_number, self.max_frame_number = cur.fetchone()


            cmd = 'SELECT value FROM METADATA WHERE field = "idtrackerai_conf";'
            cur.execute(cmd)
            conf = cur.fetchone()[0]
            self._number_of_animals = int(json.loads(conf[0])["_number_of_animals"]["value"])
            
            cmd = 'SELECT value FROM METADATA WHERE field = "framerate";'
            cur.execute(cmd)
            self.framerate=int(cur.fetchone()[0])

            cmd = 'SELECT value FROM METADATA WHERE field = "chunksize";'
            cur.execute(cmd)
            self.chunksize=int(cur.fetchone()[0])


        if value is None:
            self._value = (self.min_frame_number, self.max_frame_number)

        else:
            assert value[0] >= self.min_frame_number
            assert value[1] <= self.max_frame_number
            self._value = value

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

    def init_video_writer(self, frameSize):
        folder = os.path.join(self._basedir, "flyhostel")
        os.makedirs(folder, exist_ok=True)

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
            fmt=ENCODER_FORMAT_GPU,
            framerate=self.framerate,
            basedir=os.path.join(folder, "flyhostel", "single_animal"),
            imgshape=frameSize[::-1],
            chunksize=self.chunksize,
            imgdtype=np.uint8,
        )

    def frame_number2chunk(self, frame_number):

        with sqlite3.connect(self._index, check_same_thread=False) as conn:

            cur = conn.cursor()
            cmd = "SELECT chunk FROM frames WHERE frame_number = (?);"
            cur.execute(cmd, frame_number)
            chunk = cur.fetchone()[0]
            return chunk


    def make_single_video(self, frameSize):

        width, height = frameSize

        with sqlite3.connect(self._index, check_same_thread=False) as conn:

            cur = conn.cursor()
            for chunk in range(self._value[0], self._value[1]+1):
                episode_images = sorted(glob.glob(f"idtrackerai/session_{str(chunk).zfill(6)}/segmentation_data/episode_images*"), key=lambda f: int(os.path.splitext(f)[0].split("_")[-1]))
                for episode_image in tqdm(episode_images, desc=f"Producing single animal video for {os.path.basename(self._flyhostel_dataset)}. Chunk {chunk}"):
                    key_counter=0
                    with h5py.File(episode_image, "r") as file:
                        keys = list(file.keys())
                        while key_counter < len(keys):
                            img=[]
                            for animal in range(self._number_of_animals):                            
                                key=keys[key_counter]
                                frame_number, blob_index = key.split("-")

                                frame_number = int(frame_number)
                                blob_index = int(blob_index)

                                if blob_index >= self._number_of_animals or blob_index != animal:
                                    warnings.warn(f"More blobs than animals in frame_number {frame_number}")

                                img_ = file[key][:]
                                assert img.shape[0] <= height
                                assert img.shape[1] <= width
                                if self.video_writer is None: self.init_video_writer(frameSize=(width*self._number_of_animals, height))
                                
                                # angle=self.fetch_angle(frame_number, blob_index)
                                # img=self.rotate_image(img, angle)
                                
                                img_=cv2.copyMakeBorder(img, 0, 0, max(0, width-img.shape[1]), max(0, height-img.shape[0]), cv2.BORDER_CONSTANT, self.background_color)
                                if img_.shape[0] > height:
                                    top = (img_.shape[0] // 2 - height // 2)
                                    img_=img_[top:(top+height), :]
                                if img_.shape[1] > width:
                                    left = (img_.shape[1] // 2 - width // 2)
                                    img_=img_[:, left:(left+width), :]

                                assert img_.shape[0] == height
                                assert img_.shape[1] == width

                                key_counter+=1

                            img = np.hstack(img.append(img_))

                            cur.execute("SELECT frame_number FROM frames WHERE frame_number = ?", frame_number)
                            frame_time = int(cur.fetchone()[0])
                            self.video_writer.add_image(img, frame_number, frame_time)


    @staticmethod
    def rotate_image(img, angle):

        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated
