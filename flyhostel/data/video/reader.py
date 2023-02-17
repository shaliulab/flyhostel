import re
import glob
import os.path
import sqlite3

import cv2
import pandas as pd
import numpy as np

from imgstore.interface import VideoCapture
from imgstore.stores.utils.mixins.extract import _extract_store_metadata
from flyhostel.data.yolov7 import letterbox

class MP4Reader:

    _EXTENSION = ".mp4"
    BATCHES=True
    BATCH_SIZE=5
    # IDENTIFIER_COLUMN="in_frame_index"

    IDENTIFIER_COLUMN="local_identity" # identity is similar but will not be consistent across chunks


    def __init__(
        self,
        consumer, connection, store_path, chunks, width=None, height=None, resolution=None, background_color=255,
        img_size=640, stride=32, frequency=1, number_of_animals=None
        ):

        """
            consumer (str): Either flyhostel or yolov7. If in flyhostel consumer,
               frames are read and returned one frame number at a time,
               with a constant resolution

            connection (sqlite3.Connection)

            frequency (int): Frames to be sampled per second of recording
        """
        self.consumer=consumer
        self.connection=connection
        self.store_path=store_path
        self._experiment_metadata=_extract_store_metadata(self.store_path)

        assert width is not None
        assert height is not None
        assert resolution is not None
        assert background_color is not None


        self.width = width
        self.height = height
        self.background_color = background_color
        self.resolution = resolution


        self._NULL = np.ones((height, width), np.uint8) * self.background_color

        self._file_idx = -1
        self._key=None
        self._key_counter = 0
        self._file = None
        self._tqdm=None
        self._finished = False

        self._chunk = chunks[0]
        self.metadata=None
        self._number_of_animals=number_of_animals
        self.img_size=img_size
        self.stride=stride
        self.frequency=frequency
        self.mode="image"
        self.rect=True

        self._cap = VideoCapture(self.store_path, self._chunk)
        self._frame_number=self._cap.frame_number

        self._cur=self.connection.cursor()
        self._cur.execute(self.sqlite_query,(self._chunk,))

        self._data = pd.DataFrame(self._cur.fetchall())
        self._data.columns = ["frame_number", "x", "y", self.IDENTIFIER_COLUMN, "chunk"]
        self._data.set_index(["frame_number", self.IDENTIFIER_COLUMN], inplace=True)
        self._data["x"] = np.int32(np.floor(self._data["x"]))
        self._data["y"] = np.int32(np.floor(self._data["y"]))
        self._last_frame_indices=[]

    @property
    def sqlite_query(self):
        if self.IDENTIFIER_COLUMN in ["identity", "local_identity"]:
            cmd=f"""SELECT
                DT.frame_number,
                DT.x,
                DT.y,
                    ID.{self.IDENTIFIER_COLUMN},
                IDX.chunk
            FROM
                ROI_0 AS DT
                INNER JOIN STORE_INDEX AS IDX on DT.frame_number = IDX.frame_number
                INNER JOIN IDENTITY AS ID on DT.frame_number = ID.frame_number AND DT.in_frame_index = ID.in_frame_index
            WHERE
                IDX.chunk = ?
            """

        elif self.IDENTIFIER_COLUMN=="in_frame_index":
            cmd=f"""SELECT
                DT.frame_number,
                DT.x,
                DT.y,
                DT.{self.IDENTIFIER_COLUMN},
                IDX.chunk
            FROM
                ROI_0 AS DT
                INNER JOIN STORE_INDEX AS IDX on DT.frame_number = IDX.frame_number
            WHERE
                IDX.chunk = ?
            """

        return cmd

    @property
    def identifier_start(self):
        if self.IDENTIFIER_COLUMN in ["identity", "local_identity"]:
            return 1
        elif self.IDENTIFIER_COLUMN=="in_frame_index":
            return 0


    @property
    def count(self):
        return self._frame_number

    @property
    def frame(self):
        return self._frame_number

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


    @classmethod
    def from_store_path(cls, consumer, store_path, chunks=None, **kwargs):

        number_of_animals=int(os.path.basename(os.path.dirname(os.path.dirname(store_path))).split("X")[0])

        if chunks is None:
            session_folders = sorted(glob.glob(os.path.join(os.path.dirname(store_path), "idtrackerai", "session_*/")))
            s_chunk = [int(re.search("/session_([0-9]*)/", p).group(1)) for p in session_folders]
            df=pd.DataFrame({"chunk": s_chunk})
            chunks = sorted(list(set(df["chunk"].values.tolist())))

        flyhostel_dataset = glob.glob(os.path.join(os.path.dirname(store_path), "FlyHostel*db"))
        assert flyhostel_dataset, f"FlyHostel dataset not found for experiment {store_path}"
        flyhostel_dataset=flyhostel_dataset[0]
        conn=sqlite3.connect(flyhostel_dataset, check_same_thread=False)

        return cls(
            consumer, connection=conn, store_path=store_path, chunks=chunks[:1],
            number_of_animals=number_of_animals, **kwargs
        )




    def get_bounding_box(self, x_coord, y_coord):
        """
        Compute a bounding box centered at the passed centroid
        with the object's width and height

        Return:

            x, y, w, h of the box with origin at top left
        """

        tr_x = x_coord - self.width // 2
        tr_y = y_coord - self.height // 2
        return tr_x, tr_y, self.width, self.height


    def crop_image(self, img, centroid):

        bbox = self.get_bounding_box(*centroid)
        # print(bbox)
        x_coord, y_coord, width, height = bbox

        x2 = x_coord + width
        y2 = y_coord + height

        diff_x = -x_coord
        diff_y = -y_coord


        if diff_x > 0:
            x_coord = 0
            x2=width-diff_x

        if diff_y > 0:
            y_coord = 0
            y2=height-diff_y

        diff_w = x2 - img.shape[1]
        diff_h = y2 - img.shape[0]


        if diff_w > 0:
            x2 = img.shape[1]

        if diff_h > 0:
            y2 = img.shape[0]

        img_ = img[y_coord:y2, x_coord:x2]
        img_=cv2.copyMakeBorder(
            img_,
            max(0, diff_y),
            max(0, diff_h),
            max(0, diff_x),
            max(0, diff_w),
            cv2.BORDER_CONSTANT,
            value=self.background_color
        )
        # print(img_.shape, diff_y, diff_h, diff_x, diff_w)
        return img_


    def get_centroid(self, frame_number, identifier):
        try:
            x_coord, y_coord, _ = self._data.loc[frame_number, identifier].values.tolist()
        except KeyError:
            # an animal in this frame number with this identifier is not found
            return None
        except ValueError:
            # more than one match
            # TODO Do something about it, at least warn the user
            return None

        return (x_coord, y_coord)


    def read(self, frame_number, number_of_animals, stack=False):

        if frame_number is None:
            frame_number = self._cap.frame_number
            print(f"Reading frame number {frame_number}")

        img, (fn, ft) = self._cap.get_image(frame_number)
        self._frame_number = fn
        assert fn == frame_number
        if self._cap._chunk_n != self._chunk:
            self.close()
            return None

        arr = []
        for identifier in range(number_of_animals):
            identifier=identifier+self.identifier_start
            centroid = self.get_centroid(frame_number, identifier=identifier)
            if centroid is None:
                img_=self._NULL.copy()
            else:
                img_=self.crop_image(img, centroid)
            arr.append(img_)
            self._last_frame_indices.append(identifier)

        if stack:
            img = np.hstack(arr)
        else:
            img = arr

        return frame_number, img


    def __enter__(self):
        return self


    def __exit__(self, type, value, traceback):
        self.close()


    def __iter__(self):
        return self

    def __next__(self):

        if self.consumer == "flyhostel":
            assert self._number_of_animals is not None
            frame_number, frame =self.read(self.count+1, self._number_of_animals)
            return frame

        if self.consumer == "yolov7":
            return self._next_yolov7()


    def _prepare_for_yolo(self, batches):
        """

        Args:
            batches (list): Images to run YOLOv7 on
        """
        batches = [cv2.cvtColor(img_, cv2.COLOR_GRAY2BGR) for img_ in batches]

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in batches]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        return img


    def _next_yolov7(self):

        batches=[]
        paths=[]


        for _ in range(self.BATCH_SIZE):
            frame_number=self.count+1

            frame_number_, batch=self.read(frame_number, self._number_of_animals)
            if batch is None:
                raise StopIteration
            indices=self._last_frame_indices
            frame_idx = int(frame_number) % (self._chunk * self.chunksize)

            batch_paths = [
                f"{frame_number}_{self._chunk}-{frame_idx}_{identifier}.png"
                for identifier in indices
            ]
            self._last_frame_indices.clear()

            paths.extend(batch_paths)
            batches.extend(batch)


        img = self._prepare_for_yolo(batches)


        return paths, img, batches, None


    def close(self):
        if self.consumer == "yolov7":
            self.connection.close()