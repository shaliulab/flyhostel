"""
Read h5 files produced by deepethogram

The module exposes H5Reader, a class that can be used to fully load the annotations produced by deepethogram
for a specific behavior (specified in the load method).
The annotation is added so chronological order is preserved i.e. as a timeseries,
and this is ensured because all data within chunk is ordered and comes with the chunk number
so chronological order between chunks is not required in the output
"""

import glob
import os.path
import logging
import warnings

import joblib
import numpy as np
import h5py
import cv2


logger = logging.getLogger(__name__)

def get_framerate(file):
    example_video_file = os.path.splitext(file)[0].replace("_outputs", "") + ".mp4"
    cap=cv2.VideoCapture(example_video_file)
    framerate=cap.get(5)
    cap.release()
    return framerate

class H5Reader:

    def __init__(self, files, framerate, data_framerate=None, main_key="resnet18"):
        """
        Args:
            fps (int):
        """
        self._files = files
        self.main_key = main_key
        self._framerate = framerate
        self._data_framerate = data_framerate
        if not self._files:
            self._class_names = ()
        else:
            with h5py.File(self._files[0], "r") as filehandle:
                self._class_names = tuple([name.decode() for name in filehandle[self.main_key]["class_names"][:]])

    @property
    def framerate(self):
        return self._framerate


    @property
    def data_framerate(self):
        if self._data_framerate is None:
            return self.framerate
        else:
            return self._data_framerate


    @property
    def step(self):
        return max(int(self.framerate / self.data_framerate), 1)

    @property
    def class_names(self):
        return self._class_names


    @classmethod
    def from_outputs(cls, data_dir, prefix, local_identity, frequency, *args, **kwargs):
        key=f"{prefix}_[0-9][0-9][0-9][0-9][0-9][0-9]_{str(local_identity).zfill(3)}"
        pattern=os.path.join(data_dir, key, f"{key}_outputs.h5")

        files = sorted(glob.glob(pattern))

        if len(files) == 0:
            pass

        framerate = get_framerate(files[0])
        return cls(*args, framerate=framerate, data_framerate=frequency, files=files, **kwargs)


    def load(self, behavior, n_jobs):

        assert behavior in self._class_names

        logger.debug(f"Loading %s data from %d files using %d cores", (behavior, len(self._files), n_jobs))

        Output = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(
                self.load_from_one_file
            )(file, behavior, self._class_names, step=self.step, main_key=self.main_key)
            for file in self._files
        )
        logger.debug("DONE")

        chunks=[]
        p_list=[]
        n_frames=0
        for chunk, probability in Output:
            p_list.append(probability)
            n_frames+=len(probability)
            chunks.append(chunk)


        hours = round(n_frames/self.data_framerate/3600, 2)
        logger.info(f"Loaded {hours} hours ({len(chunks)} chunks)")

        assert all(np.diff(chunks) == 1)

        return chunks, p_list



    @staticmethod
    def load_from_one_file(file, behavior, classes, step, main_key="resnet18"):

        file_split = os.path.basename(file).split("_")
        if len(file_split) == 6:
            # old format, deprecated
            chunk = int(file_split[-2])
        elif len(file_split) == 7:
            # file should have following structure:
            # FlyHostelX_xX_date_time_chunk_identity_outputs.h5
            chunk = int(file_split[-3])

        with h5py.File(file, "r") as f:
            if main_key not in list(f.keys()):
                warnings.warn(f"Cannot load {file}. {main_key} not available!")
                return chunk, np.array([])

            local_classes = tuple([name.decode() for name in f[main_key]["class_names"][:]])
            assert local_classes == classes
            # features = f[self.main_key]["flow_features"][:]
            p_array = f[main_key]["P"][:, classes.index(behavior)][::step, :]

        return chunk, p_array


