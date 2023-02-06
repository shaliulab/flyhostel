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

logger = logging.getLogger(__name__)


class H5Reader:
    
    def __init__(self, files, fps=160, main_key="resnet18"):
        self._files = files
        self.main_key = main_key
        self.fps = fps
        if not self._files:
            self._class_names = ()
        else:
            with h5py.File(self._files[0], "r") as f:
                self._class_names = tuple([name.decode() for name in f[self.main_key]["class_names"][:]])
        

    @property
    def class_names(self):
        return self._class_names

    @classmethod
    def from_outputs(cls, data_dir, prefix, in_frame_index, *args, **kwargs):
        key=f"{prefix}_[0-9][0-9][0-9][0-9][0-9][0-9]"
        files = sorted(glob.glob(os.path.join(data_dir, key, f"{key}_{in_frame_index}_outputs.h5")))[:-1]

        if len(files) == 0:
            if in_frame_index == 0:
                files = sorted(glob.glob(os.path.join(data_dir, key, f"{key}_outputs.h5")))[:-1]
            else:
                # no files found
                pass
        

        return cls(*args, files=files, **kwargs)



    def load(self, behavior, n_jobs):
        
        assert behavior in self._class_names
        
        logger.debug(f"Loading {behavior} data from {len(self._files)} files using {n_jobs} cores")

        Output = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(
                self.load_from_one_file
            )(file, behavior, self._class_names, main_key=self.main_key)
            for file in self._files
        )
        logger.debug("DONE")

        chunks=[]
        P=[]
        n_frames=0
        for chunk, p in Output:
            P.append(p)
            n_frames+=len(p)
            chunks.append(chunk)

    
        hours = round(n_frames/self.fps/3600, 2)
        logger.info(f"Loaded {hours} hours ({len(chunks)} chunks)")

        assert all(np.diff(chunks) == 1)
        
        return chunks, P

        

    @staticmethod
    def load_from_one_file(file, behavior, classes, main_key="resnet18"):
    
        chunk = int(os.path.basename(file).split("_")[-2])

        with h5py.File(file, "r") as f:
            if main_key not in list(f.keys()):
                warnings.warn(f"Cannot load {file}. {main_key} not available!")
                return chunk, np.array([])

            local_classes = tuple([name.decode() for name in f[main_key]["class_names"][:]])
            assert local_classes == classes
            # features = f[self.main_key]["flow_features"][:]
            p = f[main_key]["P"][:, classes.index(behavior)]
        
        return chunk, p

    
