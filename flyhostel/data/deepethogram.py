import glob
import os.path
import logging

import joblib
import h5py

logger = logging.getLogger(__name__)


class H5Reader:
    
    def __init__(self, files, fps=160, main_key="resnet18"):
        self._files = files
        self.main_key = main_key
        with h5py.File(self._files[0], "r") as f:
             self._class_names = tuple([name.decode() for name in f[self.main_key]["class_names"][:]])
        
        self.fps = fps


    @property
    def class_names(self):
        return self._class_names

    @classmethod
    def from_outputs(cls, data_dir, prefix, *args, **kwargs):
        files = sorted(glob.glob(os.path.join(data_dir, prefix + "*", "*_outputs.h5")))
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
        
        return chunks, P

        

    @staticmethod
    def load_from_one_file(file, behavior, classes, main_key="resnet18"):
    
        chunk = int(os.path.basename(file).split("_")[-2])

        with h5py.File(file, "r") as f:
            local_classes = tuple([name.decode() for name in f[main_key]["class_names"][:]])
            assert local_classes == classes
            # features = f[self.main_key]["flow_features"][:]
            p = f[main_key]["P"][:, classes.index(behavior)]
        
        return chunk, p

    
