import glob
import os.path

import numpy as np
import joblib
import h5py

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
        
        Output = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(
                self.load_from_one_file
            )(file, behavior, self._class_names, main_key=self.main_key)
            for file in self._files
        )
        
        data = np.concatenate(Output)
        hours = round(data.shape[0]/self.fps/3600, 2)
        
        print(f"Loaded {hours} hours")
        
        return data
        

    @staticmethod
    def load_from_one_file(file, behavior, classes, main_key="resnet18"):
        
        with h5py.File(file, "r") as f:
            local_classes = tuple([name.decode() for name in f[main_key]["class_names"][:]])
            assert local_classes == classes
            # features = f[self.main_key]["flow_features"][:]
            p = f[main_key]["P"][:, classes.index(behavior)]
        
        return p

    
