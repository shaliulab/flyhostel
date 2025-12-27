import os.path

import h5py
import numpy as np

# Fix these imports
# from flyhostel.data.pose.constants import framerate as FRAMERATE
# from flyhostel.data.pose.constants import chunksize as CHUNKSIZE

def add_frame_number_index(path, ds):
    
    with h5py.File(path, "r") as f:
        files=[e.decode() for e in f["files"][:]]
        chunks=[int(os.path.basename(e).split(".")[0]) for e in files]
        frame_number=np.concatenate([chunk*CHUNKSIZE + np.arange(CHUNKSIZE) for chunk in chunks]).tolist()
    
    ds["frame_number"]=frame_number
    ds = ds.assign_coords(frame_number=("time", frame_number))
    return ds

def from_sleap_file(path):
    raise NotImplementedError()
    if load_poses is None:
        raise ImportError("Movement installation is not updated")
    ds = load_poses.from_sleap_file(path, fps=FRAMERATE)
    ds=add_frame_number_index(path, ds)
    return ds
