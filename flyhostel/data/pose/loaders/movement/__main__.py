import os.path
import logging
import h5py
import pandas as pd
import numpy as np
logger=logging.getLogger(__name__)

class MovementLoader:
    
    def load_movement_data(self):
        raise NotImplementedError
        
        
try:
    from movement.io import load_poses
except ModuleNotFoundError:
    pass
else:
    from flyhostel.data.interactions.touch_from_pose.utils import add_centroid_offset_single
    from flyhostel.data.pose.constants import legs as all_legs
    legs=[leg for leg in all_legs if "J" not in leg]




    class MovementLoader:
        KEYPOINTS=["head", "thorax", "abdomen", "proboscis"] + legs

        def __init__(self, *args, **kwargs):
            self.ds=None
            self.square_width=None
            self.dt=None
            self.framerate=None
            self.chunksize=None
            super(MovementLoader, self).__init__(*args, **kwargs)
        
        @property
        def datasetnames(self):
            raise NotImplementedError()


        def get_pose_file_h5py(self, pose_name):
            raise NotImplementedError()


        def load_centroid_data(self, cache):
            raise NotImplementedError()

        def annotate_pose_frame_number(self, ds, pose_file):

            with h5py.File(pose_file, "r") as handle:
                files=[f.decode() for f in handle["files"]]
                chunks=[int(os.path.basename(f).split(".")[0]) for f in files]
                frame_numbers=np.concatenate([
                    np.arange(chunk*self.chunksize, (chunk+1)*self.chunksize, 1)
                    for chunk in chunks
                ])
            ds = ds.assign_coords(
                frame_number=("time", frame_numbers),
            )
            return ds

        def sync_pose_to_centroids(self, ds):
            pose_frames=pd.DataFrame({"frame_number": ds["frame_number"].data})
            dt=self.dt.merge(pose_frames, on="frame_number", how="inner")
            mask = ds["frame_number"].isin(dt["frame_number"])
            ds  = ds.isel(time=mask)

            assert ds.position.shape[0]==dt.shape[0]
            # add good time measurement
            times  = dt["t"].to_numpy()
            ds = ds.assign_coords(
                time=("time", times)
            )
            return ds, dt

        def load_movement_data(self):

            self.load_centroid_data(cache="/flyhostel_data/cache")
            pose_file=self.get_pose_file_h5py(pose_name="raw")
            ds=load_poses.from_file(pose_file, source_software="SLEAP", fps=self.framerate)
            ds=self.annotate_pose_frame_number(ds, pose_file)
            ds, dt=self.sync_pose_to_centroids(ds)
                
            ds=ds.sel(keypoints=self.KEYPOINTS)

            # project to top left corner of original frame
            cx = dt["center_x"].to_numpy() - self.square_width // 2
            cy = dt["center_y"].to_numpy() - self.square_width // 2
            ds = add_centroid_offset_single(ds, cx, cy)  # now positions are absolute

            # If you want time strictly increasing (and dt might not be sorted):
            ds = ds.sortby("time")
            ds = ds.assign_coords(individuals=self.datasetnames)
            self.ds=ds

