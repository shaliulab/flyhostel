from abc import ABC

import numpy as np
import pandas as pd
from flyhostel.data.interactions.bouts import compute_bouts_, DEFAULT_STRIDE

class PEDetector(ABC):
    """
    Given an estimate of the pose of multiple animals, stored in a pd.DataFrame
    with columns id, frame_number and bp_x bp_y bp_is_interpolated where bp is any bp,
    detects proboscis extension bouts, defined as episodes where the proboscis acquires a non-zero distance
    from the head when looking at the top view of the fly

    How to use:

    Run detect_proboscis_extension() to access the pose dataframe and run the whole analysis
    Present filters 
    * position of proboscis canot be interpolated (rows where bp_is_interpolated is False are discarded)
    * distance from proboscis to head in any frame cannot be > 20 pixels. Such frames are discarded prior to the detection of bouts
    """

    video_chunksize=CHUNKSIZE
    video_framerate=FRAMERATE
    MAX_DISTANCE_HEAD_PROB_PIXELS=20

    def __init__(self, *args, **kwargs):

        self.experiment=None
        self.pose=None
        self.ids=None
        self.roi_width=None
        self.dt=None
        self.pe_df=None
        super(PEDetector, self).__init__(*args, **kwargs)


    def compute_prob_head_distance(self, pose, min_distance=1):

        x = pose["proboscis_x"]-pose["head_x"]
        y = pose["proboscis_y"]-pose["head_y"]
        pose["frame_idx"]=pose["frame_number"]%self.video_chunksize
        pose["chunk"]=pose["frame_number"]//self.video_chunksize

        head_prob_pose = pose[["frame_number", "chunk", "frame_idx", "id", "t", "head_x", "head_y", "proboscis_x", "proboscis_y", "head_likelihood", "proboscis_likelihood", "proboscis_is_interpolated"]]

        dist = np.sqrt((np.array([x,y])**2).sum(axis=0))
        dist_df=pd.DataFrame({"frame_number": pose["frame_number"], "distance": dist})
        dist_df=dist_df.merge(head_prob_pose, on="frame_number")
        dist_df=dist_df.loc[dist_df["distance"] > min_distance]
        dist_df["mistrack"]=False
        dist_df.loc[dist_df["distance"] > self.MAX_DISTANCE_HEAD_PROB_PIXELS, "mistrack"]=True
        dist_df=dist_df.loc[~dist_df["mistrack"]]
        dist_df=dist_df.loc[~dist_df["proboscis_is_interpolated"]]
        return dist_df
    

    def compute_bouts(self, pos_events_df, stride=DEFAULT_STRIDE):
        """
        Compute length and duration of bouts of a positive event
        
        Args:

            pos_events_df (pd.DataFrame): Dataset of instances of an event that can last an undefinite amount of time
            Each row represents one frame where the event is present
            A frame_number column must be present


        """
        assert "frame_number" in pos_events_df.columns, f"Please provide a frame_numbe column"
        encoding_df, self.stride=compute_bouts_(pos_events_df)
        if self.stride != stride:
            print(f"Stride of dataset {self.experiment} is {self.stride}")

        encoding_df["chunk"]=encoding_df["frame_number"]//self.video_chunksize
        encoding_df["frame_idx"]=encoding_df["frame_number"]%self.video_chunksize
        return encoding_df


    def analyse_bouts(self, dist_df, encoding_df):
        intervals=[]
        dynamics_df=[]
        for i, row in encoding_df.iterrows():
            fn_start = row["frame_number"]-self.stride
            fn_end = row["frame_number"] + (row["length"]+1)*self.stride
            df_block = dist_df.loc[(dist_df["frame_number"]>=fn_start) & (dist_df["frame_number"]<fn_end)]
                
            distance=df_block["distance"]
            likelihood=df_block["proboscis_likelihood"]
            n_points=df_block.shape[0]
            
            dist_ts = distance.iloc[[0, n_points//2, n_points-1]]
            lk_ts = likelihood.iloc[[0, n_points//2, n_points-1]]
            dynamics_df.append(np.concatenate([dist_ts, lk_ts]))

            max_dist = distance.max()
            mean_lk =  likelihood.mean()
            min_lk =  likelihood.min()
            max_lk =  likelihood.max()
            lk_at_max_dist = likelihood.iloc[distance.argmax()]
            
            
            interval=(fn_start, fn_end, max_dist, mean_lk, min_lk, max_lk, lk_at_max_dist)
            intervals.append(interval)
            
        intervals_df = pd.DataFrame.from_records(intervals, columns=["start", "end", "max_dist", "mean_lk", "min_lk", "max_lk", "lk_at_max_dist"])
        intervals_df["chunk"]=intervals_df["start"]//self.video_chunksize
        intervals_df["frame_idx_start"]=intervals_df["start"]%self.video_chunksize
        intervals_df["frame_idx_end"]=intervals_df["end"]%self.video_chunksize
        intervals_df["duration"]=(intervals_df["end"]-intervals_df["start"])/self.video_framerate
        intervals_df=pd.concat([intervals_df, pd.DataFrame(dynamics_df)], axis=1)
        return intervals_df


    def detect_proboscis_extension(self, pose):
        all_dfs=[]
        for id, pose_d in pose.groupby("id"):
            if not id in self.ids:
                print(f"No pose data is available for id {id}")
                continue


            dist_df = self.compute_prob_head_distance(pose_d)
            bouts_df = self.compute_bouts(dist_df)
            pe_df = self.analyse_bouts(dist_df, bouts_df)
            pe_df["id"]=id
            pe_df["experiment"]=self.experiment
            pe_df["frame_number"]=pe_df["start"]

            # where?
            dt=self.dt.loc[self.dt["id"] == id]
            centroid_data=pd.DataFrame({"frame_number": dt["frame_number"].astype(np.int64), "x": self.roi_width*dt["x"], "y": self.roi_width*dt["y"]})
            pe_df["frame_number"]=pe_df["frame_number"].astype(np.int64)

            pe_df=pd.merge_asof(pe_df, centroid_data, on="frame_number")
            all_dfs.append(pe_df)

        if len(all_dfs) == 0:
            return

        pe_df=pd.concat(all_dfs)

        # when? (it's the same for all ids so it can be done outside of the loop once for all)
        pe_df=pe_df.merge(self.pose[["id", "frame_number", "t"]], on=["id", "frame_number"])
        pe_df["chunk"]=pe_df["chunk"].astype(np.int64)
        pe_df["frame_idx_start"]=pe_df["frame_idx_start"].astype(np.int64)

        self.pe_df=pe_df
        return pe_df