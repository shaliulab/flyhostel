import glob
import h5py
import logging
import os.path

import numpy as np
import pandas as pd
from flyhostel.data.sleep import (
    PURE_INACTIVE_STATES,
    bin_apply_all,
    sleep_annotation_rf_all
)

from flyhostel.data.interactions.classifier.inter_orientation import calculate_angles_with_vertical_batch
from flyhostel.utils.pose_export import load_frame_numbers

logger=logging.getLogger(__name__)

class SleepLoader:
    """
    A class to load the result of the first step in the interactions pipeline
    """
    
    datasetnames=[]
    behavior=None
    sleep=None
    pixels_per_mm=None
    metadata=None
    framerate=None
    experiment=None
    square_width=None
    chunksize=None
    dt=None
    ids=[]

    def __init__(self, *args, **kwargs):
        self.interaction=None
        self.all_interactions=None
        super(SleepLoader, self).__init__(*args, **kwargs)


    def load_sleep_data(
            self,
            min_time=None,
            max_time=None,
            min_time_immobile=300,
            bin_size=300,
            errors="raise",
            complete_cases=True,

        ):

        """
        Arguments:
        
        Populates self.sleep

        Returns
            None
        """

        dataset=self.load_sleep_data_from_file(
            min_time_immobile=min_time_immobile,
            bin_size=bin_size,
        )
        if isinstance(dataset, str):
        
            feather_file=dataset
            if errors=="raise":
                raise FileNotFoundError(feather_file)
            elif errors=="warning":
                logger.warning("FileNotFound %s", feather_file)
            
            self.sleep=None
               
            return None
            
            dataset=self.compute_sleep_from_behavior(
                min_time_immobile=min_time_immobile,
                bin_size=bin_size,
            )
            if "*" not in feather_file:
                dataset.to_feather(feather_file)
        else:
            pass

        if "t_round" in dataset.columns:
            dataset=dataset\
                .drop("t", axis=1, errors="ignore")\
                .rename({"t_round": "t"}, axis=1, errors="ignore")

        assert "t" in dataset.columns
        assert "frame_number" in dataset.columns
        

        if min_time is not None:
            dataset=dataset.loc[dataset["t"]>=min_time]
        
        if max_time is not None:
            dataset=dataset.loc[dataset["t"]<max_time]

        self.sleep=dataset
        self.sleep["asleep"]=self.sleep["inactive_rule"]
        
        if complete_cases:
            fn_isna=self.sleep["frame_number"].isna()

        self.sleep["frame_number"]=self.sleep["frame_number"].astype(int)
        return None

    def annotate_frame_number_in_dataset(self, df):
        raise NotImplementedError()


    def load_sleep_data_from_file(self, min_time_immobile, bin_size):
            
        min_time_immobile_min=int(min_time_immobile//60)
        root_dir=f"/home/vibflysleep/FlySleepLab_Dropbox/Antonio/FSLLab/Projects/FlyHostel4/code/scripts/figures_wo_feed/Figure*/sleep={min_time_immobile_min}min"
      
        if bin_size is None:
            feather_file_r=os.path.join(
                root_dir,
                f"{self.datasetnames[0]}_sleep={min_time_immobile_min}.feather"
            )
        else:
            feather_file_r=os.path.join(
                root_dir,
                f"{self.datasetnames[0]}_mean_sleep={min_time_immobile_min}_bin={bin_size}.feather"
            )
        
        feather_file=glob.glob(feather_file_r)
    
        if len(feather_file)>1:
            logger.warning("More than 1 feather file detected for %s", self.datasetnames[0])
        # no hits found
        elif len(feather_file)==0:
            return feather_file_r
    
        feather_file=feather_file[0]
            
        dataset=pd.read_feather(feather_file)

        if bin_size is None and "t_round" in dataset.columns:
            del dataset["t_round"]
        else:
            dataset["t"]=dataset["t_round"]
        dataset=self.annotate_frame_number_in_dataset(dataset)
        return dataset
        

    def compute_sleep_from_behavior(self, min_time_immobile, bin_size):
        self.load_behavior_data()
        dataset=self.behavior.copy()

        
        if "inactive_states" not in dataset.columns:
            dataset["inactive_states"]=dataset["prediction2"].isin(PURE_INACTIVE_STATES)
        
        dt_sleep=sleep_annotation_rf_all(
            dataset,
            min_time_immobile=min_time_immobile,
            time_window_length=1,
            threshold=10
        )

        if bin_size is not None:
            dt_sleep=bin_apply_all(
                dt_sleep,
                feature="inactive_rule",
                summary_FUN="mean",
                x_bin_length=bin_size
            )
        return dt_sleep


    def get_pose_file_h5py(self, *args, **kwargs):
        raise NotImplementedError

    def load_centroid_data(self, *args, **kwargs):
        raise NotImplementedError


    def load_data_for_social_regression(self, meta_vars=[]):
        """
        Produce timeseries of this fly with columns id, t, frame_number, asleep, orientation, x, y, and meta_vars

        Requires sleep analysis to be done before it can be called

        x, y are in mm relative to the top left corner of the frame
        Sampling rate given by sleep dataset. Typically 1Hz (i.e. every second new data point)
        """
        self.load_centroid_data(cache="/flyhostel_data/cache")
        try:
            self.load_sleep_data(bin_size=None)
        except FileNotFoundError:
            logger.error("%s no sleep data available", self)
            return None
            
        self.sleep=self.sleep[["id", "frame_number", "asleep"]]
        path = self.get_pose_file_h5py("raw", dt=self.dt)
        
        frame_numbers=load_frame_numbers(path, self.chunksize)
        with h5py.File(path, "r") as f:
            assert "anchor" in f.keys()
            assert f["anchor"].shape[0]==f["tracks"].shape[3]
            bps=[bp.decode() for bp in f["node_names"][:]]
            head=f["tracks"][0, :, bps.index("head"), :].T
            abdomen=f["tracks"][0, :, bps.index("abdomen"), :].T
            centroids = f["anchor"][:] + self.square_width//2
            t = f["t"][:]
            assert (np.diff(t)>0).all(), "Repeated timestamps found. Did h5py corrupt some small time deltas?"
            points=np.stack([head, abdomen], axis=1)
        assert points.shape[1] == 2
        assert points.shape[2] == 2
        angle=calculate_angles_with_vertical_batch(points)
        
        df=pd.DataFrame(centroids, columns=["x", "y"])
        df/=self.pixels_per_mm
        df.insert(0, "frame_number", frame_numbers)
        df.insert(1, "t", t)
        df.insert(0, "id", self.ids[0])
        df["orientation"]=angle

        df=df.merge(self.sleep[["id", "frame_number", "asleep"]], on=["id", "frame_number"], how="left")
        df["asleep"] = df["asleep"].ffill(limit=int(self.framerate))
        assert df["asleep"].isna().mean() < 0.01
        df.insert(1, "experiment", self.experiment)

        for meta_var in meta_vars:
            df[meta_var]=self.metadata[meta_var].item()

        return df