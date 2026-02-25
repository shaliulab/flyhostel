import shutil
import glob
import os.path
import pickle
import time
import sqlite3
import logging
import joblib
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import cudf

from flyhostel.data.interactions.detector import InteractionDetector
from flyhostel.data.synchrony.main import compute_synchrony, DEFAULT_LAGS
from flyhostel.data.hostpy import load_hostel
from flyhostel.utils.utils import (
    get_dbfile,
    get_framerate,
    get_chunksize,
    rsync_files_from,
    dunder_to_slash,
)

from flyhostel.data.human_validation.cvat.cvat_integration import (
    get_tasks_for_project,
    get_project_id_from_name,
    download_task_annotations_to_zip
)

logger=logging.getLogger(__name__)
time_counter=logging.getLogger("time_counter")

# flag to optionally create a cache.pkl file
# which contains the result of load_centroid_data
DEBUG=False

class FlyHostelGroup(InteractionDetector):

    """
    
    How to use:

    # as dictionary of FlyHostelLoaders
    group=FlyHostelGroup(loaders, dist_max_mm, min_interaction_duration)
    # as list
    group=FlyHostelGroup.from_list(loaders, dist_max_mm, min_interaction_duration)
    
    group.find_interactions(BODYPARTS_XY)

    """

    def __init__(self, flies, *args, protocol="centroids", min_time=None, max_time=None, stride=1, load_deg=True, load_behavior=True, **kwargs):
        self.flies=flies

        path="cache.pkl"
        cached_data=None

        for i, fly in enumerate(flies.values()):
            if protocol=="full":
                if fly.pose is None:
                    fly.load_and_process_data(
                        stride=stride,
                        cache="/flyhostel_data/cache",
                        min_time=min_time,
                        max_time=max_time,
                        filters=None,
                        useGPU=0,
                        load_deg=load_deg,
                        load_behavior=load_behavior,
                    )
                    fly.compile_analysis_data()
            elif protocol=="centroids":
                logger.debug("Loading %s centroids", fly)
                if DEBUG and os.path.exists(path):
                    with open("cache.pkl", "rb") as handle:
                        cached_data=pickle.load(handle)
                    fly.dt, fly.meta_info=cached_data[i]
                else:
                    fly.load_centroid_data(
                        stride=stride,
                        min_time=min_time,
                        max_time=max_time,
                        identity_table=fly.identity_table,
                        roi_0_table=fly.roi_0_table,
                    )
            elif protocol is None:
                pass
            else:
                raise ValueError("Please set protocol to one of centroids/full/None")

        if cached_data is None and DEBUG:
            cached_data=[(fly.dt, fly.meta_info) for fly in self.flies.values()]
            with open(path, "wb") as handle:
                pickle.dump(cached_data, handle)

        self.basedir=self.flies[list(self.flies.keys())[0]].basedir

        if self.group_is_real:
            self.experiment=self.flies[list(self.flies.keys())[0]].experiment
            self.number_of_animals=int(self.experiment.split("_")[1].replace("X", ""))
            self.framerate=get_framerate(self.experiment)
            self.chunksize=get_chunksize(self.experiment)
            if not self.group_is_complete:
                logger.warning("You have loaded flies from the same experiment, but at least one is missing")

        else:
            logger.warning("Virtual group of flies detected")
            self.experiment=None
            self.number_of_animals=None
            self.framerate=None
            self.chunksize=None
            
            
        super(FlyHostelGroup, self).__init__(*args, **kwargs)


    @property
    def group_is_real(self):
        return len(set([fly.basedir for fly in self.flies.values()])) == 1
    
    @property
    def group_is_complete(self):
        return len(self.flies)==self.number_of_animals



    @property
    def animals(self):
        return sorted(list(self.flies.keys()))
    
    @property
    def ids(self):
        return [self.flies[animal].ids[0] for animal in self.animals]

    def __len__(self):
        return len(self.flies)

    def compute_synchrony(self, time_window_length=1, min_time_immobile=300, source="inactive", lags=DEFAULT_LAGS):

        lags_seconds=np.array(lags)
        lags=(lags_seconds//time_window_length).tolist()

        dt_sleep=self.apply("sleep_annotation", source=source, min_time_immobile=min_time_immobile, time_window_length=time_window_length)
        return compute_synchrony(dt_sleep, lags=lags)


    def apply(self, function, *args, n_jobs=1, **kwargs):
        
        out=joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(
                getattr(fly, function)
            )(
                pd.DataFrame(fly.load_analysis_data()), *args, **kwargs
            )
            for fly in self.flies.values()
            
        )        
        return pd.concat(out, axis=0)


    def get_simple_metadata(self, flies=None):
        if flies is None:
            flies=self.flies

        return pd.concat([
            fly.get_simple_metadata()
            for fly in flies.values()
        ], axis=0)

    def load_analysis_data(self, n_jobs=1):
        #TODO Check if data is already loaded

        cached_flies={fly: self.flies[fly].analysis_data for fly in self.flies}
        flies={fly: self.flies[fly] for fly in self.flies if cached_flies[fly] is None}
        cached_data=pd.concat(
            [cached_flies[fly] for fly in cached_flies if cached_flies[fly] is not None],
            axis=0
        )

        metadata=self.get_simple_metadata(flies)
        data=load_hostel(metadata, n_jobs=n_jobs, value="dataframe")
        data=pd.concat([cached_data, data], axis=0)
        return data

    @classmethod
    def from_metadata(cls, metadata):
        return cls(flies=load_hostel(metadata, n_jobs=1, value="object"))


    @classmethod
    def from_list(cls, flies, *args, **kwargs):
        
        flies_dict={fly.datasetnames[0]: fly for fly in flies}
        return cls(flies=flies_dict, *args, **kwargs)


    def full_interpolation_all(self, pose="pose_boxcar", **kwargs):
        dfs=[]
        for fly in self.flies.values():
            df=getattr(fly, pose).copy()
            dfs.append(fly.full_interpolation(df, **kwargs))
        pose_df=pd.concat(dfs, axis=0)
        return pose_df


    def load_centroid_data(self, fps=30, useGPU=True):
        skip=max(1, self.framerate//fps)
        if useGPU:
            xf=cudf
        else:
            xf=pd
        
        dfs=[]
        for fly in self.flies.values():
            df=xf.DataFrame(fly.dt)
            df=df.loc[df["frame_number"]%skip==0]
            dfs.append(df)
        dt=xf.concat(dfs, axis=0)
        return dt

    def load_pose_data(self, pose_name="filter_rle-jump", min_time=None, max_time=None, fps=30, useGPU=True, skip=None):
        if skip is None:
            skip=max(1, self.framerate//fps)

        if useGPU:
            xf=cudf
        else:
            xf=pd

        pose=None
        for fly in tqdm(self.flies.values(), desc="Loading pose"):
            before=time.time()
            filtered_pose=fly.get_pose_file_h5py(pose_name=pose_name)
            raw_pose=fly.get_pose_file_h5py(pose_name="raw")
            files=[(filtered_pose, raw_pose)]
            fly.load_pose_data(identity=fly.identity, min_time=min_time, max_time=max_time, verbose=False, cache="/flyhostel_data/cache", files=files, write_only=False, stride=skip)
            pose_cpu=fly.pose

            after_cpu=time.time()
            time_counter.debug("load_filtered_pose in %s seconds", round(after_cpu-before, ndigits=2))
            pose_gpu=xf.DataFrame(pose_cpu)
            after_xf=time.time()
            time_counter.debug("called %s in %s seconds", xf.DataFrame, round(after_xf-after_cpu, ndigits=2))

            # TODO
            # Decide whether we keep a reference in the fly object or not
            # Cons: takes up gpu memory
            # del fly.pose

            df=pose_gpu.loc[(pose_gpu["frame_number"]%skip)==0]
            
            if pose is None:
                pose=df
            else:
                before=time.time()
                pose=xf.concat([pose, df], axis=0)
                after=time.time()
                time_counter.debug("concat in %s seconds", round(after-before, ndigits=2))
                del df
        return pose


    def load_behavior_data(self, fps=30, useGPU=True, skip=None):
        if skip is None:
            skip=max(1, self.framerate//fps)

        if useGPU:
            xf=cudf
        else:
            xf=pd
        
        dfs=[]
        for fly in self.flies.values():
            df=getattr(fly, "behavior",None)
            if df is None:
                continue

            df=xf.DataFrame(df)            
            df=df.loc[(df["frame_number"]%skip)==0]
            dfs.append(df)
        if dfs:
            behavior=xf.concat(dfs, axis=0)
        else:
            behavior=None
        return behavior


    def load_deg_data(self, fps=30, verbose=False, key="deg", skip=None, **kwargs):
        if skip is None:
            skip=max(1, self.framerate//fps)
        xf=pd
        
        dfs=[]
        for fly in self.flies.values():
            if getattr(fly, key, None) is None:
                fly.load_deg_data(ground_truth=True, stride=skip, verbose=verbose, **kwargs)
            
            if getattr(fly, key, None) is None:
                continue
            df=xf.DataFrame(getattr(fly, key))
            df=df.loc[(df["frame_number"]%skip)==0]
            dfs.append(df)
        if dfs:
            deg=xf.concat(dfs, axis=0)
        else:
            deg=None
        return deg

    
    def load_concatenation_table(self):
        dbfile=get_dbfile(self.basedir)
        concatenation=None
        with sqlite3.connect(dbfile) as conn:
            concatenation=pd.read_sql(con=conn, sql="SELECT * FROM CONCATENATION_VAL;")
        return concatenation
    

    def backup(self, path, chunks=None, dry_run=False, debug=False):

        assert self.group_is_real and self.group_is_complete
        subpath=dunder_to_slash(self.experiment)
        new_basedir=os.path.join(path, subpath)

        os.makedirs(new_basedir, exist_ok=True)
        dbfile=os.path.join(self.basedir, ".", self.experiment + ".db")
        mp4_files=sorted(glob.glob(os.path.join(self.basedir, ".", "*mp4")))
        npz_files=sorted(glob.glob(os.path.join(self.basedir, ".", "*npz")))
        png_files=sorted(glob.glob(os.path.join(self.basedir, ".", "*png")))
        metadata_file=os.path.join(self.basedir, ".", "metadata.yaml")
        landmarks_file=os.path.join(self.basedir, ".", "landmarks.toml")
        index_file=os.path.join(self.basedir, ".","index.db")
        camera_file=glob.glob(os.path.join(self.basedir, ".", "*pfs"))[0]

        date_time = "_".join(self.experiment.split("_")[2:4])
        config_files=[
            os.path.join(self.basedir, ".", f"{date_time}.conf"),
            os.path.join(self.basedir, ".", f"{date_time}.toml")
        ]

        data_files=[]
        for file in mp4_files + npz_files + png_files:
            if chunks is None:
                data_files.append(file)
            else:
                chunk=int(os.path.basename(file).split(".")[0])
                if chunk in chunks:
                    data_files.append(file)


        validation_folder=os.path.join(self.basedir, ".", "flyhostel", "validation")
        if debug:
            files=[metadata_file]
        else:
            validation_files=[]
            if self.number_of_animals > 1:
                try:
                    validation_files=self.download_annotations_from_cvat(validation_folder)
                except Exception as error:
                    logger.error(error)
                    validation_files=[]
                    status="NO_CVAT_VALIDATION_FOUND"

            # handmade
            validation_csv=os.path.join(validation_folder, "validation.csv")
            if os.path.exists(validation_csv):
                validation_files.append(validation_csv)

            files=[
                dbfile, *data_files, metadata_file, landmarks_file,
                index_file, camera_file, *config_files, *validation_files,
            ]
        
            # single file for all flies in the group
            # interactions-from-centroids
            interactions_file=os.path.join(
                self.basedir, ".", "interactions", self.experiment + "_database.feather"
            )
        
            if os.path.exists(interactions_file):
                files.append(interactions_file)

        rsync_files_from(files, new_basedir, dry_run=dry_run)

        if not debug:
            for fly in self.flies.values():
                fly.backup(new_basedir, chunks=chunks, dry_run=dry_run)
        
        return status


    def download_annotations_from_cvat(self, path):
        tasks=sorted(get_tasks_for_project(get_project_id_from_name(self.experiment, errors="raise")))
        zip_files=[]

        for task in tqdm(tasks, desc="Downloading CVAT annotations to .zip"):
            zip_files.append(download_task_annotations_to_zip(task, path = path, redownload=True))
        return zip_files