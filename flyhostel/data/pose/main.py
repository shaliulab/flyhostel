from abc import ABC, abstractmethod
import sqlite3
import glob
import time
import pickle
import os.path
import itertools
import logging
import pandas as pd
import numpy as np
import codetiming
import matplotlib.pyplot as plt
import h5py
from tqdm.auto import tqdm
try:
    import cupy as cp
    useGPU=True
except:
    cp=None
    useGPU=False

from flyhostel.data.interactions.centroids import (
    load_centroid_data, to_behavpy,
    flyhostel_sleep_annotation,
    time_window_length,
)
from flyhostel.data.interactions.distance import (
    compute_distance_matrix,
    compute_distance_matrix_bodyparts
)
from flyhostel.data.pose.h5py import (
    load_pose_data_processed,
    load_pose_data_compiled,
)
from flyhostel.data.pose.pose import FilterPose
from flyhostel.data.pose.filters import impute_proboscis_to_head
from flyhostel.data.interactions.bouts import annotate_interaction_bouts, compute_bouts_, DEFAULT_STRIDE
from flyhostel.data.bodyparts import make_absolute_pose_coordinates, legs
from flyhostel.data.bodyparts import bodyparts as BODYPARTS
from flyhostel.utils import load_roi_width, load_metadata_prop
from flyhostel.data.pose.movie import make_pose_movie
from flyhostel.data.pose.gpu_filters import filter_pose_df, PARTITION_SIZE

from imgstore.interface import VideoCapture

bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in BODYPARTS]))
bodyparts_speed=list(itertools.chain(*[[bp + "_speed"] for bp in BODYPARTS]))

logger = logging.getLogger(__name__)

def dunder_to_slash(experiment):
    tokens = experiment.split("_")
    return tokens[0] + "/" + tokens[1] + "/" + "_".join(tokens[2:4])


# keep only interactions where the distance between animals is max mm_max mm
roi_width_mm=60
dist_max_mm=4

MOTIONMAPPER_DATA=os.environ["MOTIONMAPPER_DATA"]
POSE_DATA=os.environ["POSE_DATA"]


from flyhostel.data.pose.filters import filter_pose, filter_pose_far_from_median, interpolate_pose, arr2df
from flyhostel.data.pose.sleap import draw_video_row

from flyhostel.data.interactions.mmpy_umap import UMAPLoader
from flyhostel.data.pose.fh_umap import PreprocessUMAP
from flyhostel.data.deg import DEGLoader
from flyhostel.data.pose.video_crosser import CrossVideo

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

    video_chunksize=45000
    video_framerate=150
    MAX_DISTANCE_HEAD_PROB_PIXELS=20

    def __init__(self, *args, **kwargs):

        self.experiment=None
        self.pose=None
        self.identities=None
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
            if not id in self.identities:
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
    
    

class FlyHostelLoader(PEDetector, CrossVideo, DEGLoader, PreprocessUMAP, FilterPose):
    """
    Analyse microbehavior produced in the flyhostel

    experiment="FlyHostelX_XX_XX-XX-XX_XX-XX-XX"

    loader = FlyHostelLoader(experiment, n_jobs=20)
    # n_jobs simply controls how many processes to use in parallel when loading idtrackerai (centroid) data

    # loads centroid data (idtrackerai) and pose data (SLEAP)
    loader.load_data(min_time=14*3600, max_time=22*3600, time_system="zt")
    # populates loader.dt (centroid) and loader.pose (pose)

    # quantifies bouts of proboscis extension
    loader.detect_proboscis_extension(self.pose)
    # output is saved in loader.pe_df
    
    # annotate sleep using centroid data
    loader.dt_sleep = loader.annotate_sleep(loader.dt)
    # output is saved in loader.dt_sleep (original framerate)
    # and loader.dt_sleep_2fps

    ## annotate interactions between flies and keep track of which body part was used
    # connect pose and centroid
    loader.integrate(self.dt, self.pose_boxcar)
    # pre-filter frames so only frames where at least two animals are at < 3 mm of each other are kept
    loader.compute_pairwise_distances(dist_max=3)

    # now on this subset, compute the interfly body pair distance
    # find the minimum distance between 2 bodyparts of different flies
    # and require it to be less than 2 mm for 3 seconds
    loader.annotate_interactions(dist_max=2, min_bout=3)

    # output is saved in
    loader.interactions_sleep


    # to load DEG human made labels (ground_truth)
    
    # if identity is None, all available identities in the loader are loaded
    loader.load_deg_data(identity=None)
    # now loader.deg is populated

    """

    def __init__(self, experiment, *args, lq_thresh=1, roi_width_mm=roi_width_mm, dist_max_mm=dist_max_mm, n_jobs=-2, pose_source="compiled", chunks=None, **kwargs):
        super(FlyHostelLoader, self).__init__(*args, **kwargs)

        basedir = os.environ["FLYHOSTEL_VIDEOS"] + "/" + dunder_to_slash(experiment)
        if not os.path.exists(basedir):
            dirs=glob.glob(basedir + "*")
            if len(dirs) == 1:
                basedir = dirs[0]
            else:
                raise Exception(f"{basedir} not found")
        self.basedir=basedir


        self.experiment = experiment
        self.lq_thresh = lq_thresh
        self.n_jobs=n_jobs
        self.pose_source=pose_source
        self.datasetnames=self.load_datasetnames(source=pose_source)
        self.identities = self.make_identities(self.datasetnames)
        self.dbfile = self.load_dbfile()
        self.store_path=os.path.join(os.path.dirname(self.dbfile), "metadata.yaml")
        self.store=None
        self.store_index=None
        self.chunks=chunks

        self.roi_width = load_roi_width(self.dbfile)
        self.framerate= int(float(load_metadata_prop(dbfile=self.dbfile, prop="framerate")))
        self.roi_width_mm=roi_width_mm
        self.px_per_mm=self.roi_width/roi_width_mm
        self.dist_max_mm=dist_max_mm

        self.index_pandas = []
        self.h5s_pandas = []
        self.pose=None
    
        self.dt=None
        self.dt_sleep=None
        self.dt_with_pose=None
        self.out=None
        self.out_filtered=None
        self.pose_and_centroid=None
        self.m3=None
        self.dt_sleep_raw=None
        self.dt_with_pose_nns=None
        self.distances=None
        self.distances_sleep=None
        self.interactions_sleep=None
        self.rejections=None
        self.dt_sleep_2fps=None
    
        self.pose_raw=None
        self.pose_jumps=None
        self.pose_filters=None
        self.pose_interpolated=None
        self.pose_speed=None
        self.pose_speed_max=None
        self.pose_annotated=None
        self.pose_speed_boxcar=None
        self.pose_boxcar_2=None

        
        for index in self.index_pandas:
            index["predictions"]=[e.decode().replace(".h5", ".slp") for e in index["files"]]
            index["video"]=[e.decode().replace(".predictions.h5", "") for e in index["files"]]


        self.meta_info=None
        self.window_size_seconds=0.5
        self.min_window_size=40
        self.min_supporting_points=3
    
    @property
    def pose_boxcar(self):
        return self.pose_filters["nanmean"]

    @property
    def pose_median(self):
        return self.pose_filters["nanmedian"]

    def list_ids(self):
        return np.unique(self.pose["id"])

    def export(self, pose=None, dest_folder=None, bodyparts=BODYPARTS, id=None):
        f"""
        Export the pose to an .h5 dataset compatible with motionmapper, bsoid, etc
        If the pose dataset is not specified, it will use pose_interpolated by default
        If the dest_folder is not specified, it will use {self.root} by default
        """

        bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))
        node_names=[bp.encode() for bp in bodyparts]

        if pose is None:
            assert self.pose_interpolated is not None
            if id is not None:
                pose_out=self.pose_interpolated.loc[self.pose_interpolated["id"]==id]
            else:
                pose_out=self.pose_interpolated
        else:
            if id is not None:
                pose_out=pose.loc[pose["id"]==id]
            else:
                pose_out=pose

        id=pose_out["id"].iloc[0]

        assert len(np.unique(pose_out["id"]))==1, f"Exported dataset contains more than 1 id. Please filter it or provide an id to .export()"
        identity = int(id.split("|")[1])

        input_array=pose_out[bodyparts_xy].values
        N, m=input_array.shape
        number_of_bodyparts = m // 2
        # Reshape and transpose the array
        reshaped_array = input_array.reshape(N, number_of_bodyparts, 2).transpose(2, 1, 0)
        reshaped_array=reshaped_array[np.newaxis, :]

        files=[e.decode() for e in self.index_pandas[identity-1]["files"].values]
        files=sorted(list(set(files)))
        
        key=f"{self.experiment}__{str(identity).zfill(2)}"

        if dest_folder is None:
            dest_folder=self.root

        folder=f"{dest_folder}/{key}"
        os.makedirs(folder, exist_ok=True)
        filename=f"{folder}/{key}.h5"
        print(f"Saving to --> {filename}")
        f=h5py.File(filename, "w")
        f.create_dataset("tracks", data=reshaped_array)
        f.create_dataset("node_names", data=node_names)
        f.create_dataset("files", data=files)
        f.close()
        return


    def load_dbfile(self):
        dbfiles=glob.glob(self.basedir + "/FlyHostel*.db")
        assert len(dbfiles) == 1, f"{len(dbfiles)} dbfiles found in {self.basedir}: {' '.join(dbfiles)}"
        return dbfiles[0]


    def load_datasetnames(self, source):
        if source == "processed":
            pickle_files, experiments = self.load_experiments(self.experiment + "*")
            datasetnames = []
            settings={}
            for i, pickle_file in enumerate(pickle_files):
                with open(pickle_file, "rb") as handle:
                    params = pickle.load(handle)
                    settings[experiments[i]] = params
                    datasetnames.extend(params["animals"])
        elif source == "compiled":
            animals=os.listdir(POSE_DATA)
            datasetnames=list(filter(lambda animals: animals.startswith(self.experiment), animals))

        assert datasetnames, f"No datasets starting with {self.experiment} found in {POSE_DATA}"
        return datasetnames
    

    def load_and_process_data(self, *args, min_time=-float('inf'), max_time=+float('inf'), stride=1, cache=None, **kwargs):
        self.load_data(min_time=min_time, max_time=max_time, stride=stride, cache=cache)
        assert self.dt is not None
        assert self.pose is not None

        # processing happens with stride = 1 and original framerate (150)
        self.process_data(*args, cache=cache, **kwargs)
        self.apply_stride_all(stride=stride)
    
    def apply_stride_all(self, stride=1):
        
        for df_name in ["pose", "pose_boxcar", "pose_speed", "pose_speed_boxcar"]:
            df=getattr(self, df_name)
            setattr(self, df_name, self.apply_stride(df, stride=stride))


    @staticmethod
    def apply_stride(df, stride):
        out=[]
        for id, df_single_animal in df.groupby("id"):
            out.append(df_single_animal.loc[df_single_animal["frame_number"] % stride == 0])
        out=pd.concat(out, axis=0)
        return out


    def process_data(self, *args, useGPU=-1, framerate=150, **kwargs):
        self.filter_and_interpolate_pose(*args, useGPU=useGPU, framerate=framerate, **kwargs)
        window_size=int(self.window_size_seconds*framerate)

        # if useGPU >= 0:
        #     pose_boxcar_2=filter_pose_df(
        #         self.pose_boxcar, f=cp.mean,
        #         columns=bodyparts_xy,
        #         window_size=window_size,
        #         partition_size=PARTITION_SIZE,
        #         pad=True,
        #         # download from the GPU back into a pd.DataFrame
        #         download=True,
        #     )

        # else:
        #     pose_boxcar_2=self.boxcar_filter(self.pose_boxcar, ["x", "y"], bodyparts=BODYPARTS)

        
        # self.pose_boxcar_2=impute_proboscis_to_head(pose_boxcar_2)
        self.compute_speed(self.pose_boxcar)
        if useGPU >= 0:
            self.pose_speed_boxcar=filter_pose_df(
                self.pose_speed, f=cp.mean, columns=bodyparts_speed,
                window_size=window_size,
                partition_size=PARTITION_SIZE,
                pad=True,
                download=True
            )
          
        else:
            self.pose_speed_boxcar=self.boxcar_filter(self.pose_speed, ["speed"], bodyparts=BODYPARTS, framerate=framerate)


    def boxcar_filter(self, pose, features, bodyparts=BODYPARTS, framerate=150):

        filtered_pose_arr, _ = filter_pose(
            "nanmean", pose, bodyparts,
            window_size=int(self.window_size_seconds*framerate),
            min_window_size=self.min_window_size,
            min_supporting_points=self.min_supporting_points,
            features=features
        )
        filtered_pose=arr2df(pose, filtered_pose_arr, bodyparts, features=features)
        return filtered_pose


    def load_store_index(self):
        if self.store_index is None:
            self.store=VideoCapture(self.store_path, 1)
            if self.chunks is not None:
                self.store_index=pd.concat([pd.DataFrame(self.store._index.get_chunk_metadata(chunk)) for chunk in self.chunks], axis=0)
            else:
                self.store_index=pd.DataFrame(self.store._index.get_all_metadata())
            self.store_index["frame_time"]/=1000
            self.store.release()


    def load_data(self, *args, min_time=-float('inf'), max_time=+float('inf'), stride=1, n_jobs=1, cache=None, **kwargs):

        if cache is not None:
            path = f"{cache}/cached_flyhostel_raw_datasets_{self.experiment}_{min_time}_{max_time}_{stride}.pkl"
            if os.path.exists(path):
                print(f"Loading ---> {path}")
                with open(path, "rb") as filehandle:
                    self.store_index, self.dt, self.meta_info, self.pose, self.h5s_pandas, self.index_pandas, _=pickle.load(filehandle)
                    self.load_deg_data(*args, identity=None, ground_truth=True, verbose=False, **kwargs)

                return
            
        self.load_store_index()


        print("Loading centroid data")
        self.load_centroid_data(*args, **kwargs, n_jobs=n_jobs, stride=stride, verbose=False)
        print("Loading pose data")
        # we always load all pose data, no matter the stride
        self.load_pose_data(*args, verbose=False, **kwargs)
        print("Loading DEG data")
        self.load_deg_data(*args, identity=None, ground_truth=True, verbose=False, **kwargs)

        
        if cache is not None:
            print(f"Caching --> {path}")
            with open(path, "wb") as filehandle:
                pickle.dump((self.store_index, self.dt, self.meta_info, self.pose, self.h5s_pandas, self.index_pandas, self.deg), filehandle)
        

    def load_deg_data(self, *args, min_time=-np.inf, max_time=+np.inf, time_system="zt", ground_truth=True,  **kwargs):
        self.deg=None
        if ground_truth:
            self.load_deg_data_gt(*args, **kwargs)
        else:
            self.load_deg_data_prediction(*args, **kwargs)

        if self.deg is not None:
            self.load_store_index()
            self.deg=self.annotate_time_in_dataset(self.deg, self.store_index, "frame_time", self.meta_info["t_after_ref"])
            self.deg=self.deg.loc[
                (self.deg["t"] >= min_time) & (self.deg["t"] < max_time)
            ]
            self.deg["behavior"].loc[pd.isna(self.deg["behavior"])]="unknown"
            self.deg=self.annotate_two_or_more_behavs_at_same_time(self.deg)



    @staticmethod
    def annotate_two_or_more_behavs_at_same_time(deg):
        """
        If more than 1 behavior is present in a given frame,
        create a new behavioral label by chaining said behaviors with +
        So for example, if the fly is walking and feeding at the same time,
        make it the behavior feed+walk
        """

        behav_index=deg[["frame_number", "id", "behavior"]].groupby(["frame_number", "id"]).count().iloc[:,:1]
        multi_behav_index=behav_index.loc[behav_index.iloc[:,0]>1].reset_index()

        deg.set_index(["frame_number", "id"], inplace=True)
        for i, row in tqdm(multi_behav_index.iterrows()):
            behaviors=deg.loc[pd.IndexSlice[row["frame_number"], row["id"]], "behavior"]
            behavior="+".join(sorted(behaviors))
            deg.loc[pd.IndexSlice[row["frame_number"], row["id"]], "behavior"]=behavior
        deg.reset_index(inplace=True)
        deg=deg.loc[~deg[["frame_number", "id"]].duplicated()]
        return deg



    @staticmethod
    def annotate_time_in_dataset(dataset, index, t_column="t", t_after_ref=None):
        assert index is not None
        assert "frame_number" in dataset.columns

        if t_column in dataset.columns:
            dataset_without_t=dataset.drop(t_column, axis=1)
        else:
            dataset_without_t=dataset
        dataset=dataset_without_t.merge(index[["frame_number", t_column]], on=["frame_number"])
        if t_after_ref is not None and t_column == "frame_time":
            dataset["t"]=dataset[t_column]+t_after_ref
        return dataset
    
    def load_pose_data(self, min_time=-np.inf, max_time=+np.inf, time_system="zt", stride=1, verbose=False):
        if self.pose_source == "processed":
            raise NotImplementedError()
            # out=load_pose_data_processed(min_time, max_time, time_system, self.datasetnames, self.identities, self.lq_thresh)
        elif self.pose_source == "compiled":
            out=load_pose_data_compiled(self.datasetnames, self.identities, self.lq_thresh, stride=stride)
        else:
            raise Exception("source must be processed or compiled")

        if out is not None:
            pose, self.h5s_pandas, self.index_pandas=out
            pose2=[]
            for d in pose:
                pose2.append(d.reset_index())
            del pose
            # one for each animal in the experiment
            self.pose=pd.concat(pose2, axis=0)
            corrupt_pose=np.bitwise_or(pd.isna(self.pose["thorax_x"]), pd.isna(self.pose["frame_number"]))

            self.pose.loc[corrupt_pose, bodyparts_xy + [bp + "_likelihood" for bp in BODYPARTS]]=np.nan
            self.pose["frame_number"]=self.pose["frame_number"].astype(np.int32)
            self.pose.sort_values(["id", "frame_number"], inplace=True)
            self.pose.reset_index(inplace=True)
            self.pose["id"]=pd.Categorical(self.pose["id"])
            self.pose["identity"]=[int(e.split("|")[1]) for e in self.pose["id"]]

            if self.dt is not None and len(self.pose) > 0:
                self.load_store_index()    
                self.pose=self.annotate_time_in_dataset(self.pose, self.store_index, "frame_time", self.meta_info["t_after_ref"])
                self.pose=self.pose.loc[
                    (self.pose["t"] >= min_time) & (self.pose["t"] < max_time)
                ]

        

    def make_movie(self, ts=None, frame_numbers=None, **kwargs):
        return make_pose_movie(self.basedir, self.dt_with_pose, ts=ts, frame_numbres=frame_numbers, **kwargs)


    def integrate(self, dt, pose):
        # NOTE
        # If self.dt is small (few rows) then the code downstream starts to break

        assert dt is not None, f"Please load centroid data"
        assert pose is not None, f"Please load pose data"
        
        print("Annotating sleep")
        self.dt_sleep = self.annotate_sleep(dt)
        print("Merging centroid and pose data")
        
        self.pose_and_centroid = self.merge_datasets(dt.drop("t", axis=1), pose, check_columns=["x"])

        # NOTE
        # Figure out why the pose is missing the first few seconds
        self.pose_and_centroid=self.pose_and_centroid.loc[~pd.isna(self.pose_and_centroid["thorax_x"])]

        print("Projecting to absolute coordinates")
        self.dt_with_pose = make_absolute_pose_coordinates(self.pose_and_centroid, bodyparts, roi_width=self.roi_width)


    def annotate_sleep(self, dt):
        """
        Annotate sleep on a downsampled timeseries (to 2 Hz)
        and bring annotation back to original framerate by interpolating
        """
        annotation_input_columns=["id", "t", "xy_dist_log10x1000", "frame_number", "x", "y", "w", "h", "phi"]
        annotation_output_columns=["id", "frame_number", "asleep", "moving", "max_velocity"]

        dt_to_annotate = dt[annotation_input_columns]
        dt_sleep = dt_to_annotate.groupby(dt_to_annotate["id"]).apply(flyhostel_sleep_annotation).reset_index()
        
        # a row full of nans is produced when no data is available for one fly in one time window
        # they need to be removed so that frame number can be integer
        dt_sleep=dt_sleep.loc[~np.isnan(dt_sleep["frame_number"])]
        dt_sleep["frame_number"]=dt_sleep["frame_number"].astype(np.int32)
                
        dt_sleep=dt_sleep[annotation_output_columns].sort_values(["frame_number", "id"])
        dt_to_merge = dt[annotation_input_columns].sort_values(["frame_number", "id"])
        
        dt_sleep=pd.merge_asof(
            dt_to_merge,
            dt_sleep,
            on="frame_number",
            by="id",
            direction="backward",
            tolerance=time_window_length*self.framerate
        )
        self.dt_sleep_2fps=dt_sleep.loc[dt_sleep["frame_number"] % (self.framerate//2) == 0]
        return dt_sleep


    def merge_datasets(self, slow_data, quick_data, tolerance=1, check_columns=None):
        """
        Combine 2 timeseries data of multiple individuals at a potentially different framerate

        Arguments:
            slow_data (pd.DataFrame): Dataset with columns id, frame_number and other columns, at a given frequency
            quick_data (pd.DataFrame): Dataset with columns id, frame_number and other columns, at a faster frequency
            tolerance (int): Number of seconds that data1 is allowed to be off from data2
            check_columns (list): If provided, the merged output will filter out all
            rows where these columns have a non assigned value (nan) 

        Returns:
            merged (pd.DataFrame): Merged dataset containing both timeseries and interpolating the slowest one
            to match the quickest one
        """

        def verify_sorted_frame_number(data):
            assert (data.groupby("id").apply(lambda x: np.mean(np.diff(x["frame_number"]) > 0)) == 1.0).all()
        # merge centroid and pose data
        # verify_sorted_frame_number(dt)
        # verify_sorted_frame_number(pose)
        if tolerance is not None:
            tolerance=int(self.framerate*tolerance)


        merged=pd.merge_asof(
            quick_data.sort_values(["frame_number", "id"]),
            slow_data.sort_values(["frame_number", "id"]),
            on="frame_number", by="id",
            tolerance=tolerance
        )

        if check_columns is not None:
            for col in check_columns:
                discard_rows=np.isnan(merged[col])
                n_discard=discard_rows.sum()
                if n_discard==0:
                    msg=logger.info
                else:
                    msg=logger.warning
                msg(f"{n_discard} rows have pose data but no centroid data")
            
                merged=merged.loc[~discard_rows.values]
        
        # NOTE
        # A perfect only merges 1 frame every second
        # because the pose has framerate 10
        # and the centroid has framerate 2
        # (which means the second frame of the centroid has no match with the pose and is discarded)
        # resulting in only 1 frame of the centroid being used -> framerate=1
        # pose_and_centroid=pose_complete.merge(
        #     dt_reset.drop("t", axis=1), left_on=["frame_number", "id"], right_on=["frame_number", "id"],
        # )
        # This is why a merge_asof is better
        # TLDR = the framerate of both timeseries is not the same and so a perfect merge entails data loss

        # check that no pose data comes with missing centroid data (which would make no sense)
        if check_columns is not None:
            for col in check_columns:
                assert merged.iloc[
                    np.where(np.isnan(merged[col]))
                ].shape[0] / merged.shape[0] == 0, f"The column check has failed in merge_datasets"
            # line below is not needed if above assertion is ok
        
        return merged


    def compute_pairwise_distances(self, dist_max=None, bodyparts=legs):

        if dist_max is None:
            dist_max=self.dist_max_mm

        assert self.dt_with_pose is not None, f"After .load_data() you still need to run .integrate(self.dt, self.pose_boxcar)"

        with codetiming.Timer():
            print("Computing distance between animals")
            # even though the pose is not needed to find the nns
            # dt_with_pose is the most raw dataset that contains the centroid_x and centroid_y
            # of the agents in absolute coordinates
            self.dt_with_pose_nns = self.find_nns(self.dt_with_pose)

            distances=self.dt_with_pose_nns[
                ["id", "nn", "distance_mm", "distance", "frame_number", "t"]
            ]

            if dist_max is not None:
                print("Keeping only close animals")
                distances=distances.loc[
                    distances["distance_mm"]<dist_max
                ]

            self.distances = self.compute_pairwise_distances_using_bodyparts(distances, self.dt_with_pose, bodyparts=bodyparts)
            self.distances_sleep = self.merge_with_sleep_status(self.dt_sleep, self.distances)


    def annotate_interactions(self, dist_max, min_bout):
        def keep_at_least_one_fly_sleep(dt):
            dt = dt.loc[
                dt["asleep1"] | dt["asleep2"]
            ]
            return dt
        def keep_max_one_fly_asleep(dt):
            dt.loc[
                ~(dt["asleep1"] & dt["asleep2"])
            ]
            return dt

        self.interactions_sleep=annotate_interaction_bouts(self.distances_sleep, dist_max, min_bout)
        if self.interactions_sleep is not None:
            self.rejections=keep_max_one_fly_asleep(keep_at_least_one_fly_sleep(self.interactions_sleep))
        else:
            self.rejections=None

    
    def compute_pairwise_distances_using_bodyparts(self, distances, pose, bodyparts, precision=100):
        """
        Compute distance between two closest bodyparts of two already close animals
        See compute_distance_matrix_bodyparts
        """

        if useGPU:
            impl = cp
        else:
            impl = np
    
        bodyparts_arr = np.array(bodyparts)
        distance_matrix, bp_pairs = compute_distance_matrix_bodyparts(distances, pose, impl, bodyparts, precision=precision)
        with codetiming.Timer(text="Computing closest body parts spent: {:.2f} seconds"):
            selected_bp_pairs = impl.argmin(distance_matrix, axis=1)
        min_distance=(distance_matrix[impl.arange(distance_matrix.shape[0]), selected_bp_pairs]/precision)

        if useGPU:
            min_distance=min_distance.get()
            bp_pairs=bp_pairs.get()
            selected_bp_pairs=selected_bp_pairs.get()
    
        pairs=np.stack([bodyparts_arr[bp_pairs[selected]] for selected in selected_bp_pairs], axis=0)
        min_distance_mm = min_distance / self.px_per_mm

        results=pd.DataFrame({"bp_distance_mm": min_distance_mm, "bp_distance": min_distance, "bp1": pairs[:, 0], "bp2": pairs[:, 1]})
        distances = pd.concat([distances.reset_index(drop=True), results], axis=1)
        distances["id"]=pd.Categorical(distances["id"])
        distances["nn"]=pd.Categorical(distances["nn"])
        distances.sort_values(["frame_number", "id", "nn"], inplace=True)
        return distances


    def merge_with_sleep_status(self, dt_sleep, distances):

        dt_sleep_to_merge=dt_sleep.reset_index()[["id", "asleep", "frame_number"]]
        distances_sleep=pd.merge_asof(
            distances,
            dt_sleep_to_merge.rename({"asleep": "asleep1"}, axis=1),
            on="frame_number", by=["id"], direction="backward", tolerance=time_window_length*self.framerate
        ).rename({"id": "id1"}, axis=1)

        distances_sleep=pd.merge_asof(
            distances_sleep.rename({"nn": "id"}, axis=1),
            dt_sleep_to_merge.rename({"asleep": "asleep2"}, axis=1),
            on="frame_number", by=["id"], direction="backward", tolerance=time_window_length*self.framerate
        ).rename({"id": "id2"}, axis=1)

        return distances_sleep


    
    def find_nns(self, dt, useGPU=True):
        """
        Annotate nearest neighbor (NN) of each agent at each timestamp

        Arguments
            dt (pd.DataFrame): Dataset with columns id, frame_number, centroid_x, centroid_y
        
        Returns
            dt_annotated (pd.DataFrame): Dataset with same columsn as input plus
                nn, distance, distance_mm
        """

        distance_matrix, identities, frame_number = compute_distance_matrix(dt, use_gpu=useGPU)
        if useGPU:
            neighbor_matrix = cp.argmin(distance_matrix, axis=1)
            min_distance_matrix = cp.min(distance_matrix, axis=1)
        else:
            neighbor_matrix = np.argmin(distance_matrix, axis=1)
            min_distance_matrix = np.min(distance_matrix, axis=1)


        nns = []
        focal_identities=identities

        for i, this_identity in enumerate(focal_identities):
            neighbors=identities.copy()
            neighbors.pop(neighbors.index(this_identity))
            this_distance=min_distance_matrix[i,:].get()
            nearest_neighbors=[neighbors[i] for i in neighbor_matrix[i,:].get()]

            out = pd.DataFrame({
                "id": this_identity,
                "nn": nearest_neighbors,
                "distance": this_distance,
                "frame_number": frame_number,
            })
            nns.append(out)
        nns=pd.concat(nns, axis=0)
        nns["distance_mm"] = nns["distance"] / self.px_per_mm
        dt_annotated = dt.merge(nns, on=["id", "frame_number"])

        return dt_annotated

    def main(self):
        self.integrate(self.dt, self.pose_boxcar)
        self.make_movie(ts=np.arange(60000, 60300, 1))
        self.annotate_interactions(n_jobs=self.n_jobs)


    def load_centroid_data(self, *args, reference_hour=None, **kwargs):
        dt, meta_info=load_centroid_data(*args, experiment=self.experiment, **kwargs)
        dt.reset_index(inplace=True)

        missing_frame_number=pd.isna(dt["frame_number"])
        missing_frame_number_count = missing_frame_number.sum()

        if missing_frame_number_count > 0:
            print(f"{missing_frame_number_count} bins are missing the frame_number")

        dt=dt.loc[~missing_frame_number]
        dt["frame_number"]=dt["frame_number"].astype(np.int32)
        dt["id"]=pd.Categorical(dt["id"])
        self.dt=dt
        self.meta_info=meta_info[0]

    def qc_plot1(self):
        bodyparts=self.h5s_pandas[0].columns.unique("bodyparts")
        lks = [pd.DataFrame({bodypart: self.h5s_pandas[i]["SLEAP"][bodypart]["likelihood"] for bodypart in bodyparts}) for i in range(len(self.h5s_pandas))]
        rejoined = [
            pd.concat(
                [
                    self.h5s_pandas[i].loc[:,  pd.IndexSlice["SLEAP", :, ["is_interpolated"]]].T.reset_index(level=2, drop=True).reset_index(level=0, drop=True).T,
                    self.index_pandas[i]
                ], axis=1)
            for i in range(len(self.h5s_pandas))
        ]

        iintp = [rejoined[i].groupby("chunk").aggregate({bp: np.mean for bp in bodyparts}) for i in range(len(self.h5s_pandas))]
        for bp in iintp[0]:
            plt.plot(iintp[0].index, iintp[0][bp])
        plt.show()
            
    def qc_plot2(self, lks):
        bodyparts=self.h5s_pandas[0].columns.unique("bodyparts")
        lks = [pd.DataFrame({bodypart: self.h5s_pandas[i]["SLEAP"][bodypart]["likelihood"] for bodypart in bodyparts}) for i in range(len(self.h5s_pandas))]
        
        for bp in lks[0]:
            if bp == "animal":
                continue
            plt.hist(lks[0][bp], bins=100)

        plt.show()
        plt.show()

    @staticmethod
    def load_experiments(pattern="*"):
        pickle_files = sorted(glob.glob(f"{MOTIONMAPPER_DATA}/{pattern}.pkl"))
        experiments = [os.path.splitext(os.path.basename(path))[0] for path in pickle_files]
        return pickle_files, experiments

    
    @staticmethod
    def make_identities(datasetnames):
        identities = [
            d[:26] +  "|" + d[-2:]
            for d in datasetnames
        ]
        return identities


    @staticmethod
    def compute_pose_distance(pose):
        diff=np.diff(pose[bodyparts_xy], axis=0)
        diff=np.vstack([[np.nan,] * diff.shape[1], diff])
        N=diff.shape[0]
        m=diff.shape[1]//2
        diff=diff.reshape(N, m, 2).transpose(2, 1, 0)
        
        nansum=np.nansum(diff**2, axis=0)
        distance=np.sqrt(nansum)
        bps = bodyparts_xy[::2]
        columns = [bp.replace("_x", "_speed") for bp in bps]
        for i, column in enumerate(columns):
            pose[column]=distance[i,:]
        return pose

    def compute_speed(self, pose):
        self.pose_speed=self.compute_pose_distance(pose)
        # pose_filtered, _ = filter_pose("nanmax", self.pose_speed, BODYPARTS, window_size=1)
        # self.pose_speed_max = arr2df(self.pose_speed, pose_filtered, BODYPARTS)


    @staticmethod
    def annotate_pose(pose, behaviors):
        if behaviors is None:
            pose_annotated=pose.copy()
            pose_annotated["behavior"]="unknown"
        else:
            pose_annotated=pose.merge(behaviors[["frame_number", "t", "id", "behavior"]], on=["frame_number", "t", "id"], how="left")
            pose_annotated.loc[pd.isna(pose_annotated["behavior"]), "behavior"]="unknown"
        return pose_annotated

    
    def draw_videos(self, index):
        for i, row in index.iterrows():
            draw_video_row(self, row["identity"], i, row, output=self.experiment + "_videos")





class InteractionDetector(FlyHostelLoader):

    def __init__(self, *args, **kwargs):
        print(f"InteractionDetector is deprecated. Please use flyhostel loader")
        super(InteractionDetector, self).__init__(*args, **kwargs)