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
import cupy as cp
from .centroids import (
    load_centroid_data, to_behavpy,
    sleep_annotation,
    time_window_length,
)
from .distance import (
    compute_distance_matrix,
    compute_distance_matrix_bodyparts
)

from .bouts import annotate_interaction_bouts

from .bodyparts import make_absolute_pose_coordinates, bodyparts, legs
from .utils import load_roi_width, load_metadata_prop
from flyhostel.data.pose.movie import make_pose_movie
from ethoscopy.analyse import downsample_to_fps

logger = logging.getLogger(__name__)

root_folder = "/Users/FlySleepLab Dropbox/Data/flyhostel_data/fiftyone/FlyBehaviors/MOTIONMAPPER_DATA"

def dunder_to_slash(experiment):
    tokens = experiment.split("_")
    return tokens[0] + "/" + tokens[1] + "/" + "_".join(tokens[2:4])


# keep only interactions where the distance between animals is max mm_max mm
roi_width_mm=60
dist_max_mm=4
useGPU=True




from .load_data import get_sqlite_file

class CrossVideo(ABC):
    
    @staticmethod
    def cross_with_video_data(dt):
        dt["identity1"] = [int(e.split("|")[1]) for e in dt["id1"]]
        dt["identity2"] = [int(e.split("|")[1]) for e in dt["id2"]]
        dt["id1"].iloc[0].split("|")[0]
        dt["dbfile"]=[get_sqlite_file(animal.split("|")[0] + "*") for animal in dt["id1"]]
        dt.sort_values("bp_distance_mm", inplace=True)    
        dbfile=dt.iloc[0]["dbfile"]

        with sqlite3.connect(dbfile) as conn:
            cursor=conn.cursor()
            cursor.execute(f"SELECT value FROM METADATA WHERE field = 'chunksize';",)
            chunksize = int(float(cursor.fetchone()[0]))

        frame_numbers=[int(e) for e in sorted(dt["frame_number"].unique())]    
        table = get_local_identities(dt.iloc[0]["dbfile"], frame_numbers=frame_numbers)
        dt["frame_idx"]=dt["frame_number"] % chunksize
        dt["video_time"]=dt["frame_idx"] / 150

        dt["video1"]=None
        dt["video2"]=None

        for i, row in dt.iterrows():
            dt["video1"].loc[i]=get_single_animal_video(row["dbfile"], row["frame_number"], table=table, identity=row["identity1"], chunksize=chunksize)
            dt["video2"].loc[i]=get_single_animal_video(row["dbfile"], row["frame_number"], table=table, identity=row["identity2"], chunksize=chunksize)

        dt=dt.loc[~pd.isna(dt["video1"])]
        return dt
        # dt.sort_values("bp_distance_mm", ascending=False).to_csv(f"2023-10-24_interaction-scenes/{experiment}.csv")


import zeitgeber
class PEDetector(ABC):
    """
    Given an estimate of the pose of multiple animals, stored in a pd.DataFrame
    with columns id, frame_number and bp_x bp_y bp_is_interpolated where bp is any bp
    detects proboscis extension bouts, defined as episodes where the proboscis acquires a non-zero distance
    from the head when looking at the top view of the fly

    How to use:

    Run detect_proboscis_extension() to access the pose dataframe and run the whole analysis
    Present filters 
    * position of proboscis canot be interpolated (rows where bp_is_interpolated is False are discarded)
    * distance from proboscis to head in any frame cannot be > 20 pixels. Such frames are discarded prior to the detection of bouts
    """

    experiment=None
    pose=None
    identities=None
    roi_width=None
    dt=None
    pe_df=None
    video_chunksize=45000
    video_framerate=150
    MAX_DISTANCE_HEAD_PROB_PIXELS=20


    def compute_prob_head_distance(self, pose):


        x = pose["proboscis_x"]-pose["head_x"]
        y = pose["proboscis_y"]-pose["head_y"]
        pose["frame_idx"]=pose["frame_number"]%self.video_chunksize
        pose["chunk"]=pose["frame_number"]//self.video_chunksize

        head_prob_pose = pose[["frame_number", "chunk", "frame_idx", "id", "t", "head_x", "head_y", "proboscis_x", "proboscis_y", "head_likelihood", "proboscis_likelihood", "proboscis_is_interpolated"]]

        dist = np.sqrt((np.array([x,y])**2).sum(axis=0))
        dist_df=pd.DataFrame({"frame_number": pose["frame_number"], "distance": dist})
        dist_df=dist_df.merge(head_prob_pose, on="frame_number")
        dist_df=dist_df.loc[dist_df["distance"] > 1]
        dist_df["mistrack"]=False
        dist_df.loc[dist_df["distance"] > self.MAX_DISTANCE_HEAD_PROB_PIXELS, "mistrack"]=True
        dist_df=dist_df.loc[~dist_df["mistrack"]]
        dist_df=dist_df.loc[~dist_df["proboscis_is_interpolated"]]
        dist_df["df"]=np.concatenate([[np.inf], np.diff(dist_df["frame_number"])])
        return dist_df
    

    def compute_pe_bouts(self, dist_df):
        self.stride=np.unique(dist_df["df"], return_counts=True)[0][0]
        if self.stride != 10:
            print(f"Stride of dataset {self.experiment} is {self.stride}")

        encoding = zeitgeber.rle.encode([str(e)[0] for e in dist_df["df"]<=self.stride])
        encoding_df=pd.DataFrame.from_records(encoding, columns=["status", "length"])
        if encoding_df.shape[0]%2==1:
            encoding_df=encoding_df.iloc[:-1]
            
        encoding_df["bout"]=np.repeat(np.arange(encoding_df.shape[0]//2), 2)
        encoding_df["row"]=encoding_df["length"].cumsum()
        encoding_df["frame_number"]=dist_df["frame_number"].iloc[(encoding_df["row"]-1)].values

        encoding_df=encoding_df.groupby("bout").apply(lambda df: [df["frame_number"].iloc[0], df["length"].iloc[1]+1]).reset_index()
        encoding_df["frame_number"]=[e[0]for e in encoding_df[0]]
        encoding_df["chunk"]=encoding_df["frame_number"]//self.video_chunksize
        encoding_df["frame_idx"]=encoding_df["frame_number"]%self.video_chunksize
        encoding_df["length"]=[e[1]for e in encoding_df[0]]
        del encoding_df[0]
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
    
    def detect_proboscis_extension(self):
        all_dfs=[]
        for id in self.identities:
            if self.pose is None:
                print(f"No pose data is available for experiment {self.experiment}")
                continue
            else:
                pose=self.pose.loc[self.pose["id"]==id]
                if pose.shape[0] == 0:
                    print(f"No pose data is available for id {id}")
                    continue


            dist_df = self.compute_prob_head_distance(pose)
            bouts_df = self.compute_pe_bouts(dist_df)
            pe_df = self.analyse_bouts(dist_df, bouts_df)
            pe_df["id"]=id
            pe_df["experiment"]=self.experiment
            pe_df["frame_number"]=pe_df["start"]

            # where?
            dt=self.dt.loc[self.dt["id"] == id]
            centroid_data=pd.DataFrame({"frame_number": dt["frame_number"].astype(np.int64), "x": self.roi_width*dt["x"], "y": self.roi_width*dt["y"]})
            pe_df=pd.merge_asof(pe_df, centroid_data, on="frame_number")
            all_dfs.append(pe_df)

        if len(all_dfs) == 0:
            return

        pe_df=pd.concat(all_dfs)

        # when? (it's the same for all ids so it can be done outside of the loop once for all)
        pe_df=pe_df.merge(self.pose[["id", "frame_number", "t"]], on=["id", "frame_number"])
        self.pe_df=pe_df

        return pe_df
    
    

class InteractionDetector(PEDetector, CrossVideo):
    """
    Analyse microbehavior produced in the flyhostel

    experiment="FlyHostelX_XX_XX-XX-XX_XX-XX-XX"

    detector = InteractionDetector(experiment, n_jobs=20)
    # n_jobs simply controls how many processes to use in parallel when loading idtrackerai (centroid) data
    

    # loads centroid data (idtrackerai) and pose data (SLEAP)
    detector.load_data(min_time=14*3600, max_time=22*3600, time_system="zt")
    # populates detector.dt (centroid) and detector.pose (pose)

    # quantifies bouts of proboscis extension
    detector.detect_proboscis_extension()
    # output is saved in detector.pe_df
    
    # annotate sleep using centroid data
    detector.dt_sleep = detector.annotate_sleep(detector.dt)
    # output is saved in detector.dt_sleep (original framerate)
    # and detector.dt_sleep_2fps

    ## annotate interactions between flies and keep track of which body part was used
    # connect pose and centroid
    detector.integrate()
    # pre-filter frames so only frames where at least two animals are at < 3 mm of each other are kept
    detector.compute_pairwise_distances(dist_max=3)

    # now on this subset, compute the interfly body pair distance
    # find the minimum distance between 2 bodyparts of different flies
    # and require it to be less than 2 mm for 3 seconds
    detector.annotate_interactions(dist_max=2, min_bout=3)

    # output is saved in
    detector.interactions_sleep
    """

    def __init__(self, experiment, lq_thresh=0.8, roi_width_mm=roi_width_mm, dist_max_mm=dist_max_mm, n_jobs=-2):

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
        self.datasetnames=self.load_datasetnames()
        self.identities = self.make_identities(self.datasetnames)
        self.dbfile = self.load_dbfile()
        self.roi_width = load_roi_width(self.dbfile)
        self.framerate= int(float(load_metadata_prop(dbfile=self.dbfile, prop="framerate")))
        self.roi_width_mm=roi_width_mm
        self.px_per_mm=self.roi_width/roi_width_mm
        self.dist_max_mm=dist_max_mm

        self.index_pandas = []
        self.h5s_pandas = []
        self.pose=[]
    
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
        

    def load_dbfile(self):
        dbfiles=glob.glob(self.basedir + "/FlyHostel*.db")
        assert len(dbfiles) == 1, f"{len(dbfiles)} dbfiles found in {self.basedir}: {' '.join(dbfiles)}"
        return dbfiles[0]


    def load_datasetnames(self):
        pickle_files, experiments = self.load_experiments(self.experiment + "*")
        datasetnames = []
        settings={}
        for i, pickle_file in enumerate(pickle_files):
            with open(pickle_file, "rb") as handle:
                params = pickle.load(handle)
                settings[experiments[i]] = params
                datasetnames.extend(params["animals"])

        return datasetnames
        

    def load_data(self, *args, **kwargs):
        print("Loading centroid data")
        self.load_centroid_data(*args, **kwargs, n_jobs=self.n_jobs)
        print("Loading pose data")
        self.load_pose_data(*args, **kwargs)

    
    def load_pose_data(self, min_time=-np.inf, max_time=+np.inf, time_system="zt"):
        self.index_pandas = []
        self.h5s_pandas = []
        self.pose=[]

        for animal_id, d in enumerate(self.datasetnames):
            h5_file = '%s/%s_positions.h5' % (os.environ["MOTIONMAPPER_DATA"], d)

            if not os.path.exists(h5_file):
                print(f"{h5_file} not found")
                continue

            index = pd.read_hdf(h5_file, key="index")
            index["t"] = index["zt"]*3600

            keep_rows=np.where((index["t"] >= min_time) & (index["t"] < max_time))[0]
            first_row=keep_rows[0]
            last_row=keep_rows[-1]+1
            
            index=index.iloc[first_row:last_row]
            pose=pd.read_hdf(h5_file, key="pose", start=first_row, stop=last_row)
            pose=self.clean_bad_proboscis([pose], self.lq_thresh)[0]
            index["animal"]=d
            index["index"]=index["frame_number"]
            index.set_index("index", inplace=True)
            
            self.h5s_pandas.append(pose)

            self.pose.append(self._simplify_columns(index, pose, self.identities[animal_id]))
            self.index_pandas.append(index)
        
        if len(self.pose) == 0:
            self.pose=None
            return

        self.pose=pd.concat(self.pose, axis=0)
        self.pose.reset_index(inplace=True)

        missing_pose=np.bitwise_or(pd.isna(self.pose["thorax_x"]), pd.isna(self.pose["frame_number"]))
        self.pose = self.pose.loc[~missing_pose]
        self.pose["frame_number"]=self.pose["frame_number"].astype(np.int32)
        self.pose.sort_values(["frame_number"], inplace=True)
        self.pose.reset_index(inplace=True)
        self.pose["id"]=pd.Categorical(self.pose["id"])
        

    @staticmethod
    def _simplify_columns(index, pose, id):
        bp="thorax"
        bps=np.unique(pose.columns.get_level_values(1).values)
        pose=pose.loc[:, pd.IndexSlice[:, bps, ["x","y", "is_interpolated", "likelihood"]]]
        pose.columns=itertools.chain(*[[bp + "_x", bp + "_y", bp + "_is_interpolated", bp + "_likelihood"] for bp in bps])
        pose=pose[itertools.chain(*[[bp + "_x", bp + "_y", bp + "_is_interpolated", bp + "_likelihood"] for bp in bodyparts])]
        pose=pose.merge(index[["frame_number", "zt"]].set_index("frame_number"), left_index=True, right_index=True)
        pose["t"]=pose["zt"]*3600
        del pose["zt"]
        pose.insert(0, "t", pose.pop("t"))
        pose.insert(0, "id", id)
        return pose
        

    def make_movie(self, ts=None, frame_numbers=None, **kwargs):
        return make_pose_movie(self.basedir, self.dt_with_pose, ts=ts, frame_numbres=frame_numbers, **kwargs)


    def integrate(self):
        # NOTE
        # If self.dt is small (few rows) then the code downstream starts to break

        assert self.dt is not None, f"Please load centroid data"
        assert self.pose is not None, f"Please load pose data"
        
        print("Annotating sleep")
        self.dt_sleep = self.annotate_sleep(self.dt)
        print("Merging centroid and pose data")
        
        self.pose_and_centroid = self.merge_datasets(self.dt.drop("t", axis=1), self.pose, check_columns=["x"])

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
        dt_sleep = dt_to_annotate.groupby(dt_to_annotate["id"]).apply(sleep_annotation).reset_index()
        
        # a row full of nans is produced when no data is available for one fly in one time window
        # they need to be removed so that frame number can be integer
        dt_sleep=dt_sleep.loc[~np.isnan(dt_sleep["frame_number"])]
        dt_sleep["frame_number"]=dt_sleep["frame_number"].astype(np.int32)
                
        dt_sleep=dt_sleep[annotation_output_columns].sort_values(["frame_number", "id"])
        dt_to_merge = dt[annotation_input_columns].sort_values(["frame_number", "id"])
        
        self.dt_sleep_2fps=dt_sleep.copy()
        dt_sleep=pd.merge_asof(
            dt_to_merge,
            dt_sleep,
            on="frame_number",
            by="id",
            direction="backward",
            tolerance=time_window_length*self.framerate
        )
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
        self.rejections=keep_max_one_fly_asleep(keep_at_least_one_fly_sleep(self.interactions_sleep))

    
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
        neighbor_matrix = cp.argmin(distance_matrix, axis=1)
        min_distance_matrix = cp.min(distance_matrix, axis=1)

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
        self.integrate()
        self.make_movie(ts=np.arange(60000, 60300, 1))
        self.annotate_interactions(n_jobs=self.n_jobs)


    def load_centroid_data(self, *args, **kwargs):
        dt=load_centroid_data(*args, experiment=self.experiment, **kwargs).reset_index()
        missing_frame_number=pd.isna(dt["frame_number"])
        missing_frame_number_count = missing_frame_number.sum()

        if missing_frame_number_count > 0:
            print(f"{missing_frame_number_count} bins are missing the frame_number")

        dt=dt.loc[~missing_frame_number]
        dt["frame_number"]=dt["frame_number"].astype(np.int32)
        dt["id"]=pd.Categorical(dt["id"])
        self.dt=dt


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
        pickle_files = sorted(glob.glob(f"{root_folder}/{pattern}.pkl"))
        experiments = [os.path.splitext(os.path.basename(path))[0] for path in pickle_files]
        return pickle_files, experiments


    @staticmethod
    def clean_bad_proboscis(h5s_pandas, threshold):
        """
        If the score of the proboscis is too low
        the position is ignored
        and instead is set to be on the head
        is_interpolated in this case becomes True
        """
        for i, h5 in enumerate(h5s_pandas):
            bad_quality_rows=(h5.loc[:, pd.IndexSlice[:, ["proboscis"], "likelihood"]] < threshold).values.flatten()
            h5.loc[bad_quality_rows, pd.IndexSlice[:, "proboscis", "x"]]=h5.loc[bad_quality_rows, pd.IndexSlice[:, "head", "x"]]
            h5.loc[bad_quality_rows, pd.IndexSlice[:, "proboscis", "y"]]=h5.loc[bad_quality_rows, pd.IndexSlice[:, "head", "y"]]
            h5.loc[bad_quality_rows, pd.IndexSlice[:, "proboscis", "likelihood"]]=h5.loc[bad_quality_rows, pd.IndexSlice[:, "head", "likelihood"]]
            h5.loc[bad_quality_rows, pd.IndexSlice[:, "proboscis", "is_interpolated"]]=True
            h5s_pandas[i]=h5
        
        return h5s_pandas
    
    @staticmethod
    def make_identities(datasetnames):
        identities = [
            d[:26] +  "|" + d[-2:]
            for d in datasetnames
        ]
        return identities
    


def get_local_identities(dbfile, frame_numbers):

    with sqlite3.connect(dbfile) as conn:
        cursor = conn.cursor()
        query = "SELECT frame_number, identity, local_identity FROM identity WHERE frame_number IN ({})".format(
            ','.join(['?'] * len(frame_numbers))
        )
        cursor.execute(query, frame_numbers)
        
        table = cursor.fetchall()
    
    table=pd.DataFrame.from_records(table, columns=["frame_number", "identity", "local_identity"])
    return table

def get_single_animal_video(dbfile, frame_number, table, identity, chunksize):
    chunk = frame_number // chunksize
    table_current_frame = table.loc[(table["frame_number"] == frame_number)]
    if (table_current_frame["local_identity"] == 0).any():
        single_animal_video=None    
    else:
        local_identity = table_current_frame.loc[table_current_frame["identity"] == identity, "local_identity"]
        if local_identity.shape[0] == 0:
            single_animal_video=None
        else:
            local_identity=local_identity.item()
            single_animal_video = os.path.join(os.path.dirname(dbfile), "flyhostel", "single_animal", str(local_identity).zfill(3), str(chunk).zfill(6) + ".mp4")
    
    return single_animal_video
