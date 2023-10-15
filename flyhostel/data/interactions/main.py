import sqlite3
import glob
import pickle
import os.path
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .centroids import load_centroid_data, to_behavpy, sleep_annotation, time_window_length
from .bodyparts import make_absolute_pose_coordinates, find_closest_bps_parallel, bodyparts, legs
from flyhostel.data.pose.movie import make_pose_movie

root_folder = "/Users/FlySleepLab Dropbox/Data/flyhostel_data/fiftyone/FlyBehaviors/MOTIONMAPPER_DATA"



def dunder_to_slash(experiment):
    tokens = experiment.split("_")
    return tokens[0] + "/" + tokens[1] + "/" + "_".join(tokens[2:4])


# keep only interactions where the distance between animals is max mm_max mm
roi_width_mm=60
mm_max=4

class InteractionDetector:

    def __init__(self, experiment, lq_thresh=0.8, roi_width_mm=60, mm_max=4):
        self.experiment = experiment
        pickle_files, experiments = self.load_experiments(experiment)
        datasetnames = []
        settings={}
        for i, pickle_file in enumerate(pickle_files):
            with open(pickle_file, "rb") as handle:
                params = pickle.load(handle)
                settings[experiments[i]] = params
                datasetnames.extend(params["animals"]) 

        self.datasetnames=datasetnames
        self.identities = self.make_identities(datasetnames)
        self.h5s_pandas = [pd.read_hdf('%s/%s_positions.h5' % (os.environ["MOTIONMAPPER_DATA"], d), key="pose") for d in datasetnames]
        self.h5s_pandas=self.clean_bad_proboscis(self.h5s_pandas, lq_thresh)
        index_pandas = [pd.read_hdf('%s/%s_positions.h5' % (os.environ["MOTIONMAPPER_DATA"], d), key="index") for d in datasetnames]
        metadata= pd.concat(index_pandas)
        metadata["animal"] = list(itertools.chain(*[[datasetnames[i],]*index_pandas[i].shape[0] for i in range(len(index_pandas))]))
        metadata["index"]=metadata["frame_number"]
        metadata.set_index("index", inplace=True)
        self.index_pandas=np.split(metadata, np.cumsum(metadata.groupby("animal").count()["frame_number"])[:-1])
        del metadata
        self.basedir = os.environ["FLYHOSTEL_VIDEOS"] + "/" + dunder_to_slash(experiment)
        self.dbfile = glob.glob(self.basedir + "/FlyHostel*.db")[0]
        self.roi_width_mm=roi_width_mm
        self.mm_max=mm_max

        self.roi_width = self.load_roi_width(self.dbfile)
        self.px_per_mm=self.roi_width/roi_width_mm
        self.dist_max=self.px_per_mm*self.mm_max


        self.dt=None
        self.dt_sleep=None
        self.dt2_absolute=None
        self.out=None
        self.out_filtered=None


    def pipeline1(self):


        print("Loading centroid data")
        # NOTE
        # If self.dt is small (few rows) then the code downstream starts to break
        self.dt = self.load_centroid_data(self.experiment)
        print("Annotating sleep")
        self.dt_sleep = self.annotate_sleep()
        print("Merging centroid and pose data")
        self.pose = self.simplify_h5s_index()#.iloc[:1000]
        pose_and_centroid_clean = self.merge_centroid_and_pose(self.dt, self.pose)
        dt2 = to_behavpy(data = pose_and_centroid_clean, meta = self.dt.meta, check = True)
        
        print("Projecting to absolute coordinates")
        self.dt2_absolute = make_absolute_pose_coordinates(dt2, bodyparts, roi_width=self.roi_width)
        

    def make_movie(self, ts=None, frame_numbers=None, **kwargs):
        return make_pose_movie(self.basedir, self.dt2_absolute, ts=ts, frame_numbres=frame_numbers, **kwargs)


    def annotate_sleep(self):

        dt_to_annotate = self.dt[["t", "xy_dist_log10x1000", "frame_number", "x", "y", "w", "h", "phi"]]
        dt_sleep = dt_to_annotate.groupby(dt_to_annotate.index).apply(sleep_annotation).reset_index()
        dt_sleep.set_index("id", inplace=True)
        dt_sleep = to_behavpy(data = dt_sleep, meta = self.dt.meta, check = True)
        return dt_sleep

    def pipeline2(self, n_jobs=-2, dist_max=None, bodyparts=legs):

        if dist_max is None:
            dist_max=self.dist_max
        
        print("Filling time index")
        dt = self.fill_time_index()
        
        dt.sort_values(["t", "id"], inplace=True)
        dt = to_behavpy(data = dt, meta = self.dt.meta, check = True)

        print("Computing distance between animals")
        distance = self.compute_distance_between_centroids(dt)

        print("Keeping only close animals")
        interactions=distance.loc[
            distance["distance"]<dist_max
        ]
        # combine information about the distance between centroids (interactions)
        # and the coordinates of the bps (dt3_reset)
        dt_reset = dt.reset_index()
        interactions_complete = interactions.merge(dt_reset, how="left", on=["frame_number"])
        # interactions contains 2 rows per animal pair (identity of id1 and id2 swap)
        # we want to instead keep one row for each (because which one is which has no meaning)
        interactions_complete=interactions_complete.loc[interactions_complete["id1"] == interactions_complete["id"]]
        
        self.out, self.out_filtered = find_closest_bps_parallel(interactions_complete, n_jobs=n_jobs, chunksize=100, max_distance=10, parts=bodyparts)
        
        m2 = self.merge_with_sleep_status()

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

        m2=keep_at_least_one_fly_sleep(m2)
        m3=keep_max_one_fly_asleep(m2)
        m3.sort_values("distance", inplace=True)
        self.m3=m3
        self.m4=self.annotate_bouts(m3)


    @staticmethod
    def annotate_bouts(dt): 
        dt=dt.loc[dt["distance"] < 5]
        dt["pair"] = list(zip([e[1] for e in dt["interaction"]], [e[2] for e in dt["interaction"]]))
        m4=[]
        for key, df in dt.groupby("pair"):
            df.sort_values("t", inplace=True)
            df["new"]=np.concatenate([
                np.array([True]), np.diff(df["t"]) > 3
            ])
            df["count"] = df["new"].cumsum()
            del df["new"]
            df3=[]
            for key2, df2 in df.groupby("count"):
                df2["length"]=df2.shape[0]
                df2=df2.iloc[:1]
                df3.append(df2)
            df3=pd.concat(df3)
            m4.append(df3)
        m4=pd.concat(m4, axis=0)
        m4.sort_values("distance", inplace=True)
        return m4

    def merge_with_sleep_status(self):
        framerate=150

        dt_sleep_to_merge=self.dt_sleep.reset_index()[["id", "asleep", "frame_number"]]
        dt_sleep_to_merge=dt_sleep_to_merge.loc[~np.isnan(dt_sleep_to_merge["frame_number"])]
        dt_sleep_to_merge.sort_values(["frame_number", "id"], inplace=True)
        self.out_filtered.sort_values(["frame_number", "id1", "id2"], inplace=True)
        m1=pd.merge_asof(
            self.out_filtered.rename({"id1": "id"}, axis=1),
            dt_sleep_to_merge.rename({"asleep": "asleep1"}, axis=1),
            on="frame_number", by=["id"], direction="backward", tolerance=time_window_length*framerate
        ).rename({"id": "id1"}, axis=1)

        m2=pd.merge_asof(
            m1.rename({"id2": "id"}, axis=1),
            dt_sleep_to_merge.rename({"asleep": "asleep2"}, axis=1),
            on="frame_number", by=["id"], direction="backward", tolerance=time_window_length*framerate
        ).rename({"id": "id2"}, axis=1)

        return m2



    def compute_distance_between_centroids(self, dt):
        # compute distance between animal centroids
        x0=np.repeat(dt["centroid_x"], 6)
        y0=np.repeat(dt["centroid_y"], 6)

        out=[]
        for i in range(0, dt.shape[0], 6):
            out.extend(
                np.tile(dt["centroid_x"][i:(i+6)], 6)
            )
        x1=np.array(out)

        out=[]
        for i in range(0, dt.shape[0], 6):
            out.extend(
                np.tile(dt["centroid_y"][i:(i+6)], 6)
            )
        y1=np.array(out)
        distance=pd.DataFrame(np.sqrt((x0-x1)**2 + (y0-y1)**2))
        distance.reset_index(inplace=True)
        distance["id1"]=distance["id"]
        distance.insert(loc=2, column="frame_number", value=np.repeat(dt["frame_number"].values, 6))
        del distance["id"]
        distance["id2"] = np.array(list(itertools.chain(*[self.identities,]* (distance.shape[0]//6))))
        distance.columns=("distance", "frame_number", "id1", "id2")
        distance=distance.loc[~np.isnan(distance["distance"])]

        # verify that any distance between the same animal is always 0
        # otherwise there is a bug for sure
        assert np.all(distance.loc[distance["id1"] == distance["id2"]]["distance"] == 0)
        # remove such rows
        distance=distance.loc[~(distance["id1"] == distance["id2"])]
        distance["pair"] = [sorted(e) for e in zip(distance["id1"], distance["id2"])]
        distance["interaction"] = list(zip(distance["frame_number"], [e[0] for e in distance["pair"]],  [e[1] for e in distance["pair"]]))
        return distance

        
    def fill_time_index(self):
        # make sure a row is present for every timepoint and animal
        # some animals may be missing in some timepoints
        meta=self.dt2_absolute.meta
        dt=self.dt2_absolute.reset_index()
        n_rows_original = dt.shape[0]
        time_index = dt["t"].unique()
        time_index=pd.concat([pd.DataFrame({"t": time_index, "id": id}) for id in self.identities])
        dt=time_index.merge(dt, on=["t", "id"], how="left")
        added_fraction=1 - n_rows_original / dt.shape[0]
        print(f"{round(added_fraction*100, 2)} % of animal-frames were missing")
        return dt

    def main(self):
        self.pipeline1()
        self.make_movie(ts=np.arange(60000, 60300, 1))
        self.pipeline2()


    @staticmethod
    def merge_centroid_and_pose(dt, pose):

        # merge centroid and pose data
        dt_reset=dt.reset_index()
        dt_reset.sort_values("frame_number", inplace=True)

        missing_pose=pd.isna(pose["thorax_x"])
        pose_complete = pose.loc[~missing_pose]
        pose_complete.loc[pd.isna(pose_complete["frame_number"])]
        pose_complete.sort_values(["frame_number"], inplace=True)
        assert np.mean(np.diff(pose_complete["frame_number"])>=0) == 1
        assert np.mean(np.diff(dt_reset["frame_number"])>=0) == 1
        assert dt_reset.iloc[:-1].loc[np.diff(dt_reset["frame_number"])<0].shape[0] == 0
        dt_reset=dt_reset.loc[~pd.isna(dt_reset["frame_number"])]

        pose_and_centroid=pose_complete.merge(
            dt_reset.drop("t", axis=1), left_on=["frame_number", "id"], right_on=["frame_number", "id"],
        )
        # check that no pose data comes with missing centroid data (which would make no sense)
        assert pose_and_centroid.iloc[
            np.where(np.isnan(pose_and_centroid["x"]))
        ].shape[0] / pose_and_centroid.shape[0] == 0, f"Pose and centroid data are not consistent. Pose found but centroid not found, which cannot be because pose needs centroid"
        # line below is not needed if above assertion is ok
        pose_and_centroid_clean=pose_and_centroid.loc[~np.isnan(pose_and_centroid["x"]).values]
        return pose_and_centroid_clean

    def simplify_h5s_index(self):
        bp="thorax"
        xs=[]
        for animal_id, animal in enumerate(self.datasetnames):
            bps=np.unique(self.h5s_pandas[animal_id].columns.get_level_values(1).values)
            x=self.h5s_pandas[animal_id].loc[:, pd.IndexSlice[:, bps, ["x","y", "is_interpolated"]]]
            x.columns=itertools.chain(*[[bp + "_x", bp + "_y", bp + "_is_interpolated"] for bp in bps])
            x=x[itertools.chain(*[[bp + "_x", bp + "_y", bp + "_is_interpolated"] for bp in bodyparts])]
            x=x.merge(self.index_pandas[animal_id][["frame_number", "zt"]].set_index("frame_number"), left_index=True, right_index=True)
            x["t"]=x["zt"]*3600
            del x["zt"]
            x.insert(0, "t", x.pop("t"))
            x["id"] = self.identities[animal_id]
            x.insert(0, "id", x.pop("id"))
            
            xs.append(x)
            
        pose=pd.concat(xs, axis=0)
        pose.reset_index(inplace=True)
        return pose
    

    @staticmethod
    def load_centroid_data(*args, **kwargs):
        return load_centroid_data(*args, **kwargs)

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
    def load_roi_width(dbfile):
        with sqlite3.connect(dbfile) as conn:
            cursor=conn.cursor()

            cursor.execute(
                """
            SELECT w FROM ROI_MAP;
            """
            )
            [(roi_width,)] = cursor.fetchall()
            cursor.execute(
                """
            SELECT h FROM ROI_MAP;
            """
            )
            [(roi_height,)] = cursor.fetchall()

        roi_width=int(roi_width)
        roi_height=int(roi_height)
        roi_width=max(roi_width, roi_height)
        return roi_width


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