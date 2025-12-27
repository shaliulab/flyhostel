import glob
import pandas as pd
import numpy as np
from flyhostel.data.interactions.constants import DATA_DIR
from flyhostel.data.interactions.utils import read_label_file_rejections
from flyhostel.data.pose.ethogram.utils import annotate_bouts, annotate_bout_duration

class RejectionsLoader:

    def __init__(self, *args, **kwargs):
        self.basedir=None
        self.experiment=None
        self.rejections=None
        self.rejections_gt=None
        self.framerate=None
        self.chunksize=None
        self.approximations=None
        self.interactions_index=None
        self.touch=None
        super(RejectionsLoader, self).__init__(*args, **kwargs)

    def load_interactions_index(self):
        self.interactions_index=pd.read_csv(f"{self.basedir}/interactions/{self.experiment}_interactions_v2.csv", index_col=0)
        self.interactions_index=self.interactions_index.query("id in @self.ids")

    def load_touch_database(self):
        self.touch=pd.read_feather(f"{self.basedir}/interactions/{self.experiment}_touch_database.feather")
        self.touch=self.touch.query("id in @self.ids")
        
    def load_rejections_database(self):
        self.rejections=pd.read_feather(f"{self.basedir}/interactions/{self.experiment}_rejection_database.feather")
        self.rejections=self.rejections.query("id in @self.ids")

    def load_rejections_gt(self):

        label_files=glob.glob(f"{DATA_DIR}/{self.experiment}*/*csv")
        df=[]
        for path in label_files:
            labels=read_label_file_rejections(path, chunksize=self.chunksize)
            df.append(labels)
        df=pd.concat(df, axis=0).reset_index(drop=True)
        df["chunk"]=df["frame_number"]//self.chunksize
        df["animal"]=df["data_entry"].str.slice(0, 33) + "__" + df["id"].str.slice(start=-2)

        df=annotate_bout_duration(
            annotate_bouts(df, variable="rejection")[["id", "animal", "rejection", "rejection_touch", "touch_focal", "touch_side", "data_entry", "chunk", "frame_idx", "frame_number", "bout_in", "bout_out", "bout_count"]],
            fps=self.framerate
        )

        df["touch"]=np.bitwise_or(df["touch_focal"], df["touch_side"])
        df["touch_and_rejection"]=np.bitwise_and(df["rejection"], df["touch"])
        df=df.groupby("data_entry").agg({
            "touch_and_rejection": np.sum,
            "rejection_touch": np.sum,
            "rejection": np.sum,
            "touch": np.sum,
        })
        df["framerate"]=self.framerate
        df.reset_index(inplace=True)
        df.rename({"data_entry": "key"}, axis=1, inplace=True)
        self.rejections_gt=df


    def infer_approximations(self, threshold):
        self.load_rejections_database()
        approximations=self.rejections.groupby("t").first().reset_index()
        approximations["approximation"]=approximations["distance"]>=threshold
        approximations["bout"]=approximations["t"].diff()>1
        approximations["bout"]=approximations["bout"].cumsum()
        approximations=approximations.query("approximation==True")
        stats=approximations.groupby("bout").agg({"distance": [np.min, np.argmin, len]})\
            .reset_index()
        stats.columns=["bout", "min_distance", "min_distance_i", "size"]
        assert (stats["size"]>stats["min_distance_i"]).all()

        approximations=approximations.merge(
            stats,
            on=["bout"]
        )
        self.approximations=approximations
        
    # def infer_approximations(self, threshold):

    #     approximations=self.rejections.groupby("t").first().reset_index()
    #     approximations["close"]=approximations["distance"]<threshold


    #     cols1=["min_distance", "min_distance_arg", "max_touch"]
    #     cols2=["fn_closest_approach"]
    #     approximations.drop(cols1+cols2, axis=1, errors="ignore", inplace=True)
    #     approximations["bout"]=approximations["close"].astype(int).diff()==1
    #     approximations["bout"]=approximations["bout"].cumsum()

    #     stats=approximations.groupby("bout").agg({"distance": [np.min, np.argmin], "touch": np.max})\
    #         .rename({"touch": "max_touch", "distance": "min_distance"}, axis=1)\
    #         .reset_index()
    #     stats.columns=["bout"] + cols1

    #     approximations=approximations.merge(
    #         stats,
    #         on=["bout"]
    #     )
    #     self.approximations=approximations