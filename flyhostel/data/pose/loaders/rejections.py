import glob
import pandas as pd
import numpy as np
from flyhostel.data.interactions.constants import DATA_DIR
from flyhostel.data.interactions.utils import read_label_file_rejections
from flyhostel.data.pose.ethogram.utils import annotate_bouts, annotate_bout_duration


def approximation_hysteresis(distance, touch, *, min_th, max_th, eps):
    """
    Returns array of {0,1,2} with hysteresis to prevent chatter.
    eps sets the hysteresis half-width around thresholds.
    """
    d = np.asarray(distance, dtype=float)
    t = np.asarray(touch, dtype=bool)

    # Enter/exit thresholds
    min_in  = min_th - eps
    min_out = min_th + eps
    max_in  = max_th + eps
    max_out = max_th - eps

    out = np.empty(len(d), dtype=np.int8)
    state = 2  # default initial state; you can set based on d[0] if you prefer

    for i, _ in enumerate(d):

        # if touching or very close (= 0)
        if t[i] or d[i] < min_in:
            state = 0
        
        # if prior was very close (= 0)
        elif state == 0:
            # stay 0 until we truly leave the close zone
            if d[i] > min_out and not t[i]:
                state = 1
        elif state == 2:
            # stay 2 until we truly leave the far zone
            if d[i] < max_out:
                state = 1
        else:  # state == 1
            if d[i] >= max_in:
                state = 2
            # else remain 1

        out[i] = state

    # Touch always wins (ensures no 1/2 while touching)
    out[t] = 0
    return out



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
        self.pixels_per_mm=None
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
            annotate_bouts(df, variable="rejection")[["id", "animal", "reaction", "rejection", "rejection_touch", "touch_focal", "touch_side", "data_entry", "chunk", "frame_idx", "frame_number", "bout_in", "bout_out", "bout_count"]],
            fps=self.framerate
        )

        df["touch"]=np.bitwise_or(df["touch_focal"], df["touch_side"])
        df["responder"]=np.bitwise_or(np.bitwise_and(df["touch_focal"], ~df["rejection_touch"]), df["reaction"])
        df["touch_and_rejection"]=np.bitwise_and(df["rejection"], df["touch"])
        df_summ=df.groupby("data_entry").agg({
            "touch_and_rejection": np.sum,
            "rejection_touch": np.sum,
            "rejection": np.sum,
            "touch": np.sum,
            "responder": np.sum,
        })

        df_summ["framerate"]=self.framerate
        df_summ.reset_index(inplace=True)
        df_summ.rename({"data_entry": "key"}, axis=1, inplace=True)
        self.rejections_gt=df_summ


    @staticmethod
    def annotate_approximation(df):
        df["min_distance"]=df["distance"].min()
        df["max_distance"]=df["distance"].max()
        df["approximation_mark"]=False
        
        try:
            mark=np.where(df["distance"].values==df["min_distance"].values)[0].item()
            df["approximation_mark"].iloc[mark]=True
        except ValueError:
            pass

        df["first_frame"]=df["frame_number"].iloc[0]
        df["last_frame"]=df["frame_number"].iloc[-1]
        return df


    def infer_approximations(self, min_threshold, max_threshold):
        """        

        Arguments
            min_threshold (int): Less than this distance in mm between the body parts
            makes the flies too close to one another. Even if the distance is less than this
            but the flies are touching according to the touch column (from the touch classifier),
            the flies are too close

            max_threshold (int): Flies whose distance is less than this are in "proximity"
        
        """
        self.load_touch_database()
    
        self.touch["distance"]=(self.touch["app_dist_best"]/self.pixels_per_mm)
        df=self.touch.reset_index()

        # eps = max(0.2 * min_threshold, 0.5 * df["distance"].diff().abs().median())
        # print(f"EPS = {eps}")
        
        # disable hysteresis. correct for it offline
        eps=0

        df["approximation"]=approximation_hysteresis(
            df["distance"], df["touch"],
            min_th=min_threshold, max_th=max_threshold, eps=eps
        )
        df=annotate_bout_duration(annotate_bouts(df, "approximation"), self.framerate)
        approximations_file=f"{self.basedir}/interactions/{self.experiment}_approximations.feather"
        df.to_feather(approximations_file)
       
        df=df.groupby("bout_count").apply(self.annotate_approximation)
        self.approximations=df
        

    # def infer_approximations(self, threshold):
    #     """

    #     Arguments
    #         threshold (int): Less than this distance in mm between the centroids
    #         makes the flies too close to one another
    #     """
    #     self.load_rejections_database()
    #     approximations=self.rejections.groupby("t").first().reset_index()
    #     approximations["distance_t"]=approximations["distance"].values

    #     approximations["distance_tm1"] = np.concatenate([
    #         [np.nan],
    #         approximations["distance"].iloc[:-1].values,
    #     ])
    #     approximations["distance_tp1"] = np.concatenate([
    #         approximations["distance"].iloc[1:].values,
    #         [np.nan],
    #     ])

    #     approximations["prior_t"]=np.concatenate([
    #         [np.nan],
    #         approximations["t"].iloc[:-1].values,

    #     ])

    #     approximations["next_t"]=np.concatenate([
    #         approximations["t"].iloc[1:].values,
    #         [np.nan]
    #     ])

    #     approximations["prior_valid"]=(approximations["t"]-approximations["prior_t"])==1
    #     approximations["next_valid"]=(approximations["next_t"]-approximations["t"])==1
    #     approximations.loc[~approximations["next_valid"], "distance_tp1"]=10
    #     approximations.loc[~approximations["prior_valid"], "distance_tm1"]=10

    #     approximations["approximation"]=approximations["distance_t"]>=threshold
    #     approximations["bout"]=np.bitwise_and(~approximations["approximation"], approximations["t"].diff()>1)

    #     approximations["bout"]=approximations["bout"].cumsum()
    #     approximations=annotate_bouts(approximations, "bout")
        
    #     approximations=approximations.query("approximation==True")
    #     approximations=annotate_bouts(approximations, "bout").drop("bout", axis=1)
    #     approximations_before_contact=approximations.query("bout_out==1").query(f"distance_tp1<{threshold}")
    #     approximations_after_contact=approximations.query("bout_in==1").query(f"distance_tm1<{threshold}")
    #     approximations.loc[approximations["bout_count"].isin(approximations_before_contact["bout_count"].values), "approximation"]=False
    #     approximations.loc[approximations["bout_count"].isin(approximations_after_contact["bout_count"].values), "approximation"]=False
    #     approximations.rename({"bout_count": "bout"}, axis=1, inplace=True)
    #     stats=approximations.groupby("bout").agg({"distance": [np.min, np.argmin, len]})\
    #         .reset_index()
    #     stats.columns=["bout", "min_distance", "min_distance_i", "size"]
    #     assert (stats["size"]>stats["min_distance_i"]).all()

    #     approximations=approximations.merge(
    #         stats,
    #         on=["bout"]
    #     )
    #     self.approximations=approximations
        
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