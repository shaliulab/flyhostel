import logging
import os.path
import numpy as np
import pandas as pd
from flyhostel.utils import (
    get_chunksize,
    annotate_local_identity,
    build_interaction_video_key
)

logger=logging.getLogger(__name__)

CONTACT_THRESHOLD=4 # mm
PROXIMITY_THRESHOLD=5 # mm
DURATION_THRESHOLD=.5 # seconds
MIN_TIME_BETWEEN_INTERACTIONS=1 # seconds

def partition_interactions(
        neighbors,
        contact_threshold=CONTACT_THRESHOLD,
        duration_threshold=DURATION_THRESHOLD,
        min_time_between_interactions=MIN_TIME_BETWEEN_INTERACTIONS,
    ):
    """
    
    Arguments:
        neighbors (pd.DataFrame): Dataset of fly pairs (id, nn)
            and the distance between them (distance_mm)
            with time annotation (t and frame_number)
            which are in proximity to each other (but maybe not in contact)
        framerate (float): number of data points per second in the original data (number of different frame_number values per second)
        contact_threshold (float): Min distance between a pair of flies for them to be in contact (< proximity) in mm
            Pairs of neighbors which achieve in some pairwise proximity event a distance between them under this value are said to be interacting
        duration_threshold (float): Min time the interaction needs to last, in seconds.
            NOTE: not all the time during the interaction
                the flies need to be in contact distance. But the time between contacts
                DOES need to be less than min_time_between_interactions 
        min_time_between_interactions (float): minimum time between two interactions for them to be considered separately and not combined

    Return:
        interaction_index (pd.DataFrame): contains columns id, nn, interaction, t1, t2, duration, size, first_frame, last_frame_number, within_limits
    
    """
    
    df=[]
    groupby=["id", "nn", "interaction"]
    neighbors.sort_values(["frame_number", "id", "nn"], inplace=True)
    for pair, dff in neighbors.groupby(groupby[:2]):
        diff_t=np.diff(dff["t"])
        dff["interaction"]=[0] + (np.cumsum(diff_t>min_time_between_interactions)).tolist()
        df.append(dff)
    
    if df:
        interactions=pd.concat(df, axis=0).reset_index(drop=True)
    else:
        raise ValueError("No interactions detected")
    
    del df
    
    interaction_stats=interactions.groupby(groupby).agg({
        "distance_mm": np.min,
        "t": [np.min, np.max, len],
        "frame_number": [np.min, np.max]
    }).reset_index()
    
    interaction_stats.columns=groupby + [
        "distance_mm_min",  # np.min above
        "t1", "t2", "size",  # np.min, np.max and len above
        "first_frame", "last_frame_number" # np.min and np.max above
    ]
    interaction_stats["duration"]=interaction_stats["t2"]-interaction_stats["t1"]
    interaction_stats["within_limits"]=interaction_stats["size"]/\
        (1+interaction_stats["last_frame_number"]-interaction_stats["first_frame"])
    
    interaction_index=interactions.merge(interaction_stats, on=groupby, how="left")
    
    interaction_index["signif"] = np.where(
        (interaction_index["duration"] > duration_threshold) &
        (interaction_index["distance_mm_min"] < contact_threshold),
        True,
        False
    )

    return interaction_index


class InteractionsLoader:
    """
    A class to load the result of the first step in the interactions pipeline
    """
    
    basedir=None
    experiment=None
    identity=None
    chunksize=None
    ids=[]
    dt=None

    def __init__(self, *args, **kwargs):
        self.interaction=None
        self.all_interactions=None
        super(InteractionsLoader, self).__init__(*args, **kwargs)

    def get_interactions_data_dir(self):
        if self.framerate == 150:
            return "/flyhostel_data/fiftyone/FlyBehaviors/DEG-REJECTIONS/rejections_deepethogram/DATA_150fps"
        else:
            return "/flyhostel_data/fiftyone/FlyBehaviors/DEG-REJECTIONS/rejections_deepethogram/DATA"
    
    def load_interaction_database(self):
        
        """

        Populates interaction_database with a DataFrame with columns
        ___
        """

        feather_file=os.path.join(self.basedir, "interactions", self.experiment + "_database.feather")
        interaction_database=pd.read_feather(feather_file)
        interaction_database=interaction_database.loc[interaction_database["id"]==self.ids[0]]
        interaction_database["identity"]=interaction_database["id"].str.slice(start=-2).astype(int)
        interaction_database["identity_partner"]=interaction_database["nn"].str.slice(start=-2).astype(int)
        interaction_database["chunk"]=interaction_database["frame_number"]//self.chunksize
        interaction_database=annotate_local_identity(interaction_database, self.experiment)
        interaction_database["key"]=[build_interaction_video_key(self.experiment, row) for _, row in interaction_database.iterrows()]

        self.interaction_database=interaction_database       


    def load_interaction_data(
            self, experiment=None, identity=None,
            min_time=None, max_time=None,
            proximity_threshold=PROXIMITY_THRESHOLD,
            identities=None,
            **kwargs
        ):

        """
        Arguments:
            proximity_threshold (float): Min distance between a pair of flies for them to be in proximity
            framerate (float): Number of datapoints per second in the read data
                This may or may not be equal to the original framerate of the recording,
                which is the number of different values of frame_number per second
            identities (list): If not None, list of integer identities which represent
                the identity of the interaction partner flies
            **kwargs Arguments to partition_interactions
        
        Populates self.interaction with a data frame with columns:

        * id
        * frame_number
        * t
        * nn: id of interaction partner
        * distance_mm: Distance between partner centroids
        * distance_mm_min: Minimum distance_mm reached during the interaction
        * interaction: Unique interaction identifier within each pair of partners
        * duration: Duration of the interaction in seconds
        * within_limits: Fraction of frames during the interaction where the animals pass the proximity threshold
            Not necessarily 100 % (=1) because 2 interactions where within_limits=1 that are closer than 
            min_time_between_interactions seconds in time become a single interaction,
            and therefore the time in between becomes part of a new super interaction



        Returns
            None
        """

        if experiment is None:
            experiment=self.experiment
        if identity is None:
            identity=self.identity

        neighbors_features=["id", "nn", "distance_mm", "frame_number", "t"]
        csv_file=os.path.join(self.basedir, "interactions", experiment + "_neighbors.csv")
        if not os.path.exists(csv_file):
            logger.warning("%s not found", csv_file)
            return None

        neighbors=pd.read_csv(csv_file)[neighbors_features]

        if min_time is not None:
            neighbors=neighbors.loc[neighbors["t"]>=min_time]
        
        if max_time is not None:
            neighbors=neighbors.loc[neighbors["t"]<max_time]
        
        
        if identities is not None:
            neighbors["identity"]=neighbors["id"].str.slice(start=-2).astype(int)
            neighbors=neighbors.loc[neighbors["identity"].isin(identities)]
            del neighbors["identity"]

        neighbors=neighbors.loc[neighbors["id"]==self.ids[0]]
        assert neighbors.shape[0]>0, f"No flies neighboring {self.ids[0]}"

        if proximity_threshold is not None:
            neighbors=neighbors.loc[neighbors["distance_mm"]<proximity_threshold]

        assert neighbors.shape[0]>0, f"No data left for {self.ids[0]}"
        interactions=partition_interactions(
            neighbors,
            **kwargs
        )
        self.all_interactions=interactions
        self.interaction=interactions.loc[interactions["signif"]==True]
        self.interaction.drop(["signif", "size", "t1", "t2"], axis=1, inplace=True)
        return None

    def index_time(self):
        # not needed anymore
        time_index=self.dt[["frame_number", "t"]]
        return time_index