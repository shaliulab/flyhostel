import itertools
import codetiming
import logging
import pandas as pd
import numpy as np
import cupy as cp
import cudf


from flyhostel.data.interactions.neighbors_gpu import find_neighbors as find_neighbors_gpu
from flyhostel.data.interactions.neighbors_gpu import compute_pairwise_distances_using_bodyparts_gpu
from flyhostel.data.bodyparts import make_absolute_coordinates
from flyhostel.data.pose.constants import DIST_MAX_MM, ROI_WIDTH_MM, MIN_INTERACTION_DURATION, SQUARE_WIDTH, SQUARE_HEIGHT, MIN_TIME_BETWEEN_INTERACTIONS
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.utils import load_roi_width
from flyhostel.utils.filesystem import FilesystemInterface

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    useGPU=True
except:
    cp=None
    useGPU=False
    logger.debug("Cannot load cupy")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

class InteractionDetector(FilesystemInterface):


    def __init__(self, *args, dist_max_mm=DIST_MAX_MM, min_interaction_duration=MIN_INTERACTION_DURATION, min_time_between_interactions=MIN_TIME_BETWEEN_INTERACTIONS, roi_width_mm=ROI_WIDTH_MM, **kwargs):
        
        self.framerate=None
        self.dbfile = self.load_dbfile()
        self.roi_width = load_roi_width(self.dbfile)
        self.roi_height=self.roi_width
        self.roi_width_mm=roi_width_mm
        self.px_per_mm=self.roi_width/roi_width_mm
        self.neighbors_df=None
        self.dist_max_mm=dist_max_mm
        self.min_time_between_interactions=min_time_between_interactions
        self.min_interaction_duration=min_interaction_duration

        super(InteractionDetector, self).__init__(*args, **kwargs)


    def find_interactions(self, dt, pose, bodyparts, framerate=FRAMERATE, using_bodyparts=True, useGPU=True):
        if useGPU:
            xf=cudf
            nx=cp
        else:
            xf=pd
            nx=np
        
        if not isinstance(dt, cudf.DataFrame) and not pd.api.types.is_categorical_dtype(dt["id"]):
            dt["id"]=pd.Categorical(dt["id"])

        if not isinstance(pose, cudf.DataFrame) and not pd.api.types.is_categorical_dtype(pose["id"]):
            pose["id"]=pd.Categorical(pose["id"])

        if "thorax" not in bodyparts:
            # thorax is required in any case
            bodyparts=["thorax"] + bodyparts

        bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))
        

        if isinstance(dt, cudf.DataFrame):
            dt_cpu=dt[["id", "frame_number", "x", "y", "identity"]].to_pandas()
            dt_gpu=dt
        else:
            dt_cpu=dt
            dt_gpu=None
        
        if isinstance(dt, cudf.DataFrame):
            pose_cpu=pose[["id", "frame_number"] + bodyparts_xy].to_pandas()
            pose_gpu=pose
        else:
            pose_cpu=pose
            pose_gpu=None
        
        del dt_gpu
        del dt
        del pose_gpu
        del pose
        pose_and_centroid=pose_cpu.merge(dt_cpu, on=["id", "frame_number"], how="left")
        
        logger.debug("Projecting to absolute coordinates")

        # project pose relative to the top left corner of the 100x100 square around the animal
        # to absolute coordinates, relative to the top left corner of the original raw frame
        # (which is the same for all animals)
        all_absolute = make_absolute_coordinates(
            pose_and_centroid, bodyparts,
            square_width=SQUARE_WIDTH,
            square_height=SQUARE_HEIGHT,
            roi_height=self.roi_height,
            roi_width=self.roi_width
        )
        
        dt_absolute=all_absolute[["id", "frame_number", "centroid_x", "centroid_y"]]
        pose_absolute=all_absolute[["id", "frame_number"] + bodyparts_xy]
        del all_absolute
        
        # only upload to GPU now (and if xf is cudf)
        # this is to avoid having copies of the same data in the GPU,
        # which has limited memory
        dt_absolute=xf.DataFrame(dt_absolute)
        pose_absolute=xf.DataFrame(pose_absolute)
    
        # find frames where the centroid of at least two flies it at most dist_max_mm mm from each other
        neighbors=self.find_neighbors(dt_absolute, dist_max_mm=self.dist_max_mm)

        if using_bodyparts:

            # assert ((neighbors["frame_number"].iloc[0]-pose_absolute["frame_number"].values)==0).sum()==1

            # for those frames, go through each pair of 'neighbors' and compute the distance between the two closest bodyparts
            neighbors=self.compute_pairwise_distances_using_bodyparts(
                cudf.DataFrame(neighbors), cudf.DataFrame(pose_absolute),
                bodyparts, bodyparts_xy, useGPU=True
            )
            if neighbors is None:
                return None

            neighbors["distance_bodypart_mm"]=neighbors["distance_bodypart"]/self.px_per_mm
            neighbors=neighbors.loc[neighbors["distance_bodypart_mm"] < self.dist_max_mm]
            if neighbors.shape[0]==0:
                return neighbors, pose_absolute
            
            neighbors=neighbors.sort_values(["id", "nn", "frame_number"])

            # annotate interaction start and end
            min_steps_between=int(self.min_time_between_interactions*FRAMERATE)
            neighbors=annotate_interaction_location_all(neighbors, min_steps_between, nx=nx, xf=xf)

            # time each interaction by counting in how many frames is it detected
            neighbors=neighbors.merge(
                neighbors.groupby(["id", "nn", "interaction"]).size().reset_index(name="frames"),
                on=["id", "nn", "interaction"],
                how="left"
            ).sort_values(["id", "nn", "frame_number"])
            neighbors["duration"]=neighbors["frames"]/framerate
            interactions=neighbors.loc[(neighbors["duration"]>=self.min_interaction_duration)]


            return interactions, pose_absolute
        else:
            return neighbors, pose_absolute


    def compute_pairwise_distances_using_bodyparts(self, *args, **kwargs):
        return compute_pairwise_distances_using_bodyparts_gpu(*args, **kwargs)


    def find_neighbors(self, dt, dist_max_mm=None):

        if dist_max_mm is None:
            dist_max_mm=self.dist_max_mm


        with codetiming.Timer():
            logger.debug("Computing distance between animals")
            # even though the pose is not needed to find the nns
            # dt_with_pose is the most raw dataset that contains the centroid_x and centroid_y
            # of the agents in absolute coordinates
            dist_max_px=dist_max_mm*self.px_per_mm
            logger.info("Neighbors = < %s mm (%s pixels)", dist_max_mm, dist_max_px)

            neighbors = find_neighbors_gpu(dt, dist_max_px)
            neighbors=neighbors.loc[neighbors["frame_number"].drop_duplicates().index]
            neighbors["distance_mm"] = neighbors["distance"] / self.px_per_mm
            
            neighbors_df=neighbors[
                ["id", "nn", "distance_mm", "distance", "frame_number"]
            ]
            self.neighbors_df=neighbors_df.sort_values(["frame_number", "id"])

        self.dist_max_mm=dist_max_mm
        return self.neighbors_df
    

    @staticmethod
    def interactions_by_closest_point(interactions):
        return interactions.groupby(["id", "nn", "interaction"]).apply(lambda df: df.iloc[cp.argmin(df["distance_bodypart"].values)]).to_pandas()


    @staticmethod
    def interactions_by_first_point(interactions):
        return interactions.groupby(["id", "nn", "interaction"]).first().reset_index().to_pandas()
    

    @staticmethod
    def flatten_interactions(interactions):
        interactions["_idx"]=list(range(interactions.shape[0]))
        interactions_idx= pd.concat([
            interactions[["id", "_idx"]],
            interactions[["nn", "_idx"]].rename({"nn": "id"}, axis=1)
        ], axis=0).reset_index(drop=True)
        
        return interactions_idx.merge(interactions.drop(["id", "nn"], axis=1), on="_idx", how="left")

        
def annotate_interaction_location(neighbors, min_steps_between, nx, xf):

    neighbors["scene_start"]=nx.concatenate([
        nx.array([True]),
        nx.diff(neighbors["frame_number"])>=min_steps_between
    ])
    neighbors["interaction"]=neighbors["scene_start"].cumsum()
    return neighbors


def annotate_interaction_location_all(neighbors, min_steps_between, nx, xf):
    
    try:
        neighbors_cpu=neighbors.to_pandas()
    except:
        neighbors_cpu=neighbors
        
    neighbors_cpu=neighbors_cpu.groupby(["id", "nn"]).apply(lambda df: annotate_interaction_location(df, min_steps_between, np, pd))
    return xf.DataFrame(neighbors_cpu)