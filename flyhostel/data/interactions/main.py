import itertools
import codetiming
import logging

import pandas as pd
import cupy as cp
import cudf


from flyhostel.data.interactions.neighbors_gpu import find_neighbors as find_neighbors_gpu
from flyhostel.data.interactions.neighbors_gpu import compute_pairwise_distances_using_bodyparts_gpu
from flyhostel.data.bodyparts import make_absolute_pose_coordinates
from flyhostel.data.pose.constants import DIST_MAX_MM, ROI_WIDTH_MM, MIN_INTERACTION_DURATION, SQUARE_WIDTH, SQUARE_HEIGHT
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


class InteractionDetector(FilesystemInterface):


    def __init__(self, *args, dist_max_mm=DIST_MAX_MM, min_interaction_duration=MIN_INTERACTION_DURATION, roi_width_mm=ROI_WIDTH_MM, **kwargs):
        
        self.framerate=None
        self.dbfile = self.load_dbfile()
        self.roi_width = load_roi_width(self.dbfile)
        self.roi_height=self.roi_width
        self.roi_width_mm=roi_width_mm
        self.px_per_mm=self.roi_width/roi_width_mm
        self.neighbors_df=None
        self.dist_max_mm=dist_max_mm
        self.min_interaction_duration=min_interaction_duration
        super(InteractionDetector, self).__init__(*args, **kwargs)


    def find_interactions(self, dt, pose, bodyparts, framerate=FRAMERATE, using_bodyparts=True):
        
        if not isinstance(dt, cudf.DataFrame) and not pd.api.types.is_categorical_dtype(dt["id"]):
            dt["id"]=pd.Categorical(dt["id"])

        if not isinstance(pose, cudf.DataFrame) and not pd.api.types.is_categorical_dtype(pose["id"]):
            pose["id"]=pd.Categorical(pose["id"])
        
        dt_gpu=cudf.DataFrame(dt[["id", "frame_number", "x", "y", "identity"]])

        if "thorax" not in bodyparts:
            # thorax is required in any case
            bodyparts=["thorax"] + bodyparts

        bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))
        pose_gpu=cudf.DataFrame(pose[["id", "frame_number"] + bodyparts_xy])
        pose_and_centroid=pose_gpu.merge(dt_gpu, on=["id", "frame_number"], how="left")
        
        logger.debug("Projecting to absolute coordinates")

        # project pose relative to the top left corner of the 100x100 square around the animal
        # to absolute coordinates, relative to the top left corner of the original raw frame
        # (which is the same for all animals)
        pose_absolute = make_absolute_pose_coordinates(
            pose_and_centroid.copy(), bodyparts,
            square_width=SQUARE_WIDTH, square_height=SQUARE_HEIGHT,
            roi_height=self.roi_height,
            roi_width=self.roi_width
        )

        # find frames where the centroid of at least two flies it at most dist_max_mm mm from each other
        neighbors=self.find_neighbors(pose_absolute[["id", "frame_number", "centroid_x", "centroid_y"]], dist_max_mm=self.dist_max_mm)

        if using_bodyparts:

            # for those frames, go through each pair of 'neighbors' and compute the distance between the two closest bodyparts
            neighbors=self.compute_pairwise_distances_using_bodyparts(neighbors.copy(), pose_absolute, bodyparts, bodyparts_xy)

            neighbors["distance_bodypart_mm"]=neighbors["distance_bodypart"]/self.px_per_mm
            neighbors=neighbors.loc[neighbors["distance_bodypart_mm"] < self.dist_max_mm]
            neighbors=neighbors.sort_values(["id", "nn", "frame_number"])

            neighbors["scene_start"]=[True] + (cp.diff(neighbors["frame_number"])!=1).tolist()
            neighbors["interaction"]=neighbors["scene_start"].cumsum()

            neighbors=neighbors.merge(
                neighbors.groupby("interaction").size().reset_index().rename({0: "frames"}, axis=1),
                on=["interaction"],
                how="left"
            ).sort_values(["frame_number", "id"])
            neighbors["duration"]=neighbors["frames"]/framerate
            interactions=neighbors.loc[(neighbors["duration"]>=self.min_interaction_duration)]

            return interactions
        else:
            return neighbors
        

    def compute_pairwise_distances_using_bodyparts(self, *args, **kwargs):
        return compute_pairwise_distances_using_bodyparts_gpu(*args, **kwargs)


    def find_neighbors(self, dt, dist_max_mm=None):

        if dist_max_mm is None:
            dist_max_mm=self.dist_max_mm


        if self.neighbors_df is None or self.dist_max_mm!=dist_max_mm:
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