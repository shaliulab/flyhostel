from abc import ABC
import itertools
import codetiming
import logging

import cupy as cp
import cudf

from flyhostel.data.interactions.neighbors_gpu import find_neighbors as find_neighbors_gpu
from flyhostel.data.interactions.neighbors_gpu import compute_pairwise_distances_using_bodyparts_gpu
from flyhostel.data.bodyparts import make_absolute_pose_coordinates, legs

logger = logging.getLogger(__name__)
dist_max_mm=4

class InteractionDetector(ABC):

    framerate=None
    roi_height=None
    roi_width=None
    dist_max_mm=None
    px_per_mm=None
    neighbors_df=None

    def __init__(self, *args, dist_max_mm=dist_max_mm, **kwargs):
        self.dist_max_mm=dist_max_mm


    def find_interactions(self, dt, pose, bodyparts, min_interaction_duration=1):
        dt_gpu=cudf.DataFrame(dt[["id", "frame_number", "x", "y", "identity"]])

        bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))
        pose_gpu=cudf.DataFrame(pose[["id", "frame_number"] + bodyparts_xy])
        pose_and_centroid=pose_gpu.merge(dt_gpu, on=["id", "frame_number"], how="left")
        
        logger.debug("Projecting to absolute coordinates")

        # project pose relative to the top left corner of the 100x100 square around the animal
        # to absolute coordinates, relative to the top left corner of the original raw frame
        # (which is the same for all animals)
        pose_absolute = make_absolute_pose_coordinates(
            pose_and_centroid.copy(), bodyparts,
            square_width=100, square_height=100,
            roi_height=self.roi_height,
            roi_width=self.roi_width
        )

        # find frames where the centroid of at least two flies it at most dist_max_mm mm from each other
        neighbors=self.find_neighbors(pose_absolute[["id", "frame_number", "centroid_x", "centroid_y"]], dist_max_mm=4)

        # for those frames, go through each pair of 'neighbors' and compute the distance between the two closest bodyparts
        neighbors=compute_pairwise_distances_using_bodyparts_gpu(neighbors.copy(), pose_absolute, bodyparts, bodyparts_xy)

        neighbors["distance_bodypart_mm"]=neighbors["distance_bodypart"]/self.px_per_mm
        neighbors["scene_start"]=[True] + (cp.diff(neighbors["frame_number"])!=1).tolist()
        neighbors["interaction"]=neighbors["scene_start"].cumsum()
        neighbors=neighbors.merge(
            neighbors.groupby("interaction").size().reset_index().rename({0: "frames"}, axis=1),
            on=["interaction"],
            how="left"
        )
        neighbors["duration"]=neighbors["frames"]/self.framerate
        interactions=neighbors.loc[(neighbors["duration"]>=min_interaction_duration)]

        return interactions


    def find_neighbors(self, dt, dist_max_mm=None, useGPU=True):

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

                if useGPU:
                    neighbors = find_neighbors_gpu(dt, dist_max_px)

                else:
                     raise NotImplementedError()

                neighbors=neighbors.loc[neighbors["frame_number"].drop_duplicates().index]
                neighbors["distance_mm"] = neighbors["distance"] / self.px_per_mm

                self.neighbors_df=neighbors[
                    ["id", "nn", "distance_mm", "distance", "frame_number"]
                ]
            
        self.dist_max_mm=dist_max_mm
        return self.neighbors_df