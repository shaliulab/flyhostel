import argparse
import logging
import os.path

import pandas as pd

from flyhostel.data.groups.group import FlyHostelGroup
from flyhostel.data.pose.constants import legs, bodyparts

logger=logging.getLogger(__name__)

def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    return ap

def main():
    ap=get_parser()
    args=ap.parse_args()
    
    metadata=pd.read_csv(args.metadata)
    for (fhn, fhd, number_of_animals), meta in metadata.groupby(["flyhostel_number", "flyhostel_date", "number_of_animals"]):
        
        if number_of_animals == 1:
            continue

        if number_of_animals != meta.shape[0]:
            logger.warning("Reported number of animals does not match number of animals in group")
        compute_interactions(meta)

def compute_interactions(metadata):
       
    group=FlyHostelGroup.from_metadata(metadata)
    dt=group.load_centroid_data(framerate=30)
    pose=group.load_pose_data("pose_boxcar", framerate=30)
    
    group.dist_max_mm=2
    group.min_interaction_duration=0
    interactions_df=group.find_interactions(dt, pose, bodyparts, using_bodyparts=False)
    del dt
    del pose

    output_folder=os.path.join(group.basedir, "flyhostel", "group")
    os.makedirs(output_folder, exist_ok=True)
    output=os.path.join(output_folder, f"{group.experiment}_interactions.feather")

    del group

    interactions_df.reset_index().to_pandas().to_feather(output)
    return interactions_df
