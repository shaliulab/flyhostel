import os
import codetiming
import logging
import itertools
import argparse
import pickle

import pandas as pd

from flyhostel.data.interactions.load_data import load_animal_dataset
from flyhostel.data.movies.pose import draw_pose_on_axis
from flyhostel.data.interactions.centroid_detector import centroid_interaction_detector
from flyhostel.data.interactions.sleap_ import generate_sleap_files
raise NotImplementedError()
DATA_PATH=None
SLEAP_DATA=os.environ["SLEAP_DATA"]
SLEAP_PROJECT_PATH=os.environ["SLEAP_PROJECT_PATH"]

half_window_in_frames=75

logger = logging.getLogger(__name__)

def get_parser():

    ap = argparse.ArgumentParser()
    command_group = ap.add_mutually_exclusive_group()

    command_group.add_argument("--animals", nargs="+", type=str, help="Example = FlyHostel4_6X_2023-06-27_14-00-00__03")
    command_group.add_argument("--experiment", type=str, help="basedir in file format, example FlyHostel4_6X_2023-06-27_14-00-00")

    ap.add_argument("--distance-threshold", type=float, required=True, help="max distance between two flies during an interaction, in cm")
    ap.add_argument("--save-img", action="store_true", required=False, default=False)
    ap.add_argument("--output-folder", type=str, required=False, default="./output")
    return ap


def compute_interactions(animal0, animal1, interaction_detector_FUN, with_data=False, **kwargs):
    

    dt0, dt_index = load_animal_dataset(animal0)
    dt1, _ = load_animal_dataset(animal1)

    interactions = interaction_detector_FUN(
        animal0, animal1,
        dt0, dt1, dt_index,
        **kwargs
    )

    interactions["animal0"]=animal0
    interactions["animal1"]=animal1

    if with_data:
        return interactions, [dt0, dt1]
    else:
        return interactions


def analyze_pairwise_interactions(animal0, animal1, min_length=0, max_length=20, output_folder="./interactions", save_img=False, **kwargs):

    logger.info("Detecting interaction events")
    
    with codetiming.Timer(text="Done in {:.4f} seconds", logger=logger):
        interactions, dts = compute_interactions(
            animal0, animal1,
            interaction_detector_FUN=centroid_interaction_detector,
            with_data=True,
            **kwargs
        )
    interactions = interactions.loc[(interactions["length"] > min_length) & (interactions["length"] <= max_length)]
    interactions.to_csv(f"{output_folder}/interactions.csv")

    if save_img:

        with open(f"{DATA_PATH}/{animal0.split('__')[0]}.pkl", "rb") as handle:
            params=pickle.load(handle)
        params["animals"] = [animal0, animal1]
        
        for interaction_id in range(interactions.shape[0]):
            fig = render_interaction(dts=dts, frame_number=interactions.iloc[interaction_id]["frame_number"], params=params)
            fig.savefig(f"{output_folder}/interaction_{str(interaction_id).zfill(4)}.png", bbox_inches='tight', pad_inches=0)
            fig.clear()

    return interactions
    


def render_interaction(dts, frame_number, params):

    pose_data = []
    for dt_i in dts:
    
        pose_data.append((
            dt_i.loc[
                (dt_i.index >= frame_number-half_window_in_frames) & (dt_i.index < frame_number+half_window_in_frames)
            ]
        ))
    
    
    fns={"raw":pose_data[0].index[0]}
    h5inds=[0, 1]

    return draw_pose_on_axis(pose_data, fns, h5inds, params)


def main():

    ap = get_parser()
    args = ap.parse_args()


    output_folder=args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    if args.experiment:
        number_of_animals=int(args.experiment.split("_")[1].replace("X", ""))
        animals = [f"{args.experiment}__{str(i+1).zfill(2)}" for i in range(number_of_animals)]

    else:
        animals=args.animals

    animal_pairs = itertools.combinations(animals, 2)

    interactions=[]
    for animal0, animal1 in animal_pairs:
        subfolder=f"{output_folder}/{animal0}_{animal1}"
        os.makedirs(subfolder, exist_ok=True)

        pair_interactions = analyze_pairwise_interactions(
            animal0, animal1, max_length=20,
            output_folder=subfolder,
            save_img=args.save_img, distance_threshold=args.distance_threshold
        )

        logger.info("Generating sleap file %s_%s", animal0, animal1)
        generate_sleap_files(pair_interactions, root=SLEAP_PROJECT_PATH)

        interactions.append(
            pair_interactions
        )

    interactions=pd.concat(interactions)
    interactions.to_csv(f"{output_folder}/interactions.csv")



if __name__ == "__main__":
    main()