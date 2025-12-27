import logging
import glob

from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from movement.io import load_poses
from movement.filtering import filter_by_confidence, interpolate_over_time
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.utils import (
    get_framerate,
    get_basedir,
    get_square_width,
    get_chunksize,
    get_number_of_animals,
)
from flyhostel.data.pose.constants import legs as all_legs
from flyhostel.data.interactions.utils import read_label_file_rejections
from flyhostel.data.interactions.constants import DATA_DIR
from flyhostel.data.pose.ethogram.utils import annotate_bouts, annotate_bout_duration
from .utils import (
    add_centroid_offset_single,
    stack_individuals,
    get_frames,
)
from .constants import INTERPOLATE, FILTER_BY_CONFIDENCE

legs=[leg for leg in all_legs if "J" not in leg]


logger=logging.getLogger(__name__)

def load_experiment_features(experiment):

    number_of_animals=get_number_of_animals(experiment)
    animals=[f"{experiment}__{str(i+1).zfill(2)}" for i in range(number_of_animals)]


    datasets=[]
    framerate_index=[]

    for animal in tqdm(animals):
        experiment, identity = animal.split("__")
        identity=int(identity)


        basedir=get_basedir(experiment)
        framerate=get_framerate(experiment)
        square_width=get_square_width(experiment)
        chunksize=get_chunksize(experiment)
        framerate_index.append(
            (experiment, basedir, framerate, chunksize, square_width)
        )

        loader=FlyHostelLoader(experiment=experiment, identity=identity)
        loader.load_centroid_data(cache="/flyhostel_data/cache")

        pose_file=loader.get_pose_file_h5py(pose_name="raw")
        frames=get_frames(pose_file, chunksize)

        
        ds=load_poses.from_file(pose_file, source_software="SLEAP", fps=framerate)
        ds=ds.sel(keypoints=["head", "thorax", "abdomen", "proboscis"] + legs)
        
        # only keep pose frames where the centroid frames are available
        # used to remove padded pose data at the beginning of the first chunk
        pose_frames=pd.DataFrame({"frame_number": frames})
        pose_frames=pose_frames.merge(loader.dt[["frame_number", "id"]], how="left")
        ds=ds.isel(time=np.bitwise_not(pose_frames["id"].isna()))

        # project to top left corner of original frame
        cx = loader.dt["center_x"].to_numpy() - square_width // 2
        cy = loader.dt["center_y"].to_numpy() - square_width // 2


        # add good time measurement
        frames = loader.dt["frame_number"].to_numpy()
        times  = loader.dt["t"].to_numpy()            # seconds (float), or whatever unit you store

        assert len(frames) == ds.sizes["time"] == len(times), "Mismatch between dt and ds!"

        ds = ds.assign_coords(
            frame_number=("time", frames),            # extra coord you can keep
            time=("time", times)                      # replace time coordinate values
        )
        # If you want time strictly increasing (and dt might not be sorted):
        ds = ds.sortby("time")


        ds = add_centroid_offset_single(ds, cx, cy)  # now positions are absolute

        if FILTER_BY_CONFIDENCE:
            ds.update({"position": filter_by_confidence(ds.position, ds.confidence, print_report=True)})
        if INTERPOLATE:
            ds.update({"position": interpolate_over_time(ds.position, print_report=True)})

        datasets.append(ds)

    framerate_index=pd.DataFrame.from_records(framerate_index, columns=[
        "experiment", "basedir", "framerate", "chunksize", "square_width",
    ])
    # Give your flies readable IDs
    ds = stack_individuals(*datasets, ind_names=animals, join="outer")

    return ds


def load_labels(dataset):

    df=[]
    for experiment in dataset:
        number_of_animals=get_number_of_animals(experiment)
        framerate=get_framerate(experiment)
        chunksize=get_chunksize(experiment)
        animals=[f"{experiment}__{str(i+1).zfill(2)}" for i in range(number_of_animals)]

        for animal in tqdm(animals):
            identity=animal.split("__")[1]

            label_files=glob.glob(f"{DATA_DIR}/{experiment}_*{identity}/*csv")
            df_human=[]
            for path in label_files:
                labels=read_label_file_rejections(path, chunksize=chunksize)
                df_human.append(labels)
            df_human=pd.concat(df_human, axis=0).reset_index(drop=True)
            df_human["touch"]=np.bitwise_or(df_human["touch_focal"], df_human["touch_side"])==1
            df_human["framerate"]=framerate

            df_human=df_human[["id", "experiment", "data_entry", "touch", "chunk", "frame_idx", "frame_number", "framerate"]]\
                .groupby("id").apply(lambda dff: annotate_bout_duration(annotate_bouts(dff, "touch"), fps=dff["framerate"].iloc[0]))
            df.append(df_human)

    df=pd.concat(df, axis=0).reset_index(drop=True)
    return df


def load_features(dataset):
    features={}
    for experiment in dataset:
        features[experiment]=load_experiment_features(experiment)
    return features


def load_sleep_data(loader, bout_annotation=None):
    fps=1
    loader.load_sleep_data(bin_size=None)
    assert "windowed_var" in loader.sleep.columns
    sleep=loader.sleep.rename({
        "windowed_var": "inactive",
    }, axis=1)
    sleep=sleep[["id", "animal", "t", "frame_number", "inactive", "asleep"]]
    assert sleep.shape[1]==6
    if bout_annotation is not None:
        sleep=annotate_bout_duration(annotate_bouts(sleep, bout_annotation), fps)

    return sleep

def load_sleep_data_all(loaders, number_of_animals, bout_annotation=None):

    sleep_df=[]
    for identity in range(1, number_of_animals+1):
        loader=[loader for loader in loaders if loader.identity == identity][0]
        sleep=load_sleep_data(loader, bout_annotation=bout_annotation)
        sleep_df.append(sleep)

    sleep_df=pd.concat(sleep_df, axis=0).reset_index(drop=True)
    sleep_df["frame_number"]=np.array(sleep_df["frame_number"].values, np.int64)

    return sleep_df