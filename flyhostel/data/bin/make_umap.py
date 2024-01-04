import itertools
import logging
import os.path
import pickle
import time

import hdf5storage
import pandas as pd
import numpy as np
import plotly.express as px

from umap import UMAP

from flyhostel.data.pose.movie import connect_bps
from flyhostel.data.bodyparts import bodyparts as BODYPARTS
from flyhostel.data.pose.sleap import draw_video_row
from flyhostel.data.pose.filters import filter_pose, arr2df
from flyhostel.data.deg import read_label_file
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.pose.fh_umap import add_n_steps_in_the_past
lts_dir="/home/vibflysleep/opt/long_timescale_analysis/slurm/FlyHostel_long_timescale_analysis"
wavelet_downsample=5

POSE_DATA=os.environ["POSE_DATA"]
MOTIONMAPPER_DATA=os.environ["MOTIONMAPPER_DATA"]
OUTPUT_FOLDER=os.path.join(MOTIONMAPPER_DATA, "output")
UMAP_PATH=os.path.join(OUTPUT_FOLDER, "umap.pkl")

logger=logging.getLogger(__name__)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

experiments=[
    # "FlyHostel2_1X_2023-11-24_11-00-00",
    # "FlyHostel1_1X_2023-11-24_11-00-00",
    "FlyHostel1_1X_2023-09-28_16-00-00",
    # "FlyHostel2_1X_2023-09-28_16-00-00",
    # "FlyHostel1_1X_2023-11-08_17-00-00",
    "FlyHostel1_1X_2023-11-03_13-00-00",
    # "FlyHostel2_1X_2023-11-03_13-00-00",
    # "FlyHostel1_1X_2023-11-13_11-00-00",
    # "FlyHostel2_1X_2023-11-13_11-00-00",    
]

stride=75
chunksize=45000
framerate=10
filters={"nanmedian": {"window_size": 0.2, "min_window_size": 40, "order": 0}, "nanmean": {"window_size": 0.2, "min_window_size":10, "order": 1}}
interpolate_seconds={bp: 30 for bp in BODYPARTS}
interpolate_seconds["proboscis"]=0.5
ZT_START=14
ZT_END=15
min_score={bp: 0.5 for bp in BODYPARTS}
min_score["proboscis"]=0.8
bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in BODYPARTS]))
bodyparts_speed=list(itertools.chain(*[[bp + "_speed"] for bp in BODYPARTS]))
bodyparts_speed
min_time=ZT_START*3600
max_time=ZT_END*3600


dt0_features=["head_proboscis_distance"]
feature_cols=bodyparts_speed + dt0_features
feature_cols=dt0_features+bodyparts_xy
NUMBER_OF_SAMPLES={"walk": 10_000, "inactive": 10_000, "groom": 10_000}

def load_dataset(loader):

    # load pose data and annotate it using ground truth
    #####################################################
    pose=loader.pose_boxcar.copy()
    pose_annotated=loader.annotate_pose(pose, loader.deg)
    index=pose_annotated.index[::wavelet_downsample]

    # load wavelet transform of the data
    #####################################################
    matfile=os.path.join(lts_dir, "Wavelets", loader.datasetnames[0] + "-pcaModes-wavelets.mat")

    if not os.path.exists(matfile):
        print(f"{matfile} not found")
        return None
    
    data=hdf5storage.loadmat(matfile)
    wavelets=data["wavelets"]
    freq_names=[f.decode() for f in data["freq_names"]]
    frequencies=data["f"]

    # NOTE
    # For some reason, the wavelet transform may come with a number of rows
    # slightly different from the expected (same as in the pose input)
    # this difference can be up to 2 frames
    MAX_MISSING_WAVELET_ROWS=2
    # that means this assert is not going to be true
    # figure out why this happens, in the meantime, accept a potential misalignment of ground truth label+pose vs wavelets
    # assert wavelets.shape[0] == pose_annotated.shape[0]
    steps_off=(index.shape[0] - wavelets.shape[0])
    if 0 < steps_off < MAX_MISSING_WAVELET_ROWS:
        print(f"Misalignment of wavelets and pose of {steps_off} steps")
        index = index[:wavelets.shape[0]]

    # merge pose and wavelet information
    pose_annotated_with_wavelets=pd.merge(pose_annotated, pd.DataFrame(wavelets, index=index, columns=freq_names), left_index=True, right_index=True)
    del wavelets

    # generate a dataset of wavelets and the ground truth for all behaviors
    ##########################################################################
    pe_inactive=pose_annotated_with_wavelets.loc[pose_annotated_with_wavelets["behavior"]=="pe_inactive"]
    n=pe_inactive.shape[0]
    behaviors=np.unique(pose_annotated_with_wavelets["behavior"]).tolist()
    for behav in ["unknown", "pe_inactive"]:
        if behav in behaviors:
            behaviors.pop(behaviors.index(behav))


    dfs=[pe_inactive]
    for behav in behaviors:
        d=pose_annotated_with_wavelets.loc[pose_annotated_with_wavelets["behavior"]==behav].sample(frac=1).reset_index(drop=True)
        samples_available=d.shape[0]
        if behav=="pe_inactive":
            n_max=samples_available
        else:
            max_seconds=60
            n_max=6*max_seconds
        number_of_samples=NUMBER_OF_SAMPLES.get(behav, n_max)
        dfs.append(
            d.iloc[:number_of_samples]
        )

    labeled_dataset = pd.concat(dfs, axis=0)
    unknown_dataset = pose_annotated_with_wavelets.loc[pose_annotated_with_wavelets["behavior"]=="unknown"]

    return labeled_dataset, unknown_dataset, freq_names


def main():
    loaders={}

    for experiment in experiments:
        print(experiment)
        loader = FlyHostelLoader(experiment, chunks=range(0, 400))
        loader.load_and_process_data(
            min_time=min_time, max_time=max_time,
            stride=stride, bodyparts=BODYPARTS,
            cache="/flyhostel_data/cache",
            filters=filters, min_score=min_score,
            window_size_seconds=0.5, max_jump_mm=1,
            interpolate_seconds=interpolate_seconds,
            useGPU=0
        )
        loaders[experiment]=loader
    print("Done")

    datasets=[]
    unknown_datasets=[]
    freq_names=None

    for experiment in experiments:
        print(experiment)
        out=load_dataset(loaders[experiment])
        if out is None:
            continue
        labeled_dataset, unknown_dataset, freq_names=out
        datasets.append(labeled_dataset)
        # generate a dataset of wavelets WITHOUT ground truth (represented by unknown behavior)
        unknown_datasets.append(unknown_dataset)

    del loaders
    assert freq_names is not None


    unknown_dataset=pd.concat(unknown_datasets, axis=0)
    dataset=pd.concat(datasets, axis=0)
    unknown_dataset["frame_idx"]=unknown_dataset["frame_number"]%chunksize
    unknown_dataset["chunk"]=unknown_dataset["frame_number"]//chunksize
    dataset["frame_idx"]=dataset["frame_number"]%chunksize
    dataset["chunk"]=dataset["frame_number"]//chunksize
    training_set=dataset.iloc[::5]
    unknown_dataset_subset=unknown_dataset.iloc[::5]


    # train the UMAP model
    # and use it to project the training and the test set
    RECOMPUTE=True
    if RECOMPUTE:
        model=UMAP()
        umap_set=training_set[freq_names].values
        logger.debug("Fitting UMAP with data of shape %s", umap_set.shape)
        before=time.time()
        model.fit(umap_set)
        after=time.time()
        logger.debug("Done fitting UMAP in %s seconds", round(after-before, 1))
        with open(UMAP_PATH, "wb") as handle:
            pickle.dump(model, handle)
    else:
        with open(UMAP_PATH, "rb") as handle:
            model=pickle.load(handle)

    with open(os.path.join(OUTPUT_FOLDER, "freq_names.pkl"), "wb") as handle:
        pickle.dump(freq_names, handle)


    logger.debug("Transforming labeled dataset of shape %s", training_set.shape)
    projection=model.transform(training_set[freq_names].values)
    training_set["C_1"]=projection[:,0]
    training_set["C_2"]=projection[:,1]
    training_set.reset_index().to_feather(os.path.join(OUTPUT_FOLDER, "training_set.feather"))

    logger.debug("Generating visualization")
    fig=px.scatter(
        training_set.loc[training_set["behavior"].isin(["pe_inactive", "feed", "groom", "inactive", "walk"])], x="C_1", y="C_2", color="behavior",
        hover_data=["id", "chunk", "frame_idx", "frame_number", "behavior"],
    )
    fig.write_html(os.path.join(OUTPUT_FOLDER, "UMAP_by_behavior.html"))
    fig.show()


    logger.debug("Transforming unknown dataset of shape %s", unknown_dataset_subset.shape)
    before=time.time()
    unknown_projection=model.transform(unknown_dataset_subset[freq_names].values)
    after=time.time()
    logger.debug("Done transforming unknown dataset of shape %s in %s seconds", unknown_dataset.shape, round(after-before, 1))
    unknown_dataset_subset["C_1"]=unknown_projection[:,0]
    unknown_dataset_subset["C_2"]=unknown_projection[:,1]
    unknown_dataset_subset.reset_index().to_feather(os.path.join(OUTPUT_FOLDER, "unknown_set.feather"))


def generate_umap_dataset(pose_annotated, groupby="behavior", min_per_group=1000):
    
    behavior_target_count="pe_inactive"
    target_count=(pose_annotated[groupby]==behavior_target_count).sum()
    target_count=max(min_per_group, target_count)
    logger.debug("Keeping %s points per %s", target_count, groupby)
    pose_annotated_shuf=pose_annotated.sample(frac=1).reset_index(drop=True)

    pose_annotated_shuf = pose_annotated_shuf.groupby(groupby).apply(lambda x: x.iloc[:target_count]).reset_index(drop=True)
    return pose_annotated_shuf

if __name__ == "__main__":
    main()