import logging
import time
import logging
import os.path
import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

from flyhostel.data.pose.ethogram_utils import annotate_bout_duration, annotate_bouts
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.pose.constants import chunksize, framerate

logger=logging.getLogger(__name__)
try:
    from deepethogram.postprocessing import remove_short_bouts_from_trace, get_bout_length_percentile, compute_background
    from deepethogram import projects, file_io
except ModuleNotFoundError:
    remove_short_bouts_from_trace=None
    projects=None
    file_io=None
    logger.error("Please install deepethogram without dependencies (pip install --no-deps deepethogram")


MOTIONMAPPER_DATA=os.environ["MOTIONMAPPER_DATA"]
DEEPETHOGRAM_PROJECT_PATH=os.environ["DEEPETHOGRAM_PROJECT_PATH"]

FLYHOSTEL_VIDEOS=os.environ["FLYHOSTEL_VIDEOS"]
OUTPUT_FOLDER=os.path.join(MOTIONMAPPER_DATA, "output")
MODELS_PATH=os.path.join(MOTIONMAPPER_DATA, "models")

from motionmapperpy import setRunParameters
STRIDE=setRunParameters().wavelet_downsample

MODEL_PATH=os.path.join(MODELS_PATH, "2024-01-15_07-30-25_rf.pkl")
RECOMPUTE=True

def get_bout_length_percentile_from_project(project_path, percentile=1):
    records = projects.get_records_from_datadir(os.path.join(project_path, "DATA"))
    label_list = []
    behaviors=projects.get_classes_from_project(project_path)
    for animal, record in records.items():
        labelfile = record['label']
        if labelfile is None:
            continue
        label = file_io.read_labels(labelfile)
        # ignore partially labeled videos
        if np.any(label == -1):
            continue
        label_list.append(label)
    
    percentiles = get_bout_length_percentile(label_list, percentile)
    percentiles={behaviors[i]: percentiles[i] for i in range(len(percentiles))}
    return percentiles


def generate_path_to_output_folder(experiment, identity):
    tokens=experiment.split("_")
    key=os.path.sep.join([tokens[0], tokens[1], "_".join(tokens[2:4])])
    basedir=os.path.join(FLYHOSTEL_VIDEOS, key)
    output_folder=os.path.join(basedir, "motionmapper", str(identity).zfill(2))
    return output_folder

def generate_path_to_data(experiment, identity):
    folder=generate_path_to_output_folder(experiment, identity)
    data_src=os.path.join(folder, experiment + "__" + str(identity).zfill(2) + ".csv")
    # assert os.path.exists(data_src), f"{data_src} not found"
    return data_src


# Group by 't' and find the most common 'foo' for each group
# Modified function to include data from all other columns in the first row of each group
def most_common(group, variable="behavior"):
    most_common_val = group[variable].value_counts().idxmax()
    score = group[variable].value_counts().max() / len(group)
    
    # Select the first row of the group for additional data
    first_row = group.iloc[0]
    
    # Prepare the result including additional data
    result = first_row.to_dict()
    result[variable] = most_common_val
    result['fraction'] = np.round(score, 2)
    
    return pd.Series(result)

def set_time_resolution(df, time_window_length):
    df["t"]=np.floor(df["t"]//time_window_length)*time_window_length
    df = df.groupby(["id", "t"]).apply(most_common).reset_index(drop=True)
    return df


def enforce_behavioral_context(dataset, modify, context, replacement, seconds=5, framerate=1):
    """
    Makes sure the behavior surrounding another behavior for n seconds is fixed to the behaviors provided in context

    This is useful to for example, correct spurious bouts of pe_inactive which are not surrounded by pe_inactive
    (although the animal might correctly be presumed to be inactive _during_ the bout)

    dataset=enforce_behavioral_context(dataset, modify="pe_inactive", context=["inactive"], replacement="pe", seconds=5, framerate=1)
    """
    
    n = int(seconds * framerate) # Number of rows to consider before first and after last 'foo'
    
    # Find all rows where behavior is set to 'foo'
    modify_rows = dataset['behavior'] == modify

    behavior_bouts=dataset.loc[modify_rows]
    bouts=behavior_bouts["bout_count"].unique()
    

    # bout_start=df.loc[(modify_rows) & (df["bout_in"]==1), "frame_number"].values
    # bout_end=df.loc[(modify_rows) & (df["bout_out"]==1), "frame_number"].values
    bout_start=np.where(np.bitwise_and(modify_rows, dataset["bout_in"]==1))[0].tolist()
    bout_end=np.where(np.bitwise_and(modify_rows, dataset["bout_out"]==1))[0].tolist()
    assert len(bout_start)==len(bout_end) == len(bouts)

    suppressed=[]
    for i, (start_idx, end_idx) in enumerate(tqdm(zip(bout_start, bout_end), desc=f"Verifying {modify} bouts", total=len(bouts))):
        before_test=(dataset.iloc[(start_idx-n):start_idx]["behavior"].isin(context)).all()
        after_test=(dataset.iloc[end_idx:(end_idx+n)]["behavior"].isin(context)).all()
        if before_test and after_test:
            suppressed.append(False)
        else:
            suppressed.append(True)
            bout_count=bouts[i]
            dataset.loc[(dataset["behavior"]==modify) & (dataset["bout_count"]==bout_count), "behavior"] = replacement

    return dataset


def compute_bout_length_percentile(project_path, percentile=1):
    records = projects.get_records_from_datadir(os.path.join(project_path, "DATA"))
    label_list = []
    behaviors=projects.get_classes_from_project(project_path)
    for animal, record in records.items():
        labelfile = record['label']
        if labelfile is None:
            continue
        label = file_io.read_labels(labelfile)
        # ignore partially labeled videos
        if np.any(label == -1):
            continue
        label_list.append(label)
    
    percentiles = get_bout_length_percentile(label_list, percentile)
    percentiles={behaviors[i]: percentiles[i] for i in range(len(percentiles))}
    return percentiles


def one_hot_encoding(strings, unique_strings):
    # Map each unique string to its index
    index_map = {string: index for index, string in enumerate(unique_strings)}

    # Initialize the encoding table with zeros
    encoding_table = [[0] * len(unique_strings) for _ in strings]

    # Set the corresponding column to 1
    for i, string in enumerate(strings):
        if string in index_map:  # Only if the string is in the unique strings
            encoding_table[i][index_map[string]] = 1

    return np.array(encoding_table)


def join_strings_by_repeated_integers(integers, strings):
    """
    join_strings_by_repeated_integers(rows, prediction)
    """
    grouped_strings = {}
    for integer, string in zip(integers, strings):
        if integer in grouped_strings:
            grouped_strings[integer] += "+" + string
        else:
            grouped_strings[integer] = string
    return list(grouped_strings.values())


def load_dataset(experiment, identity, wavelets=None, cache="/flyhostel_data/cache", **kwargs):

    loader=FlyHostelLoader(experiment=experiment, identity=identity, chunks=range(0, 400))
    loader.load_and_process_data(
        stride=STRIDE,
        cache=cache,
        filters=None,
        useGPU=0,
        **kwargs
    )
    out=loader.load_dataset(segregate=False, wavelets=wavelets, deg=loader.deg)

    dataset, (frequencies, freq_names)=out
    dataset.sort_values(["id","frame_number"], inplace=True)
    return dataset, (frequencies, freq_names)
    
def make_ethogram(
        experiment, identity, model_path, input=None, output="./",
        cache="/flyhostel_data/cache", frame_numbers=None, postprocess=True,
        t0=None,
        train=RECOMPUTE,
        **kwargs):
    """
    Generate ethogram for a particular fly

    experiment
    identity
    model_path: Path to pkl file which when loaded yields
     1) an object with a predict method
     2) a list of columns that should be used as features
    """


    dataset, (frequencies, freq_names)=load_dataset(experiment=experiment, identity=identity, cache=cache, **kwargs)
    features=freq_names

    logger.debug("Read dataset of shape %s", dataset.shape)
    dataset["chunk"]=dataset["frame_number"]//chunksize
    dataset["frame_idx"]=dataset["frame_number"]%chunksize
    dataset["zt"]=(dataset["t"]/3600).round(2)
    if frame_numbers is not None:
        dataset=dataset.loc[
            dataset["frame_number"].isin(frame_numbers)
        ]

    if train:
        classifier=RandomForestClassifier(n_estimators=100, random_state=42)
        X_train=dataset[features].values
        y_train=dataset["behavior"].values
        classifier.fit(X_train, y_train)

    else:
        with open(model_path, "rb") as handle:
            classifier, features=pickle.load(handle)
    
    behaviors=classifier.classes_    
    logger.debug("Predicting behavior of %s rows", dataset.shape[0])
    before=time.time()
    probs=classifier.predict_proba(dataset[features].values)
    after=time.time()
    logger.debug(
        "Done in %s seconds (%s points/s or %s recording seconds / s)",
        round(after-before, 2),
        round(dataset.shape[0]/(after-before)),
        round((dataset.shape[0]/(framerate/STRIDE))/(after-before))
    )

    dataset["behavior"]=behaviors[probs.argmax(axis=1)]
    dataset["score"]=probs.max(axis=1)
    dataset["behavior_raw"]=dataset["behavior"].copy()


    if postprocess:
        logger.debug("Postprocessing predictions")
        unique_behaviors=dataset["behavior"].unique().tolist()
        if "background" in unique_behaviors:
            unique_behaviors.pop(unique_behaviors.index("background"))
        unique_behaviors=["background"] + unique_behaviors
        predictions=one_hot_encoding(dataset["behavior"], unique_behaviors)

        bout_length_dict=get_bout_length_percentile_from_project(DEEPETHOGRAM_PROJECT_PATH, percentile=1)
        bout_length_dict["pe_inactive"]=1
        bout_length_dict["inactive"]=6
        logger.debug("Bout length cutoff %s", bout_length_dict)
        
        bout_lengths=[int(bout_length_dict.get(behav, 1)) for behav in unique_behaviors]
        predictions_smoothed = []
        T, K = predictions.shape
        for i in range(K):
            trace = predictions[:, i]
            trace = remove_short_bouts_from_trace(trace, bout_lengths[i])
            predictions_smoothed.append(trace)
        predictions = np.stack(predictions_smoothed, axis=1)

        predictions = compute_background(predictions)

        rows,cols=np.where(predictions==1)
        prediction=[unique_behaviors[i] for i in cols]
        prediction=join_strings_by_repeated_integers(rows, prediction)
        dataset["behavior"]=prediction

    dataset=annotate_bouts(dataset, "behavior")
    dataset=annotate_bout_duration(dataset, fps=framerate/STRIDE)
    # dataset=enforce_behavioral_context(dataset, modify="pe_inactive", context=["inactive", "background", "pe_inactive"], replacement="pe_unknown", framerate=1, seconds=3)
    # dataset=annotate_bouts(dataset, "behavior")
    # dataset=annotate_bout_duration(dataset)

    feather_out=os.path.join(output, "dataset.feather")
    logger.info("Saving to ---> %s", feather_out)
    dataset[["id", "frame_number", "t", "behavior_raw","behavior", "score"]].reset_index(drop=True).to_feather(feather_out)


