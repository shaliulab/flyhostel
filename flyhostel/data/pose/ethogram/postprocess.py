import logging
import os.path
logger=logging.getLogger(__name__)
DEEPETHOGRAM_PROJECT_PATH=os.environ["DEEPETHOGRAM_PROJECT_PATH"]

import numpy as np
try:
    from deepethogram.postprocessing import remove_short_bouts_from_trace, get_bout_length_percentile, compute_background
    from deepethogram import projects, file_io
except ModuleNotFoundError:
    remove_short_bouts_from_trace=None
    projects=None
    file_io=None
    logger.error("Please install deepethogram without dependencies (pip install --no-deps deepethogram")



def get_bout_length_percentile_from_project(project_path, percentile=1, behaviors=None):
    records = projects.get_records_from_datadir(os.path.join(project_path, "DATA"))
    label_list = []
    for animal, record in records.items():
        labelfile = record['label']
        if labelfile is None:
            continue
        label = file_io.read_labels(labelfile)
        # ignore partially labeled videos
        if np.any(label == -1):
            continue
        label_list.append(label)
    
    percentiles = get_bout_length_percentile(label_list, percentile, behaviors=behaviors)
    if behaviors is None:
        behavior_list=projects.get_classes_from_project(project_path)
        percentiles={behavior_list[i]: percentiles[i] for i in range(len(percentiles))}
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

def postprocess_behaviors(dataset, percentile=1, column="behavior", behaviors=None):
    """
    Remove short bouts of behaviors and join simultanous behaviors
    
    A short bout of behavior is a bout shorter than the _percentile_ percentile bout length
    in the DEG database
    """
    
    if behaviors is None:
        unique_behaviors=dataset[column].unique().tolist()
    else:
        unique_behaviors=list(behaviors.keys())

    if "background" in unique_behaviors:
        unique_behaviors.pop(unique_behaviors.index("background"))
    unique_behaviors=["background"] + unique_behaviors
    predictions=one_hot_encoding(dataset[column], unique_behaviors)

    bout_length_dict=get_bout_length_percentile_from_project(DEEPETHOGRAM_PROJECT_PATH, percentile=percentile, behaviors=behaviors)
    logger.debug("Bout length cutoff %s", bout_length_dict)
    
    bout_lengths=[int(bout_length_dict.get(behav, 1)) for behav in unique_behaviors]
    predictions_smoothed = []
    T, K = predictions.shape
    for i in range(K):
        trace = predictions[:, i]
        print(f"{percentile}% percentile bout length for {unique_behaviors[i]}={bout_lengths[i]}")
        trace = remove_short_bouts_from_trace(trace, bout_lengths[i])
        predictions_smoothed.append(trace)

    predictions = np.stack(predictions_smoothed, axis=1)
    confusing_rows=predictions.sum(axis=1)>1
    predictions[confusing_rows, :]=0
    predictions = compute_background(predictions)
    rows,cols=np.where(predictions==1)
    prediction=[unique_behaviors[i] for i in cols]
    dataset[column]=prediction
    return dataset
