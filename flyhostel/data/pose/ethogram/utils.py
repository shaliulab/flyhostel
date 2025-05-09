import logging
import yaml
import numpy as np
from numpy.lib.stride_tricks import as_strided
from flyhostel.data.pose.constants import framerate as FRAMERATE

logger=logging.getLogger(__name__)



def load_train_test_split(file="split.yaml"):
    with open(file, "r") as handle:
        train_test_split=yaml.safe_load(handle)
    return train_test_split


def count_bout_position(df, variable, counter):
    """
    Compute position in bout of variable

    Detect bouts of rows wit the same value in column variable
    and annotate the position of each row in its bout
    """

    # Initialize a new column with zeros
    df[counter] = 0

    # Create a mask to identify the start of each bout
    bout_start_mask = df[variable] != df[variable].shift(1)

    # Create a cumulative count of bout starts
    df['bout_count'] = bout_start_mask.cumsum()

    # Calculate the rank within each bout
    df[counter] = df[["bout_count"]].groupby("bout_count").cumcount() + 1

    return df

def annotate_bout_duration(dataset, fps=FRAMERATE, on=["bout_count"]):
    duration_table=dataset.loc[dataset["bout_out"]==1, ["bout_in"] + on]
    duration_table["duration"]=duration_table["bout_in"]/fps
    dataset=dataset.drop("duration", axis=1, errors="ignore").merge(duration_table.drop("bout_in", axis=1), on=on)
    return dataset

def annotate_bouts(dataset, variable):
    """
    Annotate columns bout_out, bout_in, bout_count

    bout_in: How many rows since the bout started
    bout_out: How many rows until the bout finishes
    bout_count: How many bouts have there been already
    
    """
    dataset=count_bout_position(dataset.iloc[::-1], variable=variable, counter="bout_out").iloc[::-1]
    dataset=count_bout_position(dataset, variable=variable, counter="bout_in")
    return dataset


def identify_fluctuating_events(probabilities):
    """
    Identify fluctuating events within each second.

    :param probabilities: A NumPy array of shape (timepoints, events) representing
                          the probability of each event at each timepoint.
    :return: The number of times the winner changed
    """
    # Determine the winning event at each time point
    winning_events = np.argmax(probabilities.values, axis=1)
    # Count the number of times the winning event changes
    changes = np.count_nonzero(np.diff(winning_events) != 0)
    
    return changes


def find_window_winner(df, behaviors, time_window_length=1, other_cols=[], behavior_col="behavior"):
    # Adjust time to the nearest time_window_length
    df['t'] = (df['t'] // time_window_length) * time_window_length

    index=df[["id", "t", "frame_number"] + other_cols].groupby(["id", "t"]).first().reset_index()
    behaviors_used=[]
    for behavior in behaviors:
        if behavior in df.columns:
            behaviors_used.append(behavior)

    assert len(behaviors_used)>1, f"No behavior has its probability saved"

    logger.debug("Computing fluctuations")
    fluctuations=df[["id", "t"] + behaviors_used].groupby(["id", "t"]).apply(identify_fluctuating_events).reset_index()
    fluctuations.columns=["id", "t", "fluctuations"]


    logger.debug("Computing most abundant behavior per window")
    # Group by 'id' and 't', and get the most common behavior
    most_common_behavior = df.groupby(['id', 't', behavior_col]).size().reset_index(name='count')
    most_common_behavior = most_common_behavior.\
        sort_values(['id', 't', 'count'], ascending=[True, True, False])
    # Drop duplicates to keep the most common behavior for each group
    most_common_behavior = most_common_behavior.drop_duplicates(subset=['id', 't'])
    
    logger.debug("Computing fraction of most abundant behavior")
    # Calculate the fraction of the most common behavior
    total_counts = df.groupby(['id', 't']).size().reset_index(name='total_count')
    most_common_behavior = most_common_behavior.merge(total_counts, on=['id', 't'])
    most_common_behavior['fraction'] = np.round(
        most_common_behavior['count'] / most_common_behavior['total_count'], 2
    )

    # Select the required columns
    most_common_behavior = most_common_behavior[['id', 't', behavior_col, 'fraction']]

    logger.debug("Merging index")
    most_common_behavior=most_common_behavior.merge(index, on=["id", "t"])
    logger.debug("Merging fluctuations")
    most_common_behavior=most_common_behavior.merge(fluctuations, on=["id", "t"])
    logger.debug("find_window_winner Done")
    return most_common_behavior


def most_common_behavior_vectorized(df, time_window_length=1, other_cols=[], behavior_col="behavior"):
    # Adjust time to the nearest time_window_length
    df['t'] = (df['t'] // time_window_length) * time_window_length

    index=df[["id", "t", "frame_number"] + other_cols].groupby(["id", "t"]).first().reset_index()
    
    # Group by 'id' and 't', and get the most common behavior
    most_common_behavior = df.groupby(['id', 't', behavior_col]).size()
    most_common_behavior = most_common_behavior.reset_index(name='count')
    most_common_behavior = most_common_behavior.sort_values(['id', 't', 'count'], ascending=[True, True, False])
    
    # Drop duplicates to keep the most common behavior for each group
    most_common_behavior = most_common_behavior.drop_duplicates(subset=['id', 't'])
    
    # Calculate the fraction of the most common behavior
    total_counts = df.groupby(['id', 't']).size().reset_index(name='total_count')
    most_common_behavior = most_common_behavior.merge(total_counts, on=['id', 't'])
    most_common_behavior['fraction'] = np.round(most_common_behavior['count'] / most_common_behavior['total_count'], 2)

    # Select the required columns
    most_common_behavior = most_common_behavior[['id', 't', behavior_col, 'fraction']]

    most_common_behavior=most_common_behavior.merge(index, on=["id", "t"])
    return most_common_behavior



def remove_bout_ends_from_dataset(dataset, n_points, fps):
    """
    Set beginning and end of each bout to background, to avoid confusion between human label and wavelet defined behavior

    If the labeling strategy has more temporal resolution than the algorithm used to infer them
    there can be some artifacts where the signal inferred from a previous or future bout (very close temporally) spills over into the present bout
    This means the ground truth and inference are less likely to agree at transitions, and such frames should be labeled as such
    by setting the behavior to background (aka transition)


    Arguments:

        dataset (pd.DataFrame): contains a column called behavior and is sorted chronologically.
            All rows are equidistant in time. A single animal is present.
        n_points (int): How many points to remove at beginning AND end of each bout.
        fps (int): Number of points in this dataset that are contained within one second of recording.

    Returns:
        dataset (pd.DataFrame): rows at beginning or end of bouts are removed
    """
    dataset=annotate_bouts(dataset, variable="behavior")
    dataset=annotate_bout_duration(dataset, fps=fps)
    short_behaviors=["inactive+pe"]
    dataset.loc[((dataset["bout_in"] <= n_points) & (dataset["bout_out"] <= n_points)) | np.bitwise_not(dataset["behavior"].isin(short_behaviors)), "behavior"]="background"
    del dataset["bout_in"]
    del dataset["bout_out"]
    return dataset


def postprocessing(df, time_window_length):
    df=annotate_bout_duration(
        annotate_bouts(df, variable="behavior"),
        fps=1/time_window_length,
        on=["bout_count"]
    )

    df.loc[(df["behavior"].isin(["groom"])) & (df["fluctuations"]>0), "behavior"]="background"
    df.loc[(df["behavior"].isin(["groom"])) & (df["duration"]<5), "behavior"]="background"
    df.loc[(df["behavior"].isin(["inactive+pe"])) & (df["duration"]>3), "behavior"]="feed"

    return df


def annotate_bout_info(df_single_animal, fps, behavior=None, prediction="prediction"):
    df_single_animal.drop(["bout_in_pred","bout_out_pred","bout_count_pred","duration_pred", "bout_in","bout_out","bout_count","duration"], axis=1, inplace=True, errors="ignore")

    if behavior is not None:
        x=annotate_bout_duration(
            annotate_bouts(df_single_animal, variable=behavior),
            fps=fps,
            on=["bout_count"]
        )
    else:
        x=None
    y=annotate_bout_duration(
        annotate_bouts(df_single_animal, variable=prediction),
        fps=fps,
        on=["bout_count"]
    )
    rename_dict={
        "bout_in": "bout_in_pred",
        "bout_out": "bout_out_pred",
        "bout_count": "bout_count_pred",
        "duration": "duration_pred",
    }
    
    y.rename(rename_dict, axis=1, inplace=True)
    cols=list(rename_dict.values())

    if x is not None:
        z=x.merge(y[["frame_number", "id"] + cols], how="left", on=["frame_number", "id"])
    else:
        z=y
    return z
    

    

def generate_windows(b, window_size):
    shape=(b.size - window_size + 1, window_size)
    strides=(b.strides[0], b.strides[0])
    windowed_view=as_strided(b, shape=shape, strides=strides)
    return windowed_view


def annotate_active_state(dataset, y_true, y_pred, inactive_states):

    dataset["active.pr"]="active"
    dataset["active.gt"]="active"
    dataset.loc[dataset[y_pred].isin(inactive_states), "active.pr"]="inactive"
    dataset.loc[dataset[y_true].isin(inactive_states), "active.gt"]="inactive"
    return dataset



