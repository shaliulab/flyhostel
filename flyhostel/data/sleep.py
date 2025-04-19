import numpy as np
from ethoscopy.analyse import sleep_contiguous
INACTIVE_STATES=["inactive", "inactive+pe", "inactive+micromovement", "inactive+rejection", "feed"]
PURE_INACTIVE_STATES=["inactive", "inactive+pe", "inactive+micromovement", "inactive+rejection"]
SLEEP_STATES={"WO_FEED": PURE_INACTIVE_STATES, "WITH_FEED": INACTIVE_STATES}

# Sleep functions
def apply_inactive_rule(dataset_window, time_window_length, min_time_immobile):
    return sleep_contiguous(
            ~dataset_window["windowed_var"],
            1/time_window_length,
            min_valid_time=min_time_immobile
        )


## Apply filter to remove high speed noise
def high_speed_noise_filter(dataset, time_window_length, aggregation, threshold, feature="inactive_states"):
    dataset["zt"]=dataset["t"]/3600
    dataset["t_round"]=time_window_length*(dataset["t"]//time_window_length)

    dataset_window=dataset.groupby(["id", "t_round"]).agg(aggregation).reset_index()

    # This plot shows why 10 is a nice threshold for the filter above
    # fps=30 (there are 30 points per second)
    # _ = plt.hist(dataset_window[feature], bins=int(fps*time_window_length) + 1)

    dataset_window["windowed_var"]=dataset_window[feature]>threshold
    dataset_window["t"]=dataset_window["t_round"]
    return dataset_window


def sleep_annotation_rf(data, time_window_length=1, min_time_immobile=300, threshold=10, feature="inactive_states"):
    """
    data (pd.DataFrame). Dataset with columns feature, t
    """
    # Filtering parameters
    #
    # We perform an aggregation of time in windows
    # to filter high frequency (> 1/time_window_length Hz) noise

    # duration of the windows for which a behavior will be produced, in seconds
    time_window_length

    # name of the column being summarised in windows
    feature

    # how many points per window where the feature is set to True are required to set the whole window as True
    agg_function=np.sum

    # TODO
    # You may want to handle scenarios like FlyHostel4_6X_2023-06-20_14-00-00__03 on frame 2318591
    # within two seconds, it performs a jump, which produces a high centroid speed for that 5 frame bin
    # but it is so brief (around 10 frames in total) that the ethogram is just micromovement there

    # Downstream functions could be informed of a very brief high speed behavior by keeping the max centroid_speed of the window
    # not just the sum which makes it the brief peak "blurry" if the fly is not moving the rest of the window
    aggregation={feature: agg_function, "centroid_speed": np.sum, "walk": agg_function}
    dataset_window=high_speed_noise_filter(data, time_window_length, aggregation, threshold, feature=feature)
    # quantify sleep

    dataset_window["inactive_rule"]=apply_inactive_rule(dataset_window, time_window_length, min_time_immobile)
    return dataset_window


def sleep_annotation_rf_all(data, **kwargs):

    dt_sleep=data.groupby("id").apply(
        lambda df: sleep_annotation_rf(df, **kwargs)
    ).reset_index(drop=True)
    return dt_sleep


def bin_apply_all(data, feature, summary_FUN, x_bin_length):
    # Assign quantity to a column with this name
    aggregation={feature: summary_FUN}
    data["t_round"]=x_bin_length*(data["t"]//x_bin_length)
    grouping_columns=["id", "t_round"]
    dt_bin=data.drop(
        "zt", axis=1, errors="ignore"
    ).groupby(grouping_columns).agg(aggregation).reset_index()
    return dt_bin
