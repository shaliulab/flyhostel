import os.path


def adjust_to_zt0(experiment, timestamps, zt0, reference="start_time"):
    """

    Arguments:
        experiment (str): Path to imgstore folder where the name of the folder encodes the start of the experiment
        in format YYYY-MM-DD_HH-MM-SS
        timestamps (np.ndarray): Timestamps in ms
        zt0 (int): Time of the day in GMT when the incubator turned the lights ON, in hours
    """

    time_str=os.path.basename(experiment).split("_")[1]
    hours, minutes, seconds = [int(e) for e in time_str.split("-")]
    start_time = hours*3600 + minutes * 60 + seconds
    # convert to ms
    start_time *= 1000
    zt0 *= 3600 * 1000
    if reference == "start_time":
        # if start_time = 10 and zt0 = 6
          # this has the result of adding 4 hours to the timestamps

        # if  start _time = 4 and zt0 = 6
          # this has the effect of removing 2 hours to the timestamps
          # (signaling they started BEFORE the zt0)
        timestamps=timestamps + start_time - zt0
    elif reference == "zt0":
        # if start_time = 10 and zt0 = 6
          # this has the effect of removing 4 hours to the timestamps
        timestamps=timestamps - start_time + zt0

    return timestamps 
    