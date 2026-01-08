import numpy as np


def euclidean_distance(x):
    dist = np.sqrt(
        (x["center_x_A"]-x["center_x_B"])**2 +
        (x["center_y_A"]-x["center_y_B"])**2
    )
    return dist


def mean_distance(loader1, loader2):

    dtA=loader1.dt
    dtB=loader2.dt


    positions = dtA[["frame_number", "center_x", "center_y"]]\
      .rename({"center_x": "center_x_A", "center_y": "center_y_A"}, axis=1).\
      merge(
        dtB[["frame_number", "center_x", "center_y"]]\
          .rename({"center_x": "center_x_B", "center_y": "center_y_B"}, axis=1),
        on="frame_number",
        how="outer"
    )
    positions["distance"] = euclidean_distance(positions) / loader1.pixels_per_mm
    return np.round(positions["distance"].mean(), 3)


def mean_distance_while_asleep(loader1, loader2, asleep1=False, asleep2=False):


    dtA=loader1.dt
    dtB=loader2.dt

    dtA["t_round"]=1*(dtA["t"]//1)
    dtB["t_round"]=1*(dtA["t"]//1)


    dtA=dtA.merge(
        loader1.sleep[["t", "asleep"]].rename({"t": "t_round"}, axis=1),
        on="t_round"
    )
    
    dtB=dtB.merge(
        loader2.sleep[["t", "asleep"]].rename({"t": "t_round"}, axis=1),
        on="t_round"
    )


    positions = dtA[["frame_number", "center_x", "center_y", "asleep"]]\
      .rename({"center_x": "center_x_A", "center_y": "center_y_A", "asleep": "asleep_A"}, axis=1).\
      merge(
        dtB[["frame_number", "center_x", "center_y", "asleep"]]\
          .rename({"center_x": "center_x_B", "center_y": "center_y_B", "asleep": "asleep_B"}, axis=1),
        on="frame_number",
        how="outer"
    )
    positions["distance"] = euclidean_distance(positions) / loader1.pixels_per_mm
    if asleep1:
        positions=positions.query("asleep_A==True")
    if asleep2:
        positions=positions.query("asleep_B==True")
        
    return np.round(positions["distance"].mean(), 3)


def mean_distance_at_sleep_state(loader1, loader2, state="onset", animal1=False, animal2=False):

    dtA=loader1.dt
    dtB=loader2.dt

    dtA["t_round"]=1*(dtA["t"]//1)
    dtB["t_round"]=1*(dtA["t"]//1)


    dtA=dtA.merge(
        loader1.sleep[["t", state]].rename({"t": "t_round"}, axis=1),
        on="t_round"
    )
    dtB=dtB.merge(
        loader2.sleep[["t", state]].rename({"t": "t_round"}, axis=1),
        on="t_round"
    )


    positions = dtA[["frame_number", "center_x", "center_y", state]]\
      .rename({"center_x": "center_x_A", "center_y": "center_y_A", state: f"{state}_A"}, axis=1).\
      merge(
        dtB[["frame_number", "center_x", "center_y", state]]\
          .rename({"center_x": "center_x_B", "center_y": "center_y_B", state: f"{state}_B"}, axis=1),
        on="frame_number",
        how="outer"
    )
    positions["distance"] = euclidean_distance(positions) / loader1.pixels_per_mm
    if animal1:
        positions=positions.query(f"{state}_A==True")
        
    if animal2:
        positions=positions.query(f"{state}_B==True")
        
    return np.round(positions["distance"].mean(), 3)


def arousability(loader, brief_awakening=10):
    
    filter1="asleep_before==True & (asleep_after==True | (asleep_after==False & bout_out_asleep_after < @brief_awakening))"
    filter2="asleep_before==True"

    n_remains_asleep=loader.interactions_index.query(filter1).shape[0]
    n_was_asleep=loader.interactions_index.query(filter2).shape[0]
    ratio = n_remains_asleep / n_was_asleep
    return ratio


def time_touching(loader1, loader2):
    nn = loader2.ids[0]
    duration=loader1.touch.query("nn == @nn")["touch"].sum() # nframes
    duration/=loader1.framerate # seconds
    duration/=60 # mins
    return duration