import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from flyhostel.data.pose.ethogram.plot import bin_behavior_table
from flyhostel.data.pose.ethogram_utils import annotate_bout_duration, annotate_bouts
from flyhostel.data.pose.constants import framerate as FRAMERATE

def find_brief_interruptions(behavior, framerate, window_size=60, time_window_length=1, target_states=["inactive"], noise_states=None):
    """
    Mark bouts of interruptions in the target_states and compute how much time was spent in the target states
    window_size seconds in the future and the past.

    Arguments
        behavior (pd.DataFrame): Contains columns behavior, frame_number
        framerate (int): Contains the framerate of the original recording i.e. how much does frame number change in one second
        window_size (int): How many seconds to count in the past and future of the interruption to quantify target state time
        time_window_length (int): How many seconds to bin to produce one behavioral label.
        target_states (list): Behaviors that could be interrupted.
        noise_states (list): Behaviors that could be interrupting the target_states. If None, all behaviors not in target_states are included.

    Returns:
        intervals_table (pd.DataFrame): For every bout, it contains:
            frame_number: the frame number when the bout begins
            inactive: the fraction of the window_size in the future and past spent in the inactive state
            id:
            duration: duration of the target_states bout in seconds
            bout_count: bout identifier
    """
    try:
        behavior=behavior.to_pandas()
    except:
        pass

    behaviors=behavior["behavior"].unique()

    if noise_states is None:
        noise_states=[behavior for behavior in behaviors if behavior not in target_states]
    
    behavior, _=bin_behavior_table(behavior, time_window_length=time_window_length)
    behavior["bout_interrupt"]=False
    behavior.loc[
        behavior["behavior"].isin(target_states),
        "bout_interrupt"
    ]=False
    behavior.loc[
        behavior["behavior"].isin(noise_states),
        "bout_interrupt"
    ]=True



    behavior=annotate_bouts(behavior, "bout_interrupt")
    behavior=annotate_bout_duration(behavior, fps=1)
    
    bout_start=behavior.loc[(behavior["bout_interrupt"]) & (behavior["bout_in"]==1)]
    bout_end=behavior.loc[(behavior["bout_interrupt"]) & (behavior["bout_out"]==1)]
    bout_end.shape[0]==bout_start.shape[0]
    bout_end["frame_number"]+=int(framerate*time_window_length)

    bouts=pd.merge(bout_start, bout_end[["bout_count", "frame_number"]], on=["bout_count"]).rename(
           {"frame_number_x": "bout_start", "frame_number_y": "bout_end"}, axis=1
        ).sort_values(
            "bout_start"
        )
    assert ((bouts["bout_end"]-bouts["bout_start"]) > 0).all()
    
    windows=np.hstack([bouts[["bout_start"]].values - framerate*window_size, bouts[["bout_start"]].values, bouts[["bout_end"]].values, bouts[["bout_end"]].values + framerate*window_size])
    
    intervals_table=[]
    row_index=[]
    for i in tqdm(range(windows.shape[0])):
        rows=np.bitwise_or(
            (behavior["frame_number"]>=windows[i, 0]) & (behavior["frame_number"]<windows[i, 1]),
            (behavior["frame_number"]>=windows[i, 2]) & (behavior["frame_number"]<windows[i, 3])
        )
        interruption_rows=np.bitwise_and(
            (behavior["frame_number"]>=windows[i, 1]), (behavior["frame_number"]<windows[i, 2])
        )
        bout_count=behavior.loc[behavior["frame_number"]==windows[i, 1], "bout_count"].item()
        try:
            interruption_df=behavior.loc[interruption_rows]
            interrupting_behavior, count=interruption_df.value_counts("behavior", ascending=False).reset_index().iloc[0].values.tolist()
            score=interruption_df.loc[interruption_df["behavior"]==interrupting_behavior, "score"]
            mean, minimum, maximum=np.mean(score), np.min(score), np.max(score)
            fraction=count/interruption_rows.sum()
        except:
            interrupting_behavior=None
            mean=minimum=maximum=fraction=None
            import ipdb; ipdb.set_trace()

        intervals_table.append((i, bout_count, interrupting_behavior, fraction, mean, minimum, maximum))
        row_index.append(rows)
    intervals_table=pd.DataFrame.from_records(intervals_table, columns=["id", "bout_count", "interruption", "purity", "mean_score", "min_score", "max_score"])
    intervals_table=intervals_table.merge(behavior[["duration", "bout_count"]].drop_duplicates(), on="bout_count", how="left")

    intervals_table["frame_number"]=[windows[i, 1] for i in range(len(windows))]
    intervals_table["consistency"]=None


    
    for i in range(intervals_table.shape[0]):
        consistency_mean=behavior.loc[
                row_index[i],
                "behavior"
            ].isin(target_states).mean()
        
        intervals_table["consistency"].iloc[i]=consistency_mean
    
    return intervals_table


def list_extremely_brief_interruptions(interruptions, interactions=None, min_interaction=300):
    """
    Cross interruptions data with interactions to ignore bouts where an interaction happened within min_interaction frames

    Returns:
        extremely_brief_bouts (pd.DataFrame): A list of interruptions with high inactive time and low chance of interaction in the temporal vicinity
        Many of these bouts could be False Positives
    """
    records=[]
    
    for frame_number in interruptions["frame_number"]:
        if interactions is None:
            min_value=np.inf
            row=0
        else:
            diff=interactions["frame_number"].values-frame_number
            diff_abs=np.abs(diff)
            row=np.argmin(diff_abs)
            min_value=diff[row]
        records.append((min_value, row))
    out=pd.DataFrame.from_records(records, columns=["lag", "interaction_iloc"])
    interruptions=pd.concat([interruptions, out], axis=1)
    
    extremely_brief_bouts=interruptions.loc[(np.abs(interruptions["lag"])>min_interaction)]
    return extremely_brief_bouts


def main(group, framerate, target_states, noise_states=None, time_window_length=1, min_consistency=0.98, interactions=None, **kwargs):

    out=[]
    for fly in group.flies.values():
        if fly.behavior is None or fly.behavior.shape[0] == 0:
            raise ValueError(f"Fly {fly.identity} has no behavior dataset available")
        df=find_brief_interruptions(
            fly.behavior, framerate=framerate, time_window_length=time_window_length,
            target_states=target_states, noise_states=noise_states,
            **kwargs
        )
        df["id"]=fly.ids[0]
        out.append(df)
    interruptions=pd.concat(out, axis=0)

    if interactions is None:
        extremely_brief_bouts=interruptions.copy()
    else:
        extremely_brief_bouts=[]
        for id in group.ids:
            extremely_brief_bouts.append(
                list_extremely_brief_interruptions(
                    interruptions.loc[interruptions["id"]==id],
                    interactions.loc[interactions["id"]==id],
                )
            )
        extremely_brief_bouts=pd.concat(extremely_brief_bouts, axis=0)
    
    # require at least min_inactive fraction of time in the target_states
    extremely_brief_bouts=extremely_brief_bouts.loc[(np.abs(extremely_brief_bouts["consistency"])>min_consistency)]

    ids_to_animals={group.ids[i]: group.animals[i] for i in range(len(group.ids))}
    extremely_brief_bouts["animal"]=[ids_to_animals[id] for id in extremely_brief_bouts["id"]]

    return extremely_brief_bouts