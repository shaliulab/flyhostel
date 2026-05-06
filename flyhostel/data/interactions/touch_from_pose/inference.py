import itertools
import pandas as pd
from flyhostel.utils import get_framerate, animal_to_id
from flyhostel.data.pose.ethogram.utils import annotate_bouts, annotate_bout_duration
from .utils import detect_touch_pairs


def infer(features, **trainable_params):

    pairs=[]
    for experiment in features:
        animals=features[experiment].individuals.values.tolist()
        pairs.extend(list(itertools.combinations(animals, 2)))

    df_machine=[]
    for individuals in pairs:

        experiment=individuals[0].split("__")[0]
        framerate=get_framerate(experiment)

        ids=[
            animal_to_id(individual)
            for individual in individuals
        ]

        # ds has dims (time, space, keypoints, individuals)
        tf, df_machine_ = detect_touch_pairs(
            features[experiment],
            individuals=individuals,
            thorax="thorax", abdomen="abdomen",
            BODY=("head","thorax","abdomen"),
            APP=("fRL","mRL","rRL", "fLL","mLL","rLL"),
            **trainable_params
        )
        # tf is xr.DataArray over time with True on contact frames
        # df_machine=df_machine.merge(framerate_index, on="experiment", how="left")
        df_machine_=annotate_bout_duration(annotate_bouts(df_machine_, "touch"), fps=framerate)
        df_machine_["id"]=ids[0]
        df_machine_["nn"]=ids[1]
        df_machine_["experiment"]=experiment

        df_machine_["last_isolated"]=df_machine_["frame_number"].copy()
        touch_frames=df_machine_["touch"]
        df_machine_.loc[touch_frames, "last_isolated"]=df_machine_.loc[touch_frames, "frame_number"]-df_machine_.loc[touch_frames, "bout_in"]
        df_machine_reversible=df_machine_.copy()
        df_machine_reversible["id"]=ids[1]
        df_machine_reversible["nn"]=ids[0]

        df_machine.append(df_machine_)
        df_machine.append(df_machine_reversible)
    df_machine=pd.concat(df_machine, axis=0).reset_index(drop=True)
    return df_machine
