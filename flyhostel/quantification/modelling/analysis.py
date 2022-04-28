import itertools
import argparse
import numpy as np
import pandas as pd
from flyhostel.quantification.sleep import sleep_annotation_all

class SleepAnalyser:

    def __init__(self, min_time_immobile, time_window_length, velocity_correction_coef=0.5):
        self.min_time_immobile=min_time_immobile
        self.time_window_length=time_window_length
        self.velocity_correction_coef=velocity_correction_coef

    def summarise(self, model):

        time_steps, number_of_animals = model.timeseries.shape

        data = model.timeseries.flatten()

        data = pd.DataFrame({
            "id": list(itertools.chain(*[([i,] * time_steps) for i in range(number_of_animals)])),
            "t_round": list(itertools.chain(*[list(range(time_steps)) for i in range(number_of_animals)])),
            "velocity": 1-data
        })

        analysis_params=argparse.Namespace(
            min_time_immobile=self.min_time_immobile,
            time_window_length=self.time_window_length,
            velocity_correction_coef=self.velocity_correction_coef
        )

        dt_sleep = sleep_annotation_all(data, analysis_params=analysis_params)
        # dt_sleep["t_round"] =  np.floor(dt_sleep["t"].values / 1800) * 1800
        dt_total = dt_sleep.groupby("id").agg("mean")[["moving", "asleep"]]
        return dt_total


