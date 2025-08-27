import logging
import os.path

import pandas as pd
from flyhostel.data.sleep import (
    PURE_INACTIVE_STATES,
    bin_apply_all,
    sleep_annotation_rf_all
)

logger=logging.getLogger(__name__)

class SleepLoader:
    """
    A class to load the result of the first step in the interactions pipeline
    """
    
    datasetnames=[]
    behavior=None
    sleep=None

    def __init__(self, *args, **kwargs):
        self.interaction=None
        self.all_interactions=None
        super(SleepLoader, self).__init__(*args, **kwargs)


    def load_sleep_data(
            self,
            min_time=None, max_time=None,
            min_time_immobile=300,
            bin_size=300,
        ):

        """
        Arguments:
        
        Populates self.sleep

        Returns
            None
        """

        dataset=self.load_sleep_data_from_file(
            min_time_immobile=min_time_immobile,
            bin_size=bin_size,
        )
        if isinstance(dataset, str):
            feather_file=dataset
            dataset=self.compute_sleep_from_behavior(
                min_time_immobile=min_time_immobile,
                bin_size=bin_size,
            )
            dataset.to_feather(feather_file)
        else:
            pass

        if "t_round" in dataset.columns:
            dataset=dataset\
                .drop("t", axis=1, errors="ignore")\
                .rename({"t_round": "t"}, axis=1, errors="ignore")

        assert "t" in dataset.columns

        if min_time is not None:
            dataset=dataset.loc[dataset["t"]>=min_time]
        
        if max_time is not None:
            dataset=dataset.loc[dataset["t"]<max_time]

        self.sleep=dataset
        self.sleep["asleep"]=self.sleep["inactive_rule"]

        return None


    def load_sleep_data_from_file(self, min_time_immobile, bin_size):
            
        min_time_immobile_min=int(min_time_immobile//60)
        root_dir=f"/home/vibflysleep/FlySleepLab_Dropbox/Antonio/FSLLab/Projects/FlyHostel4/code/scripts/figures_wo_feed/Figure1/sleep={min_time_immobile_min}min"
      
        if bin_size is None:
            feather_file=os.path.join(
                root_dir,
                f"{self.datasetnames[0]}_sleep={min_time_immobile_min}.feather"
            )
        else:
            feather_file=os.path.join(
                root_dir,
                f"{self.datasetnames[0]}_sleep={min_time_immobile_min}_bin={bin_size}.feather"
            )
            
        if os.path.exists(feather_file):
            dataset=pd.read_feather(feather_file)
            return dataset
        else:
            return feather_file


    def compute_sleep_from_behavior(self, min_time_immobile, bin_size):
        self.load_behavior_data()
        dataset=self.behavior.copy()

        
        if "inactive_states" not in dataset.columns:
            dataset["inactive_states"]=dataset["prediction2"].isin(PURE_INACTIVE_STATES)
        
        dt_sleep=sleep_annotation_rf_all(
            dataset,
            min_time_immobile=min_time_immobile,
            time_window_length=1,
            threshold=10
        )

        if bin_size is not None:
            dt_sleep=bin_apply_all(
                dt_sleep,
                feature="inactive_rule",
                summary_FUN="mean",
                x_bin_length=bin_size
            )
        return dt_sleep
