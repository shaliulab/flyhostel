import pandas as pd

from flyhostel.plotting.ethogram import reshape_ethogram
from flyhostel.quantification.constants import FLYHOSTEL_ID


def prepare_data_for_ethogram_plot(data, analysis_params):
    
    dt_ethogram=[]
    
    for id in set(data[FLYHOSTEL_ID]):
        # pick a single animal
        single_animal=data.loc[data[FLYHOSTEL_ID] == id]
        # get the timeseries
        timeseries = pd.DataFrame(single_animal["velocity"])
        # discretize movement
        timeseries = timeseries < analysis_params.velocity_correction_coef
        timeseries *= 1

        # keep L annotation
        timeseries["L"] = single_animal["L"]
        timeseries[FLYHOSTEL_ID] = single_animal[FLYHOSTEL_ID]
        timeseries["fly_no"] = single_animal["fly_no"]

        dt_ethogram.append(timeseries)

    dt_ethogram=pd.concat(dt_ethogram)

    return dt_ethogram