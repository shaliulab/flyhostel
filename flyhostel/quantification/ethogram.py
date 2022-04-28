import pandas as pd

from flyhostel.plotting.ethogram import reshape_ethogram


def prepare_data_for_ethogram_plot(data, analysis_params):
    
    dt_ethogram=[]
    
    for i in set(data["id"]):
        # pick a single animal
        single_animal=data.loc[data["id"] == i]
        # get the timeseries
        timeseries = pd.DataFrame(single_animal["velocity"])
        # discretize movement
        timeseries = timeseries < analysis_params.velocity_correction_coef
        timeseries *= 1

        # keep L annotation
        timeseries["L"] = single_animal["L"]
        timeseries["id"] = single_animal["id"]
        
        dt_ethogram.append(timeseries)

    dt_ethogram=pd.concat(dt_ethogram)

    return dt_ethogram