import os.path

import pandas as pd
import matplotlib.pyplot as plt

from .core import geom_ld_annotation 

def sleep_plot(dt, plotting_params):

    for column in ["L", "asleep", "t", "id"]:
        assert column in dt.columns, f"{column} is not available"

    identities = dt["id"].unique()
    fig = plt.figure(1, figsize=(10, 7), dpi=90, facecolor="white")
    ncol = 1
    nrow = len(identities)
    axes = []
    Y_RANGE = (0, 1)

    for i, ident in enumerate(identities):
        int_str = f"{nrow}{ncol}{i+1}"
        axes.append(fig.add_subplot(int(int_str)))

        # take one fly
        dt_one_fly = dt.loc[dt["id"] == ident].reset_index()

        if plotting_params.ld_annotation:
            # plot the phases (LD)
            axes[i] = geom_ld_annotation(dt_one_fly, axes[i], yrange=Y_RANGE)

        # plot the data
        axes[i].plot(
            dt_one_fly["t"] / 3600, dt_one_fly["asleep"], linewidth=1, c="blue"
        )

        axes[i].set_ylim(Y_RANGE)
        axes[i].set_yticks([0, .5, 1])
        axes[i].set_yticklabels(["0", "50", "100"])
        axes[i].set_xlabel("ZT")

    # fig.subplots_adjust(bottom=0.0, right=0.8, top=1.0)
    fig.suptitle(f"Fly Hostel - {plotting_params.experiment_name}")
    return fig



# def make_sleep_gif(imgstore_folder, analysis_folder):
#     # dt_binned_path = os.path.join(analysis_folder, "dt_binned.csv")

#     # assert os.path.exists(
#     #     dt_binned_path
#     # )
    
#     # dt_binned = pd.read_csv(dt_binned_path)

#     # store_metadata, chunk_metadata = read_store_metadata(
#     #     imgstore_folder, chunk_numbers=[0,1]
#     # )
#     # analysis_params, plotting_params = load_params(store_metadata)
    
#     # experiment_name = os.path.basename(
#     #     os.path.realpath(os.path.basename(imgstore_folder.rstrip("/")))
#     # )

#     plotting_params.number_of_animals = 1
#     plotting_params.experiment_name = experiment_name
#     plotting_params.ld_annotation = True
  
#     fig1 = sleep_plot(
#         dt_binned,
#         plotting_params=plotting_params
#     )
#     return dt_binned, fig1


