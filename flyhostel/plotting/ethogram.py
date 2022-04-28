import matplotlib.pyplot as plt
import numpy as np

def ethogram_plot_single(
    fig, i, data, plotting_params, colors, ncols
):
     
    single_animal = data.loc[data["id"] == i]
    timeseries = single_animal["velocity"].values

    timeseries_phase = single_animal["L"].values
    timeseries = np.array(
        [
            colors[timeseries_phase[i]][timeseries[i]]
            for i in range(len(timeseries))
        ]
    )
    timeseries=reshape_ethogram(timeseries, ncols)

    noa=plotting_params.number_of_animals
    int_str = f"1{noa}{int(i)+1}"
    ax = fig.add_subplot(int(int_str))

    if ax is None:
        ax = plt

    nrows, ncols = timeseries.shape[:2]
    pos = list(range(0, 1+int(nrows / noa) * noa, 3600 // plotting_params.ethogram_frequency))

    ticks = []
    positions = []
    for p in pos:
        if p in plotting_params.chunk_index:
            ticks.append(plotting_params.chunk_index[p])
            positions.append(p)

    if ticks is not None:
        ax.set_yticks(positions, ticks)
        ax.set_xticks([0, ncols-1], [0, plotting_params.ethogram_frequency])
        ax.set_xlabel("Time in chunk (s)")

    ax.imshow(timeseries)

    if i == 0:
        ax.set_ylabel("ZT")

    return ax


def reshape_ethogram(data, ncols):
    nrows = int(data.shape[0] / ncols)
    data = data[: nrows * ncols]
    data = data.reshape((nrows, ncols, -1))
    return data


def hex_to_tuple(hex_code):
    h = hex_code.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))



def ethogram_plot(data, analysis_params, plotting_params, **kwargs):

    fig = plt.figure(2, figsize=(12, 7), dpi=90, facecolor="white")
    plt.axis("off")
    plt.title(plotting_params.experiment_name)

    for i in set(data["id"].tolist()):
        ax = ethogram_plot_single(
            fig,
            i,
            data,
            plotting_params=plotting_params,
            **kwargs
        )

    return fig