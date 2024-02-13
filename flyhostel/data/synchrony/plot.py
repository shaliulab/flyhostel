

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import colorsys


sns.set_style("whitegrid", {
    'grid.color': '.9',
    'grid.linestyle': '--'
})
sns.set_palette("colorblind")  # This palette is somewhat similar to some Prism palettes


def hue_to_rgb(hue):
    """Convert a hue value (0-1) to an RGB value."""
    return colorsys.hsv_to_rgb(hue, 1, 1)  # Assuming full saturation and value

def normalize(x, min_value, max_value):
    """Normalize an array of values to range [0, 1]."""
    return (x - min_value) / (max_value - min_value)


def synchrony_lag(dt, label="WT", correlation="corr", out="correlation.png"):
    plt.figure(figsize=(7, 8))
    this_dt=dt.loc[dt["label"]==label]
    dt_isol=this_dt.loc[this_dt["experiment"].str.startswith("1X")]
    dt_grp=this_dt.loc[this_dt["experiment"].str.startswith("FlyHostel")]

    for i, dff in enumerate([dt_isol, dt_grp, None]):

        if i == 0:
            this_lab=f"Isol {label}"
            ax = sns.lineplot(data=dff, x=dff["lag"]*300/60, y=correlation, linewidth= 5, color="#4285f4", markers=True, dashes=False, label=this_lab)
        elif i == 1:
            this_lab=f"Grp {label}"
            ax = sns.lineplot(data=dff, x=dff["lag"]*300/60, y=correlation, linewidth= 5, color="#FF0000", markers=True, dashes=False, label=this_lab)
            for experiment in dff["experiment"].unique():
                dff2 = dff.loc[dff["experiment"]==experiment]
                sns.lineplot(data=dff2, x=dff2["lag"]*300/60, y=correlation, color="#FF0000", linewidth= 2, alpha=0.4, markers=True, dashes=False)
        # else:
        #     label="Grp per"        
        #     ax = sns.lineplot(data=dff, x=dff["lag"]*300/60, y=correlation, linewidth= 5, color="#FFC000", markers=True, dashes=False, label=label)
        ax.spines['left'].set_color('gray')
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_color('gray')
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    ax.set_xlabel("Lag (min.)", fontsize=20)
    ax.set_ylabel(r'$\overline{\rho}$', fontsize=25, labelpad=-15)
    plt.xticks(np.arange(-30, 31, 10), fontsize=17)  # Change fontsize as needed
    plt.yticks(np.arange(-.1, .401, 0.1), fontsize=17)  # Change fontsize as needed
    plt.savefig(out)