import matplotlib.pyplot as plt
from .core import geom_ld_annotation, set_xticks

def geom_env_data(data, ax):

    ax2 = ax.twinx()
    ax2.set_ylabel("Temp ºC")
    ax.set_ylabel("% Hum")

    # ax.set_xticks(list(range(*xrange)) + [xrange[-1]])
    # ax.set_yticks(list(range(*yrange)) + [yrange[-1]])
    # ax.set_aspect(1)
    ax.scatter(data["t"] / 3600, data["humidity"], s=0.1)
    ax2.scatter(data["t"] / 3600, data["temperature"], c="red", s=0.1)
    return ax, ax2


def make_environment_plot(root, data, title=""):
    make_environment_log_plot(root=root, data=data,title=title)
    make_light_log_plot(root=root, data=data)


def make_environment_log_plot(root, data, title=""):
    fig = plt.figure(1, figsize=(5, 5), dpi=90)
    ax = fig.add_subplot(111)
    ax = geom_ld_annotation(data, ax, yrange=(0, 100))
    geom_env_data(data, ax)
    ax.set_title(title)
    plt.tight_layout()
    dest = root + "_environment-log.png"
    fig.savefig(dest)
    plt.close(fig)



def make_light_log_plot(root, data):

    light_log_data = data.copy()
    light_log_data["t"] /= 3600

    fig = plt.figure(2, figsize=(5, 5), dpi=90)
    ax = fig.add_subplot(111)
    ax = geom_ld_annotation(data, ax, yrange=(0, light_log_data["light"].max()))
    ax.scatter(light_log_data["t"], light_log_data["light"])
    ax = set_xticks(ax, 6, light_log_data["t"].min(), light_log_data["t"].max())
    dest = root + "_light-log.png"
    fig.savefig(dest)
    plt.close(fig)

