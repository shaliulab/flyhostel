import matplotlib.pyplot as plt
import numpy as np
from plotting import plot_stat

def select_pixels(sd_fly):
    # plot the std of the pixels that contain 90 % of the variance
    # for the rest, set them to black
    sd_fly_flattened=sd_fly.flatten()
    pixels_per_importance=np.argsort(sd_fly_flattened)[::-1] # most important first
    sd_sorted=sd_fly_flattened[pixels_per_importance]
    sd_sorted_cumsum=np.cumsum(sd_sorted)
    plt.plot(np.arange(len(sd_sorted_cumsum)), sd_sorted_cumsum)
    plt.show()
    last_important_pixel=np.abs(sd_sorted_cumsum-sd_sorted_cumsum[-1]*0.9).argmin()
    sd_fly_important=sd_fly_flattened.copy()
    sd_fly_important[pixels_per_importance[last_important_pixel:]] = 0
    sd_fly_important_2d=sd_fly_important.reshape(sd_fly.shape)
    plot_stat(sd_fly_important_2d)
