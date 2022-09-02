import os.path
import logging
import warnings
import matplotlib
import matplotlib.pyplot as plt
# plt.set_cmap("gray")
try:
    matplotlib.use('TkAgg')
except Exception:
    warnings.warn("matplotlib cannot use TkAgg backend", stacklevel=2)

from confapp import conf
import cv2
import numpy as np

try:
    import local_settings # type: ignore
    conf += local_settings
except ImportError:
    pass


logger = logging.getLogger(__name__)

def plot_cloud(cloud):
    """
    Given a cloud of points, plot it in matplotlib
    
    Arguments:
    
    * cloud (np.ndarray): nx2 array of points where the columns represent the x and y coordinates
    """
    
    ratio=1
    fig=plt.figure(1)
    ax=fig.subplots(1)

    ax.scatter(
        cloud[:,0],
        cloud[:,1]
    )
    # x_left, x_right = ax.get_xlim()
    # y_low, y_high = ax.get_ylim()
#     xlim=max(abs(x_left), x_right)
#     ylim=max(abs(y_low), y_high)
#     ax.set_xlim([-xlim, xlim])
#     ax.set_ylim([-ylim, ylim])

    # x_left, x_right = ax.get_xlim()
    # y_low, y_high = ax.get_ylim()
    # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    return fig

def plot_decomposition(crop, T):
    center = [e//2 for e in crop.shape[::-1]]
    fig=plt.figure(2)
    ax=fig.subplots(1)
    
    ax.imshow(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
    v1 = T[:, 0]
    v2 = T[:, 1]
    ax.quiver(*center, v1[0], -v1[1], clim=[-np.pi, np.pi], angles='xy', scale_units='xy', scale=1/100)
    ax.quiver(*center, v2[0], -v2[1], clim=[-np.pi, np.pi], angles='xy', scale_units='xy', scale=1/50)
    return fig

def plot_rotation(crop, mask, T, cloud, filepath):
    """
    Visualize rotation steps
    
    Arguments:
    
    * crop (np.ndarray): Crop of the raw frame
    * mask (np.ndarray): Same shape as crop, set to 0 everywhere
       except where the blob of the animal is
    * cloud (np.ndarray): nx2 array of coordinates of a centered blob
    where the first column represents the x coordinates and the second y
    * T (np.ndarray): Matrix of eigenvectors, where each vector is a column
    * filepath (str): Path to save the visualizations
    """
    path=os.path.join(conf.DEBUG_FOLDER, filepath.replace(".png", "_01_original.png"))

    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)

    logger.debug(f'Saving -> {path}')
    cv2.imwrite(path, crop)

    path=os.path.join(conf.DEBUG_FOLDER, filepath.replace(".png", "_02_mask.png"))
    logger.debug(f'Saving -> {path}')
    cv2.imwrite(path, mask)
    
    fig = plot_decomposition(crop, T)
    path=os.path.join(conf.DEBUG_FOLDER, filepath.replace(".png", "_03_v1.png"))
    logger.debug(f'Saving -> {path}')
    fig.savefig(path)
    plt.close()


    fig = plot_cloud(cloud)
    path=os.path.join(conf.DEBUG_FOLDER, filepath.replace(".png", "_04_cloud.png"))
    logger.debug(f'Saving -> {path}')
    fig.savefig(path)
    plt.close()
    

def plot_stat(arr, plot=True):
    arr = arr.copy()
    arr -= arr.min()
    img = np.uint8(255*arr/arr.max())
    if plot: plt.imshow(img)
    return img 
