"""
Utils to perform the Radon transform on image data
"""

import joblib
import codetiming
import numpy as np
import cv2
from skimage.transform import radon
from skimage.transform import iradon

from confapp import conf

try:
    import local_settings # type: ignore
    conf += local_settings
except ImportError:
    pass



def compute_radon_transform(img, normalize=False):
    img=np.float64(img)-img.mean()
    sinogram=radon(img)
    if normalize:
        sinogram_img = np.zeros(sinogram.shape, np.uint8)
        sinogram_img = cv2.normalize(sinogram, sinogram_img, 0, 255, cv2.NORM_MINMAX)
        sinogram=sinogram_img
        
    return sinogram


def compute_rotation_angle(sinogram):
    r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    rotation = np.argmax(r)
    return rotation # in degrees


def compute_radon_transform_parallel(imgs, n_jobs=None, normalize=False):
    """
    Arguments

        imgs (list): List of imgs to have the radon transform applied to
    """
    if n_jobs is None:
        n_jobs = conf.N_JOBS_RADON_TRANSFORM

    with codetiming.Timer(text="Done in {:.8f} seconds"):
        sinograms = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(compute_radon_transform)(
                imgs[i].copy(), normalize=normalize
            ) for i in range(len(imgs))
        )
    sinograms=np.stack(sinograms)
    return sinograms
