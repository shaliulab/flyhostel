"""
Utils to perform the Radon transform on image data
"""

import joblib
import numpy as np
from skimage.transform import radon
from skimage.transform import iradon

from confapp import conf

try:
    import local_settings
    conf += local_settings
except ImportError:
    pass



def compute_radon_transform(img):
    #circle_mask = np.zeros_like(img)
    #circle_mask=cv2.circle(circle_mask, (img.shape[1] // 2, img.shape[0] // 2), img.shape[0]//2, 255, -1)
    #cv2.imshow("mask", circle_mask)
    #img=cv2.bitwise_and(img, circle_mask)
    #cv2.imshow("img", img)
    #cv2.waitKey(0)
    img=np.float64(img)-img.mean()
    sinogram=radon(img)
    return sinogram


def compute_rotation_angle(sinogram):
    r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    rotation = np.argmax(r)
    return rotation #degs

def compute_radon_transform_parallel(imgs):
    """
    Arguments

        imgs (list): List of imgs to have the radon transform applied to
    """
    sinograms = joblib.Parallel(n_jobs=conf.N_JOBS_RADON_TRANSFORM)(
        joblib.delayed(compute_radon_transform)(
            imgs[i].copy()
        ) for i in range(len(imgs))
    )
    return sinograms
