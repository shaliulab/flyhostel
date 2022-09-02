"""
Utils to perform the Radon transform on image data
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

from radon import compute_radon_transform, compute_rotation_angle, compute_radon_transform_parallel, iradon
from plotting import plot_stat

def example(img):

    sinogram = compute_radon_transform(img)
    img_r = np.uint8(iradon(sinogram) + img.mean())
    return sinogram, img_r

def plot_radon_transform(img):

    sinogram, img_r = example(img)
    sinogram_img = plot_stat(sinogram, plt=False)
    cv2.imshow("sinogram", sinogram_img)
    cv2.imshow("img", img)
    cv2.imshow("recovered", img_r)

    angle = compute_rotation_angle(sinogram)

    center = (img.shape[1] // 2, img.shape[0] // 2)

    M = cv2.getRotationMatrix2D(center, -angle,1)
    dst = cv2.warpAffine(img,M, img.shape[::-1])
    cv2.imshow("rotated", dst)
    cv2.waitKey(0)



if __name__ == "__main__":
    path = "teach_data/1X-2022-05-05_14-53-28-0000161818-0_01_original.png"
    mask_path = "teach_data/1X-2022-05-05_14-53-28-0000161818-0_02_mask.png"
    img = cv2.imread(path)[:,:,0]
    mask = cv2.imread(mask_path)[:,:,0]
    x, y = np.where(mask == 255)
    centroid=(int(x.mean()), int(y.mean()))
    img=img[(centroid[1]-100):centroid[1]+100, (centroid[0]-100):(centroid[0]+100)]
    plot_radon_transform(img)
