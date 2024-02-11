import logging

import cv2
import numpy as np

from idtrackerai.animals_detection.segmentation_utils import _getCentroid
from idtrackerai.blob import _overlaps_with_fraction

logger=logging.getLogger(__name__)

def select_by_contour(contour, contours_list, debug=False):

    scores=[]
    for putative_match in contours_list:
        scores.append(
            _overlaps_with_fraction(contour, np.array(putative_match))
        )
    scores=np.array(scores)
    if debug:
        print("Scores: ", scores)
    if scores.sum()==0:
        match_idx=None
    elif (scores==1).sum()>1:
        raise ValueError("The annotated contour fully is fully contained in more than 2 yolo boxes")
    else:
        match_idx=np.argmax(scores)
    return match_idx

def contour_centroid(contour):
    # Calculate Moments
    M = cv2.moments(contour)
    
    # Calculate the centroid if M["m00"] is not zero
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        raise ValueError
        cx, cy = 0, 0  # Set to some default value or handle the error

    return (cx, cy)

def select_by_centroid(contour, centroid_arr):

    centroid=np.array(contour_centroid(contour))

    distance=((centroid_arr - centroid)**2).sum(axis=1)**0.5
    match_idx=np.argmin(distance)
    return match_idx

def get_contour_list_from_yolo_centroids(centroid_arr, size=100):
    half_size=size//2
    contour_list=[]
    for animal in range(centroid_arr.shape[0]):
        x, y=centroid_arr[animal,:]
        contour=np.array([[x-half_size, y-half_size], [x+half_size, y-half_size], [x+half_size, y+half_size], [x-half_size, y+half_size]])
        contour=contour.reshape((4, 1, 2)).astype(np.int32)
        contour_list.append(contour)
    return contour_list


def polygon_to_blob(polygon, frame_width, frame_height, number_of_cols, original_resolution):
    
    contour=np.array(polygon).reshape((-1, 1, 2)).astype(np.int32)
    frame_idx_in_block, contour=contour_to_frame_idx_in_block(contour, frame_width, frame_height, number_of_cols, original_resolution)
    x, y=contour_to_centroid(contour, frame_width, frame_height)
    return frame_idx_in_block, (x, y), contour
    
def rle_to_mask(rle, shape):
    """
    Convert RLE (run length encoding) to a binary mask.
    
    Parameters:
        rle (dict): RLE encoding with 'counts' and 'size' keys
        shape (tuple): The shape (height, width) of the mask
        
    Returns:
        np.array: Binary mask
    """
    counts = rle['counts']
    height, width = shape
    
    mask_flat = np.zeros(width * height, dtype=np.uint8)
    
    current_position = 0
    for i, count in enumerate(counts):
        if i % 2 == 0:
            # Skip background pixels
            current_position += count
        else:
            # Set object pixels to 1
            mask_flat[current_position:current_position+count] = 1
            current_position += count
    
    # Reshape the flat mask to the 2D shape
    return mask_flat.reshape((height, width)).T  # Transpose to correct orientation


def rle_to_blob(*args, frame_width, frame_height, number_of_cols, original_resolution=(1000, 1000), **kwargs):
    mask=rle_to_mask(*args, **kwargs)
    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    assert len(contours)==1
    contour=contours[0]
    frame_idx_in_block, contour=contour_to_frame_idx_in_block(contour, frame_width, frame_height, number_of_cols, original_resolution)
    x, y=contour_to_centroid(contour, frame_width, frame_height)
    return frame_idx_in_block, (x, y), contour


def contour_to_centroid(contour, frame_width, frame_height):    
    x, y= _getCentroid(contour) 
    return round(x, 2), round(y, 2)   
    

def contour_to_frame_idx_in_block(contour, frame_width, frame_height, number_of_cols, original_resolution):
    col=contour[:,:,0]//frame_width
    row=contour[:,:,1]//frame_height
    frame_idx_in_block=(col+row*number_of_cols)[0].item()

    contour[:, :, 0]%=frame_width
    contour[:, :, 1]%=frame_height

    contour[:,:,0]=original_resolution[0]*contour[:,:,0]/frame_width
    contour[:,:,1]=original_resolution[1]*contour[:,:,1]/frame_height
    
    contour=contour.astype(np.int32)
    
    return frame_idx_in_block, contour
