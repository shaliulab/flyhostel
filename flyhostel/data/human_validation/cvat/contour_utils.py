import logging

import cv2
import numpy as np

from idtrackerai.animals_detection.segmentation_utils import _getCentroid
from idtrackerai.blob import _overlaps_with_fraction

logger=logging.getLogger(__name__)

def select_by_contour(contour, contours_list, debug=False):
    """
    Select one of several machine-made squares around each centroid (contours_list) based on a human-made contour annotation (contour)
    
    The selected square will be that which maximizes the overlap with the human-made contour
    If 1) >1 square is fully contained in the annotation 2) none overlaps at all, or 3) there is a tie, a ValueError is raised

    Note:

    if the annotation does not overlap with any square, it is a 'de novo' annotation
    if the annotation fully contains 2 or more squares, the two squares are probably a failed NMS (Non Maximal Supression) artifact
    """

    scores=[]
    for putative_match in contours_list:
        scores.append(
            _overlaps_with_fraction(contour, np.array(putative_match))
        )
    scores=np.array(scores)
    
    ties=np.diff(scores[scores>0])==0
    tied=any(ties)
    if tied:
        tied = tied and scores[scores>0][np.where(ties)[0]]==max(scores)

    de_novo=scores.sum()==0
    n_winners=(scores==1).sum()

    winner_tied=n_winners>1
    if debug:
        import ipdb; ipdb.set_trace()
        print("Scores: ", scores)
    if de_novo:
        # de novo
        match_idx=None
        n=0
    elif winner_tied:
        logger.error(f"The annotated contour is fully contained in {n_winners} yolo boxes")
        match_idx=None
        n=-1
    elif tied:
        logger.error(f"The annotated contour equally overlaps with {(np.diff(scores[scores>0])==0).sum()+1} yolo boxes")
        match_idx=None
        n=-1
    else:
        match_idx=np.argmax(scores)
        n=1
    return match_idx, n

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
    frame_idx_in_block=contour_to_frame_idx_in_block(contour, frame_width, frame_height, number_of_cols)
    contour=reproject_contour(contour, frame_width, frame_height, original_resolution)

    x, y=contour_to_centroid(contour)
    # x=int(x / frame_width * original_resolution[0])
    # y=int(y / frame_height * original_resolution[1])

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


def rle_to_blob(*args, frame_width, frame_height, number_of_cols, original_resolution, **kwargs):
    """
    Return frame in block, centroid and contour
    """
    mask=rle_to_mask(*args, **kwargs)
    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    assert len(contours)==1
    contour=contours[0]
    frame_idx_in_block=contour_to_frame_idx_in_block(contour, frame_width, frame_height, number_of_cols)
    contour=reproject_contour(contour, frame_width, frame_height, original_resolution)
    x, y=contour_to_centroid(contour)
    # x=int(x / frame_width * original_resolution[0])
    # y=int(y / frame_height * original_resolution[1])
    
    return frame_idx_in_block, (x, y), contour


def contour_to_centroid(contour):
    x, y= _getCentroid(contour)
    return round(x, 2), round(y, 2)
    

def contour_to_frame_idx_in_block(contour, frame_width, frame_height, number_of_cols):
    """
    Return the frame count relative to the first frame of the block
    A block is a bunch of space images in a grid. 1 grid makes 1 blocks, and multiple blocks may make the image

    """
    col=contour[:,:,0]//frame_width
    row=contour[:,:,1]//frame_height
    frame_idx_in_block=(col+row*number_of_cols)[0].item()
    return frame_idx_in_block

def reproject_contour(contour, frame_width, frame_height, original_resolution):

    contour[:, :, 0]%=frame_width
    contour[:, :, 1]%=frame_height

    # projcting back in to the original resolution
    contour[:,:,0]=original_resolution[0]*contour[:,:,0]/frame_width
    contour[:,:,1]=original_resolution[1]*contour[:,:,1]/frame_height
    
    contour=contour.astype(np.int32)
    
    return contour
