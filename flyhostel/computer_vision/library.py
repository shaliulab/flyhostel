import os.path
import warnings
import logging
import pickle
from matplotlib.font_manager import list_fonts
import tqdm
import glob
from collections import OrderedDict

import joblib
import threading
import yaml
import numpy as np
import cv2
import idtrackerai.list_of_blobs
from imgstore.interface import VideoCapture
from imgstore.constants import STORE_MD_FILENAME
import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from codetiming import Timer # https://github.com/realpython/codetiming

from flyhostel.computer_vision.contour import center_blob_in_mask, find_contour

from flyhostel.computer_vision.plotting import plot_rotation
from flyhostel.computer_vision.get_files import (
    get_video_object,
    get_trajectories_file,
    get_timestamps_file,
    get_collections_file,
    get_store_path,
    get_experiments,
)
from flyhostel.computer_vision.utils import package_frame_for_labeling, obtain_real_frame_number, adjust_to_zt0, hours
from flyhostel.computer_vision.timepoints import TimePoints

from confapp import conf

try:
    import local_settings  # type: ignore
    conf += local_settings
except ImportError:
    pass


logger = logging.getLogger(__name__)
timer_logger = logging.getLogger(f"{__name__}.timer")

def crop_animal_in_time_and_space(frame, centroid, body_size, filepath, contour, other_contours, thresholds=None):
    """
    Crop a box in the frame around the focal fly and rotate it so the animal points east 
    
    Arguments:
    
        * frame (np.ndarray): a copy of a raw frame from a flyhostel dataset
        * centroid (tuple): x and y coordinates of the center of the focal animal, with the origin (0, 0) set to the top left corner
        * body_size (int): Expected number of pixels on the longest axes of the animal (each animal should have a value around this, but not exactly this)
        * filepath (str): Path to a .png file
        * contour (np.ndarray): outline of the contour of one of the animals (focal) in the frame, in raw coordinates
        * other_contours (list): list of outline of the contours of other animals which may appear on the frame
    
    Returns:
        * rotated (np.ndarray): crop of the raw frame with the focal animal pointing east 
    """

    if contour is None:
        warnings.warn(f"No contour detected in {filepath}", stacklevel=2)
        return None, None, None
   
    # mask the other contours
    with Timer(text="Masking other contours took {:.8f}", logger=timer_logger.debug):
        easy_frame=cv2.drawContours(frame.copy(), other_contours, -1, 255, -1)

    if conf.CENTRAL_BOX_SIZE is None:
        CENTRAL_BOX_SIZE=body_size*3
    else:
        CENTRAL_BOX_SIZE=conf.CENTRAL_BOX_SIZE

    with Timer(text="Expanding frame took {:.8f}", logger=timer_logger.debug):
        # expand the frame with white so the box is never out of bounds
        easy_frame=cv2.copyMakeBorder(
            easy_frame,
            CENTRAL_BOX_SIZE,
            CENTRAL_BOX_SIZE,
            CENTRAL_BOX_SIZE,
            CENTRAL_BOX_SIZE,
            cv2.BORDER_CONSTANT,
            255   
        )
        frame=cv2.copyMakeBorder(
            frame,
            CENTRAL_BOX_SIZE,
            CENTRAL_BOX_SIZE,
            CENTRAL_BOX_SIZE,
            CENTRAL_BOX_SIZE,
            cv2.BORDER_CONSTANT,
            255   
        )
        
        centroid=tuple([centroid[0]+CENTRAL_BOX_SIZE, centroid[1]+CENTRAL_BOX_SIZE])
        contour+=CENTRAL_BOX_SIZE

    with Timer(text="Packaging frame took {:.8f}", logger=timer_logger.debug):
        easy_crop, bbox = package_frame_for_labeling(
            easy_frame, centroid, CENTRAL_BOX_SIZE
        )
        raw_crop, bbox = package_frame_for_labeling(
            frame, centroid, CENTRAL_BOX_SIZE
        )

        top_left_corner = bbox[:2]
        centered_contour=contour-top_left_corner

    with Timer(text="Computing angle took {:.8f}", logger=timer_logger.debug):
        angle, (T, mask, cloud, cloud_centered, cloud_center)=find_angle(easy_crop, centered_contour, body_size=body_size)
    with Timer(text="Rotating frames took {:.8f}", logger=timer_logger.debug):
        rotated, rotate_matrix = rotate_frame(raw_crop, angle)
        easy_rotated, _ = rotate_frame(easy_crop, angle)

    with Timer(text="Packaging frame (II) took {:.8f}", logger=timer_logger.debug):
        mmpy_frame, _ = package_frame_for_labeling(easy_rotated, center=([e//2 for e in rotated.shape[:2][::-1]]), box_size=conf.MMPY_BOX)
        sleap_frame, _ = package_frame_for_labeling(rotated, center=([e//2 for e in rotated.shape[:2][::-1]]), box_size=conf.SLEAP_BOX)
    
    frames = {"sleap": sleap_frame, "mmpy": mmpy_frame}

    if conf.DEBUG:
        plot_rotation(raw_crop, mask, T, cloud_centered, filepath)

    # save
    if conf.COMPUTE_THRESHOLDS:
        path=os.path.join(conf.HISTOGRAM_FOLDER, filepath)
        logger.debug(f'Saving -> {path}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, frames["mmpy"])

    else:
        for folder in conf.DATASET_FOLDER:
            path=os.path.join(conf.DATASET_FOLDER[folder], "step1", filepath.replace(".png", "_05_final.png"))
            with Timer(text=f"Saving {path} took " + "{:.8f}", logger=timer_logger.debug):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                logger.debug(f'Saving -> {path}')
                cv2.imwrite(path, frames[folder])

    return raw_crop, rotated, (T, mask, cloud, cloud_centered, cloud_center, rotate_matrix, top_left_corner)


def find_angle(crop, contour, body_size, point_to=90):
    """
    Take a crop with a centered animal and some surrounding space
    and rotate so the animal points east
    
    Arguments:  
    * crop (np.ndarray): Grayscale image of the animal
    * contour (np.ndarray): Contour of the animal in the image
    * point_to (int): Direction to which the animal should point to.
      0 means east, 90 means north, -90 means south and 180 means west

    Returns:
    * rotated (np.ndarray): Grayscale image rotated
    
    Details:
    
    * a contour in opencv has always shape nx1x2
      where n is the number of points
      and we have two dimensions for x and y in the 3rd axis
    """
    
    mask = np.zeros_like(crop)
    
    # contour should have shape ?x1x2
    assert contour.shape[1] == 1
    assert contour.shape[2] == 2
    
    mask=cv2.drawContours(mask, [contour], -1, 255, -1)
    cloud=np.stack(
        np.where(mask == 255)[::-1],
        axis=1
    )
    # np.where returns the coordinates along the first axis,
    # then the second, third, and so on i.e. first rows and then columns
    # this means the first column of cloud contains the coordinates along the
    # first axis of mask (rows), i.e. the height (y)
    # this means we need to flip the order of the columns
    # so the first column represents the x coordinates
    # and the second the y coordinates
    
    # now cloud's first column represents the x (horizontal axis)
    # and the second 
    
    # also, the origin now is set to the top left corner
    # (because the first row is the row on top)
    # and we want it in the bottom left, so we need to flip the y axis
    cloud[:, 1] = mask.shape[0] - cloud[:, 1]

    cloud_center = center_blob_in_mask(mask)
    
    logger.debug(f"Cloud center {cloud_center}")

    # center the contour around its mean
    cloud_centered=cloud-cloud_center


    # compute the covariance matrix
    cov_matrix = np.cov(cloud_centered[
        np.random.choice(np.arange(cloud_centered.shape[0]), size=100, replace=False),
        :
    ].T)
    
    # Eigendecomposition
    ################################
    # compute the eigenvectors of the covariance matrix
    vals, T = np.linalg.eig(cov_matrix)
    
    T=T[:, np.argsort(vals)[::-1]]
    vals=vals[np.argsort(vals)[::-1]]
    
    
    # get the first eigen vector
    v1 = T[:, 0]
    v2 = T[:, 1]

    logger.debug(f"Eigenvalues: {vals}")
    logger.debug(f"First eigenvector {v1}")
    logger.debug(f"Second eigenvector {v2}")

    # Singular Value Decomposition
    ################################
    # u, s, T = np.linalg.svd(cloud_centered)
    # v1 = T[:,0]
   
    # compute the angle of the first eigenvector in radians
    
    angle = np.arctan2(v1[1], v1[0])
    # transform to degrees
    angle_deg=angle*360/(np.pi*2)-point_to
    
    logger.debug(f"Angle {angle_deg} degrees")


    rotated, rotate_matrix = rotate_frame(crop, angle_deg)
    flip=find_polarity(crop, mask, rotated, rotate_matrix, body_size, filepath=None)
    if flip:
        angle_deg-=180

    return angle_deg, (T, mask, cloud, cloud_centered, cloud_center)

def rotate_frame(img, angle):
    """
    Rotate the img the given angle
    
    Arguments:
    
        * img (np.ndarray)
        * angle (float): degrees
    Returns:
    
        * rotated (np.ndarray): img after applying the rotation
        * rotate_matrix (np.ndarray): rotation matrix that was used to perform the rotation
    """

    # compute the rotation matrix needed to cancel the angle
    rotate_matrix = cv2.getRotationMatrix2D(center=tuple([e//2 for e in img.shape]), angle=-angle, scale=1)
    # apply the rotation
    rotated = cv2.warpAffine(src=img, M=rotate_matrix, dsize=img.shape[:2][::-1])
    return rotated, rotate_matrix


def find_polarity(crop, mask, rotated, rotate_matrix, body_size, filepath):
    rotated_mask = cv2.warpAffine(src=mask, M=rotate_matrix, dsize=crop.shape[:2][::-1])
    boolean_mask = rotated_mask == 255

    assert boolean_mask.sum() > 0
    
    coord_max=np.where(boolean_mask)[1].max()
    coord_min=np.where(boolean_mask)[1].min()
    width = np.where(boolean_mask)[0].max() - np.where(boolean_mask)[0].min()
    height = coord_max - coord_min
    
    top_mask = np.zeros_like(rotated)
    bottom_mask = np.zeros_like(rotated)
    mask_center = tuple([e // 2 for e in rotated.shape[:2][::-1]])
    
    # radius=round(height*0.6)
    # size = int(rotated.shape[0]*0.65 / 2)
    radius=int(0.5 * body_size)
    vertical_offset=int(0.8*body_size)
    
    
    cv2.circle(top_mask, (mask_center[0], mask_center[1] - vertical_offset), radius=radius, color=255, thickness=-1)
    cv2.circle(bottom_mask, (mask_center[0], mask_center[1] + vertical_offset), radius=radius, color=255, thickness=-1)
    
    top_area=cv2.bitwise_and(rotated.copy(), top_mask)
    bottom_area=cv2.bitwise_and(rotated.copy(), bottom_mask)
    
    logger.debug(f"Intensity in top circle: {top_area.mean()}")
    logger.debug(f"Intensity in bottom circle: {bottom_area.mean()}")
    
    if top_area.mean() < bottom_area.mean():
        flip=True
    else:
        flip=False
    
    if filepath is not None and conf.DEBUG:
        cv2.imwrite(os.path.join(conf.DEBUG_FOLDER, filepath.replace(".png", "_top-area_5.png")), top_area)
        cv2.imwrite(os.path.join(conf.DEBUG_FOLDER, filepath.replace(".png", "_bottom-area_5.png")), bottom_area)    
    return flip


def plot_intensity_histogram(experiment):


    files = glob.glob(
        os.path.join(
            conf.HISTOGRAM_FOLDER, experiment, "*.png"
        )
    )

    assert len(files) != 0

    arrs=[]
    for f in tqdm.tqdm(files, desc="Loading data to generate histogram"):
        arrs.append(cv2.imread(f))
    
    data=np.float64(np.stack(arrs, axis=-1).flatten())

    data-=data.min()
    data/=data.max()
    data*=255
    y,x=np.histogram(data, bins=255)
    WINDOW_LENGTH=5
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    # smoothen potential outliers by taking a rolling mean
    y_round=moving_average(y, WINDOW_LENGTH)
    # discard the last bin because it's biased by the other contour masking
    # (which sets the pixels there to 255)
    y_round=y_round[:-1]

    # make a probability-like distrib
    y_round /= y_round.sum()
    x_round=x[:-(WINDOW_LENGTH+1)]

    fig=plt.figure(figsize=(20, 10))
    ax=fig.add_subplot()
    ax.bar(x_round, y_round)
    ax.set_xticks(np.arange(0, 255, 5), np.arange(0, 255, 5))
    fn=os.path.join(conf.HISTOGRAM_FOLDER, os.path.dirname(experiment), os.path.basename(experiment)+".jpg")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn)

def load_thresholds(experiment):

    file = os.path.join(conf.VIDEO_FOLDER, experiment, "metadata.yaml") 

    with open(file, "r") as filehandle:
        metadata=yaml.load(filehandle, yaml.SafeLoader)

    assert "thresholds" in metadata
    return metadata["thresholds"]


def compute_parts_masks(file, body, wing, min_body_area, iterations=10):

    frame=cv2.imread(file)
    img=frame.copy()
    if img.shape[2] == 3:
        img = img[:,:,0]

    body_mask=np.uint8(255 * (img < body))
    _, contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area=0
    if len(contours) > 0:
        contours = [sorted(contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)[0]]
        contour_area = cv2.contourArea(contours[0]) 


    while len(contours) == 0 or contour_area < min_body_area:
        if conf.DEBUG:
            print(f"{contour_area} < {min_body_area}")
        if len(contours) > 0:
            contours = [sorted(contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)[0]]
            contour_area = cv2.contourArea(contours[0]) 
            if conf.DEBUG_FIND_WING:
                mask = np.zeros_like(img)
                mask=cv2.drawContours(mask, contours, -1, 255, -1)
                cv2.imshow("body_mask", mask)
                cv2.waitKey(0)
        
        if contour_area >= min_body_area:
            break

        body += 1
        body_mask=np.uint8(255 * (img < body))
        _, contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    wing_mask = np.uint8(
        255 * np.bitwise_and(
            img >= body,
            img < wing
        )
    )

    if conf.DEBUG_FIND_WING:
        cv2.imshow("final_wing_mask", wing_mask)
        cv2.imshow("final_body_mask", body_mask)
        cv2.waitKey(0)
        


    all_bodies = cv2.bitwise_or(
        body_mask,
        np.uint8(255 * (img == 255))
    )

    # remove contour around body which may be assigned to wings
    all_bodies=cv2.dilate(all_bodies, kernel=np.ones((5, 5)), iterations=iterations)
    wing_mask[all_bodies==255]=0

    return img, {"body": body_mask, "wing": wing_mask, "all_bodies": all_bodies}


def correct_rotation(file, min_body_area, min_wing_area, body, wing, iterations=10):
    dest=os.path.sep.join(file.split(os.path.sep)[-3:])
    fn, ext=os.path.splitext(dest)
   
    frame, masks=compute_parts_masks(file, min_body_area=min_body_area, body=body, wing=wing, iterations=iterations)

    def find_wing_contour(masks):
        # subset central third of the image vertically
        wing_mask=masks["wing"]
        wing_mask=wing_mask[:, (wing_mask.shape[1] // 3):(2*wing_mask.shape[1] // 3)]
        _, contours, _ = cv2.findContours(wing_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contours=sorted(contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)
            if cv2.contourArea(contours[0]) < min_wing_area:
                contours=[]

        return contours
    
    if conf.DEBUG_FIND_WING:
        cv2.imshow(f"wing mask after {iterations} iterations", masks["wing"])
        cv2.imshow(f"img", frame)
        cv2.waitKey(0)

    contours=find_wing_contour(masks)

    while len(contours) == 0 and iterations > 2:
        iterations-=1
        frame, masks=compute_parts_masks(file, min_body_area=min_body_area, body=body, wing=wing, iterations=iterations)
        if conf.DEBUG_FIND_WING:
            cv2.imshow(f"wing mask after {iterations} iterations", masks["wing"])
            cv2.imshow(f"img", frame)
            cv2.waitKey(0)
        contours=find_wing_contour(masks)

    cv2.destroyAllWindows()
    
    if len(contours) == 0:
        warnings.warn(f"Cannot detect wings on {file}")
        return (255*np.ones_like(frame), False), np.zeros_like(frame)

    else:
        wing_mask=masks["wing"]
        
    for mask in masks:
        mask_fn=os.path.join(
            conf.MASKS_FOLDER, f"{fn}_{mask}{ext}"
        )
        os.makedirs(os.path.dirname(mask_fn), exist_ok=True)
        cv2.imwrite(mask_fn, masks[mask])


    mask = np.zeros_like(wing_mask)
    mask=cv2.drawContours(mask, contours, -1, 255, -1)
    
    y,x=center_of_mass(mask)

    if y < mask.shape[0] / 2:
        # wings are above
        frame=frame[::-1,:]
        flip=True
    else:
        # wings are below
        flip=False


    # hide_other_flies(file, frame, wing_mask)
    frame=hide_other_flies_v2(frame, wing)    
    return (frame, flip), mask

def hide_other_flies(file, frame, wing_mask):
   # TODO
    # To mask the wings of other animals
    # Not working well at the moment
    d=load_pickle_file(file)
    corner=d["corner"]
    contour=d["contour"]

    cv2.imshow("wing_mask", wing_mask)
    cv2.waitKey(0)
    body_mask=np.zeros_like(wing_mask)
    body_mask=cv2.drawContours(body_mask, contour, -1, 255, -1)
    cv2.imshow("body_mask", body_mask)
    cv2.waitKey(0)
    fly_mask=cv2.bitwise_or(wing_mask, body_mask)
    frame[fly_mask!=255]=255
    return frame

def hide_other_flies_v2(frame, wing):
    img=frame.copy()
    blur=cv2.GaussianBlur(img, (5, 5), 1, 1, cv2.BORDER_REPLICATE)

    _, flies_mask=cv2.threshold(blur, thresh=wing, maxval=255, type=cv2.THRESH_BINARY_INV)
    # cv2.imshow("mask_dirty", flies_mask)
    _, contours, _  = cv2.findContours(flies_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # TODO
    # Change the hardcoded min_fly_area of 2000 to something which is a function of the animal size 
    # contours=[cnt for cnt in contours if cv2.contourArea(cnt) > 2000]
    contours=sorted(contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)
    if len(contours) > 1:
        contours=[contours[0]]

    flies_mask_clean=np.zeros_like(flies_mask)
    flies_mask_clean=cv2.drawContours(flies_mask_clean, contours, -1, 255, -1)
    img[flies_mask_clean==0]=255

    if conf.DEBUG_HIDE_BACKGROUND:
        cv2.imshow("fly without neighbors", img)
        cv2.waitKey(0)
    return img

def load_pickle_file(mmpy_file_05_final):
    pickle_file = os.path.join(
        conf.CONTOURS_FOLDER, mmpy_file_05_final.replace(os.path.join(conf.DATASET_FOLDER["mmpy"], "step1"), "").lstrip("/")
    ).replace("_05_final.png", ".pkl")

    with open(pickle_file,"rb") as filehandle:
        d=pickle.load(filehandle)
    
    return d



def correct_rotation_all(experiment, index):

    thresholds=load_thresholds(experiment)
    files = glob.glob(
        os.path.join(
            conf.DATASET_FOLDER["mmpy"], "step1", experiment, "*"
        )
    )

    body_size=get_video_object(experiment, 1).median_body_length
    min_wing_area=(body_size/7)**2
    min_body_area=(body_size/2)**2

    for file in files:
        (frame, flip), wing_mask=correct_rotation(file, min_body_area, min_wing_area, **thresholds)
        fn = os.path.join(
            conf.DATASET_FOLDER["mmpy"], experiment, os.path.basename(file) + "_06_corrected.png"
        )

        os.makedirs(os.path.dirname(fn), exist_ok=True)
        cv2.imwrite(fn, frame)

        for folder in conf.DATASET_FOLDER:
            if folder == "mmpy":
                continue
            else:
                other_file = file.replace(conf.DATASET_FOLDER["mmpy"], conf.DATASET_FOLDER[folder])
                img=cv2.imread(other_file)
                if flip:
                    img=img[::-1,:]

                fn = os.path.join(
                    conf.DATASET_FOLDER[folder], experiment, os.path.basename(other_file) + "_06_corrected.png"
                )
                os.makedirs(os.path.dirname(fn), exist_ok=True)
                cv2.imwrite(fn, img)


        # NOTE
        # To make a movie with all the frames in a folder, assuming frames are 200x200
        # cat *jpg | ffmpeg -y -r 40  -vsync 0 -s 200x200   -i - -an -c:v h264_nvenc $(basename `pwd`).mp4

def crop_animals(experiment, trajectories, sampling_points, tolerance, thresholds=None):

    assert len(sampling_points) > 0

    store = VideoCapture(os.path.join(get_store_path(experiment), STORE_MD_FILENAME), 1)

    if sampling_points._type == "interval":
        index=np.array(store._index.get_all_metadata()["frame_time"])
        sampling_points.set(index)

    sampling_points_per_chunk={}
    starts_per_chunk={}
    last_chunk=store._index.find_all("frame_number", store._index._summary("frame_max"))[0]
    for chunk in range(last_chunk):
        starts_per_chunk[chunk]=store._get_chunk_metadata(chunk)["frame_time"][0]

    sampling_points_per_chunk_=OrderedDict()

    
    for chunk in starts_per_chunk:
        if (chunk+1) in starts_per_chunk:
            sampling_points_per_chunk_[chunk]=sampling_points[
                np.bitwise_and(
                    sampling_points > starts_per_chunk[chunk],
                    sampling_points < starts_per_chunk[chunk+1]
                )
            ]
        else:
            sampling_points_per_chunk_[chunk]=sampling_points[sampling_points > starts_per_chunk[chunk]]

   
    sampling_points_per_chunk=OrderedDict()
    for chunk in sampling_points_per_chunk_:
        if len(sampling_points_per_chunk_[chunk]) > 0:
            sampling_points_per_chunk[chunk] = sampling_points_per_chunk_[chunk]

    for chunk in sampling_points_per_chunk:
        print(f"{chunk}: {len(sampling_points_per_chunk[chunk])}")


    if conf.MULTIPROCRESSING:
        joblib.Parallel(n_jobs=conf.N_JOBS_CHUNKS)(
            joblib.delayed(process_chunk)(
                chunk, experiment, trajectories, tolerance, sampling_points_per_chunk[chunk], thresholds
            ) for chunk in sampling_points_per_chunk
        )
    
    # TODO Either reimplement or remove
    # elif conf.MULTITHREADING:

    #     threads=[
    #         threading.Thread(
    #             target=process_timepoint,
    #             kwargs={
    #                 "experiment": experiment, "trajectories": trajectories,
    #                 "tolerance": tolerance, "msec": msec, "thresholds": thresholds
    #             },
    #             daemon=True
    #             )
    #         for msec in sampling_points
    #     ]
    #     for t in threads:
    #         t.start()
        
    #     for t in threads:
    #         t.join()
    
    else:
        # pbar=tqdm.tqdm(sampling_points_per_chunk, desc="Processing chunk")
        # for chunk in pbar:
        for chunk in sampling_points_per_chunk:
            # pbar.set_description(f"Processing chunk {chunk}")
            process_chunk(chunk, experiment, trajectories, tolerance, sampling_points_per_chunk[chunk], thresholds)


def process_chunk(chunk, experiment, trajectories, tolerance, chunk_points, thresholds=None):
    store = VideoCapture(os.path.join(get_store_path(experiment), STORE_MD_FILENAME), 1)
    img, (fn, ft) = store.get_chunk(chunk)
    img, (fn, ft) = store.get_nearest_image(chunk_points[0]-1, past=True, future=False)

    if not chunk_points[0] >= ft:
        import ipdb; ipdb.set_trace()
    
    # assert chunk_points[0] >= ft, f"{chunk_points[0]} is already passed"


    list_of_blobs = idtrackerai.list_of_blobs.ListOfBlobs.load(
        get_collections_file(experiment, chunk)
    )
    
    video_object = get_video_object(experiment, chunk)
    body_size=round(video_object.median_body_length)
    frame_index = store._chunk_current_frame_idx-1

    for msec in tqdm.tqdm(chunk_points, desc=f"Working on chunk {chunk}"):

        #if frame_index > len(metadata["frame_number"]):
        #    break

        if conf.COMPUTE_THRESHOLDS:
            chunk_n, frame_index = store._index.find_chunk_nearest("frame_time", msec, future=False, past=True)
            assert chunk == chunk_n
        else:
            frame_index+=1
        

        process_timepoint(
            experiment=experiment, store=store, trajectories=trajectories,
            tolerance=tolerance,  msec=msec, thresholds=thresholds,
            body_size=body_size, list_of_blobs=list_of_blobs, frame_index=frame_index
        )



def process_timepoint(experiment, store, trajectories, body_size, list_of_blobs, tolerance, msec, frame_index, thresholds=None):

    # load .mp4 dataset
    with Timer(text="Getting next frame took {:.8f}", logger=timer_logger.debug):            
        if conf.COMPUTE_THRESHOLDS:
            frame, (frame_number, frame_time) = store.get_nearest_image(msec)
        else:
            frame, (frame_number, frame_time) = store.get_next_image()


    if store._index._summary("frame_max") == frame_number:
        warnings.warn("You are reading the last frame of the imgstore", stacklevel=2)

    logger.debug(f"Working on {frame_number} ({frame_time} ms)")
    chunk = store._chunk_n

    error = np.abs(msec - frame_time)  
    assert error < tolerance, f"{error} ms is higher than the tolerance value of {tolerance} ms"

    # with Timer(text="Computing real frame number took {:.8f}", logger=timer_logger.debug):            
    #     real_fn = store.obtain_real_frame_number(frame_number)
    
    real_fn = frame_number
  
    try:
        blobs_in_frame=list_of_blobs.blobs_in_video[real_fn]
    except Exception as error:
        warnings.warn(f"Chunk {chunk} does not have data for frame_index {frame_index}", stacklevel=2)
        raise error

    # cropped_contours=[]
    # corners=[]
    for animal in np.arange(trajectories.shape[1]):
    
        # define where the files will be saved
        filepath = os.path.join(
            experiment,
            os.path.join(experiment, str(frame_number).zfill(10), str(animal) + ".png").replace("/", "-")
        ) 
        # pick the data for this animal in this frame number
        with Timer(text="Picking animal took {:.8f}", logger=timer_logger.debug):            
            tr = pick_animal(trajectories, real_fn, animal)
        if type(tr) is IndexError:
            logger.debug(error)
            warnings.warn(f"{experiment} is not analyzed after {round(msec/1000/3600, 2)} hours", stacklevel=2)
            break

        try:
            centroid = tuple([round(e) for e in tr])
        except ValueError:
            # contains NaN
            warnings.warn(f"Animal #{animal} not found in frame {frame_number} (chunk {chunk}) of {experiment}", stacklevel=2)
            continue

        with Timer(text="Getting contours took {:.8f}", logger=timer_logger.debug):            
            contour, other_contours = find_contour(blobs_in_frame, centroid, frame=frame)
        # TODO Properly unpack the output of crop_animal_in_time_and_space

        with Timer(text="Cropping animal in time and space took {:.8f}", logger=timer_logger.debug):
            _,_, data2=crop_animal_in_time_and_space(frame.copy(), centroid=centroid, body_size=body_size, filepath=filepath, contour=contour, other_contours=other_contours, thresholds=thresholds)

        if data2 is None:
            return

        # corners.append(data2[-1])
        # cropped_contours.append(contour-corners[-1])
        corner = data2[-1]
        cropped_contour = contour-corner
        
        with Timer(text="Saving to pickle file took {:.8f}", logger=timer_logger.debug):
            fn, _=os.path.splitext(os.path.basename(filepath))
            pickle_file=os.path.join(conf.CONTOURS_FOLDER, experiment, f"{fn}.pkl")
            os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
            with open(pickle_file, "wb") as filehandle:
                pickle.dump({"corner": corner, "contour": cropped_contour}, filehandle)


def generate_dataset(experiment, tolerance=100, compute_thresholds=True, crop=True, rotate=True):
    """
    Generate a dataset of frames for pose labeling
    from a flyhostel experiment
    
    Arguments:
    
    * experiment (str): Path to folder containing an imgstore dataset (set of .mp4, .npz and .extra.json) files
    and a COMPUTER_VISION_FOLDER folder with two .npy files that summarise the idtrackerai output. These files should be called
    {experiment_datetime}_trajectories.npy and {experiment_datetime}_timestamps.npy
    
    * sampling_points (list): List of integers which define the time at which a frame is sampled in the experiment,
    in ms since the start of the experiment
    
    * tolerance (int): Tolerance between wished sampling time and available time in dataset, in msec
    """

    metadata = get_experiments()
    metadata = metadata.loc[metadata["experiment"] == experiment].squeeze()

    zt0 = metadata["zt0"]
    sampling_points_hour = [float(e) for e in metadata["sampling_points"].split("-")]
    sampling_points = tuple([1000*hours(e) for e in sampling_points_hour])
    sampling_points=TimePoints(sampling_points)
    logger.info(experiment)
    
    tr_file = get_trajectories_file(experiment)
    time_file = get_timestamps_file(experiment)
    
    assert os.path.exists(tr_file), f"{tr_file} does not exist"
    assert os.path.exists(time_file), f"{time_file} does not exist"

    # load trajectories and timestamps
    trajectories = np.load(tr_file)
    timestamps = np.load(time_file)
    timestamps = adjust_to_zt0(experiment, timestamps, zt0, reference="start_time")
    
    assert len(timestamps) == trajectories.shape[0]

    if compute_thresholds:
        sample=sampling_points.sample(conf.HISTOGRAM_SAMPLE_SIZE)
        crop_animals(experiment, trajectories, sample, tolerance=tolerance)
        plot_intensity_histogram(experiment)

    if crop:
        try:
            thresholds=load_thresholds(experiment)
        except Exception:
            thresholds=None
        index=crop_animals(experiment, trajectories, sampling_points, thresholds=thresholds, tolerance=tolerance)
    else:
        index=None
    if rotate:
        correct_rotation_all(experiment, index)



def pick_animal(trajectories, frame_number, animal):
        try:
            return trajectories[frame_number, animal, :]
        except IndexError as error:
            return error
        
