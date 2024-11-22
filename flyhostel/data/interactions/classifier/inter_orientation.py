import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .utils import select_loader

def calculate_angles_with_vertical_batch(points):
    """
    Calculate the angles formed by two points A and B with the vertical line starting from B.
    An angle of 0° means AB is vertical, 90° means AB is horizontal pointing East, and -90° means AB is horizontal pointing West.

    Parameters:
    points (numpy.ndarray): Array of shape (N, 2, 2), where:
        - N is the number of time points
        - The second axis contains points A and B
        - The third axis contains the (x, y) coordinates of the points

    Returns:
    numpy.ndarray: Array of shape (N,) containing the angles in degrees for each time point.
    """
    # Extract points A and B
    A = points[:, 0]  # Shape: (N, 2)
    B = points[:, 1]  # Shape: (N, 2)

    # Compute the vector AB
    AB = B - A  # Shape: (N, 2)

    # Compute the angle of AB relative to the vertical line
    angles_radians = np.arctan2(AB[:, 0], AB[:, 1])  # atan2(x, y) gives angle with respect to vertical
    angles_degrees = -np.degrees(angles_radians)
    return angles_degrees
    
def calculate_head_alignment_angle_batch(angle1, angle2, tip1, tip2):
    """
    Calculate the angles needed to rotate vectors at multiple timestamps so that their heads align.

    Parameters:
    angle1 (numpy.ndarray): Array of shape (N,) containing angles (in degrees) of the first vector at each timestamp.
    angle2 (numpy.ndarray): Array of shape (N,) containing angles (in degrees) of the second vector at each timestamp.
    tip1 (numpy.ndarray): Array of shape (N, 2) containing the tips of the first vector at each timestamp.
    tip2 (numpy.ndarray): Array of shape (N, 2) containing the tips of the second vector at each timestamp.

    Returns:
    numpy.ndarray: Array of shape (N,) containing the angles (in degrees) needed to rotate each first vector to align its head with the corresponding second vector.
    """
    # Compute the difference in angles for all timestamps
    relative_angle = angle2 - angle1  # Shape: (N,)

    # Determine the direction of rotation based on the tips
    # If tip2 is to the right of tip1, keep the angle as is; otherwise, flip the direction
    direction_vectors = tip2 - tip1  # Shape: (N, 2)
    cross_products = np.cross(direction_vectors, np.array([0, 1]))  # Cross product to determine left/right
    signs = np.sign(cross_products)  # Shape: (N,)

    # Adjust the relative angles based on the signs
    aligned_angles = signs * relative_angle  # Shape: (N,)

    # Normalize to [-180, 180]
    aligned_angles = (aligned_angles + 180) % 360 - 180

    return aligned_angles

def calculate_rotation_to_point(points_AB, points_A_prime, points_B_prime):
    """
    Calculate the rotation angle needed to rotate the segment AB (keeping A fixed) so that AB points towards A'.

    Parameters:
    points_AB (numpy.ndarray): Array of shape (N, 2, 2), where:
        - The first axis contains the time points
        - The second axis contains points A and B for the segment
    points_A_prime (numpy.ndarray): Array of shape (N, 2), containing the positions of A' at each time point.
    points_B_prime (numpy.ndarray): Array of shape (N, 2), containing the positions of B' at each time point.

    Returns:
    numpy.ndarray: Array of shape (N,) containing the angles (in degrees) needed to rotate AB to point towards A'.
    """
    # Extract points A and B from the segment AB
    A = points_AB[:, 0]  # Shape: (N, 2)
    B = points_AB[:, 1]  # Shape: (N, 2)

    # Compute the vector AB and the desired direction AA' (from A to A')
    vector_AB = B - A  # Current direction of AB
    vector_AA_prime = points_A_prime - A  # Desired direction towards A'

    # Compute the angles of the vectors relative to the vertical
    angle_AB = np.degrees(np.arctan2(vector_AB[:, 0], vector_AB[:, 1]))  # Angle of AB with vertical
    angle_AA_prime = np.degrees(np.arctan2(vector_AA_prime[:, 0], vector_AA_prime[:, 1]))  # Angle of AA' with vertical

    # Compute the rotation needed to align AB with AA'
    rotation_angles = angle_AA_prime - angle_AB

    # Normalize to [-180, 180]
    rotation_angles = (rotation_angles + 180) % 360 - 180

    return rotation_angles

def calculate_rotation_to_point_top_left_origin(points_AB, points_A_prime):
    """
    Calculate the rotation angle needed to rotate the segment AB (keeping A fixed)
    so that AB points towards A', assuming the origin is at the top-left corner.

    Parameters:
    points_AB (numpy.ndarray): Array of shape (N, 2, 2), where:
        - The first axis contains the time points
        - The second axis contains points A and B for the segment
    points_A_prime (numpy.ndarray): Array of shape (N, 2), containing the positions of A' at each time point.

    Returns:
    numpy.ndarray: Array of shape (N,) containing the angles (in degrees) needed to rotate AB to point towards A'.
    """
    # Extract points A and B from the segment AB
    A = points_AB[:, 0]  # Shape: (N, 2)
    B = points_AB[:, 1]  # Shape: (N, 2)

    # Invert the y-coordinates to account for the top-left origin
    A[:, 1] *= -1
    B[:, 1] *= -1
    points_A_prime[:, 1] *= -1

    # Compute the vector AB and the desired direction AA' (from A to A')
    vector_AB = B - A  # Current direction of AB
    vector_AA_prime = points_A_prime - A  # Desired direction towards A'

    # Compute the angles of the vectors relative to the vertical
    angle_AB = np.degrees(np.arctan2(vector_AB[:, 0], vector_AB[:, 1]))  # Angle of AB with vertical
    angle_AA_prime = np.degrees(np.arctan2(vector_AA_prime[:, 0], vector_AA_prime[:, 1]))  # Angle of AA' with vertical

    # Compute the rotation needed to align AB with AA'
    rotation_angles = angle_AA_prime - angle_AB

    # Normalize to [-180, 180]
    rotation_angles = (rotation_angles + 180) % 360 - 180

    return rotation_angles


def compute_inter_orientation(loaders, df_):
    # compute inter_orientation between each pair of flies
    # inter_orientation: how many degrees to rotate the head of the focal fly
    # around its thorax so that the thorax-head segment points towards the thorax of the other fly
    # positive degrees -> clockwise
    # negative degrees -> counter-clockwise
    df_2=[]
    pairs=df_["pair"].unique().tolist()

    for pair in tqdm(pairs):

        df_pair=df_.loc[df_["pair"]==pair]
        
        fns=df_pair["frame_number"].tolist()
        id1, id2=pairs[0].split(" ")
        identity1=int(id1.split("|")[1])
        identity2=int(id2.split("|")[1])
            
        loader1=select_loader(loaders, identity1)
        loader2=select_loader(loaders, identity2)
        
        pose1=loader1.pose.loc[loader1.pose["frame_number"].isin(fns), ["frame_number", "head_x", "head_y", "thorax_x", "thorax_y", "orientation", "x", "y"]]
        pose1["head_x"]+=pose1["x"]
        pose1["head_y"]+=pose1["y"]
        pose1["thorax_x"]+=pose1["x"]
        pose1["thorax_y"]+=pose1["y"]
        
        pose2=loader2.pose.loc[loader2.pose["frame_number"].isin(fns), ["frame_number", "head_x", "head_y", "thorax_x", "thorax_y", "orientation", "x", "y"]]

        pose1=pose1.loc[pose1["frame_number"].isin(pose2["frame_number"])]
        pose2=pose2.loc[pose2["frame_number"].isin(pose1["frame_number"])]
        df_pair=df_pair.loc[df_pair["frame_number"].isin(pose1["frame_number"])]
        pose2=pose2.loc[pose2["frame_number"].isin(df_pair["frame_number"])]
        pose1=pose1.loc[pose1["frame_number"].isin(df_pair["frame_number"])]
    
        pose2["thorax_x"]+=pose2["x"]
        pose2["thorax_y"]+=pose2["y"]

        assert pose1.shape[0]==pose2.shape[0]==df_pair.shape[0]

        inter_orientation=calculate_rotation_to_point_top_left_origin(
            np.stack([
                pose1[["thorax_x", "thorax_y"]],
                pose1[["head_x", "head_y"]],
            ], axis=1),
            pose2[["thorax_x", "thorax_y"]].values,
        )
        # if df_pair.shape[0]!=len(inter_orientation):
        #     import ipdb; ipdb.set_trace()
        df_pair["inter_orientation"]=inter_orientation
        df_2.append(df_pair)
        
    df_=pd.concat(df_2, axis=0)
    return df_