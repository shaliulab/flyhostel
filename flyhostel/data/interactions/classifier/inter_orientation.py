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
    Positive degrees = clock-wise and negative degrees = counter clock-wise
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

def calculate_interorientation_(pose1, pose2):
    return calculate_rotation_to_point_top_left_origin(
        np.stack([
            pose1[["thorax_x", "thorax_y"]],
            pose1[["head_x", "head_y"]],
        ], axis=1),
        pose2[["thorax_x", "thorax_y"]].values,
    )



def sync_datasources(*args):
    """
    Only frame number which appear in all data frames will be present in the output
    """
    df1=args[0]
    df2=args[1]
    df1=df1.loc[df1["frame_number"].isin(df2["frame_number"])]
    df2=df2.loc[df2["frame_number"].isin(df1["frame_number"])]

    out=[df1, df2]
    for df in args[2:]:
        df=df.loc[df["frame_number"].isin(df1["frame_number"])]
        df1=df1.loc[df1["frame_number"].isin(df["frame_number"])]
        df2=df2.loc[df2["frame_number"].isin(df["frame_number"])]
        out.append(df)

    for df in out:
        gaps, counts=np.unique(df["frame_number"].diff(), return_counts=True)
        
        # check that there are no gaps
        # i.e. all frames are followed by the +1 frame, and only
        # one frame (the first one) has no preceding frame
        assert gaps[0]==1
        assert np.isnan(gaps[1])
        assert counts[1]==1

    return out

def compute_inter_orientation_one_pair(loader1, loader2, nan_policy="omit"):
    """
    Return the inter orientatin of animal1 with respect to animal2

    inter_orientation contains the degrees that the head of animal1
    needs to be rotated around its thorax
    so that the thorax->head line points
    towards the thorax of animal2

    Only rows for which the pose of both animals is complete are populated 
    """
    assert 'head_x' in loader1.pose.columns
    assert 'thorax_x' in loader1.pose.columns
    assert 'thorax_x' in loader2.pose.columns

    pose1=loader1.pose[["frame_number", "head_x", "head_y", "thorax_x", "thorax_y", "center_x", "center_y"]]
    pose2=loader2.pose[["frame_number", "head_x", "head_y", "thorax_x", "thorax_y", "center_x", "center_y"]]
    pose1, pose2=sync_datasources(pose1, pose2)
    assert pose1.shape[0]==pose2.shape[0]
    
    pose1=project_to_absolute_coords(pose1, "head")
    pose1=project_to_absolute_coords(pose1, "thorax")
    pose2=project_to_absolute_coords(pose2, "thorax")

    inter_orientation_df=pd.DataFrame({
        "frame_number": pose1["frame_number"],
        "inter_orientation": calculate_interorientation_(pose1, pose2)
    })

    if inter_orientation_df["inter_orientation"].isna().sum()>0:
        if nan_policy=="raise":
            raise ValueError("Inter orientation cannot be computed for all frames requested")
        elif nan_policy=="omit":
            inter_orientation_df=inter_orientation_df.loc[
                ~inter_orientation_df["inter_orientation"].isna()
            ]
        elif nan_policy=="propagate":
            pass
    return inter_orientation_df


def compute_inter_orientation(loaders, df_):
    # compute inter_orientation between each pair of flies
    # inter_orientation: how many degrees to rotate the head of the focal fly
    # around its thorax so that the thorax-head segment points towards the thorax of the other fly
    # positive degrees -> clockwise
    # negative degrees -> counter-clockwise
    df_2=[]
    pairs=df_["pair"].unique().tolist()
    raise NotImplementedError()

    for pair in tqdm(pairs):

        df_pair=df_.loc[df_["pair"]==pair]
        compute_inter_orientation_one_pair(id1, id2, loaders, df_pair)

        df_2.append(df_pair)
        
    df_=pd.concat(df_2, axis=0)
    return df_