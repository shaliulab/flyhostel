import numpy as np

from .data_structures import (
    GroupFrame,
    FlyPose
)
from flyhostel.data.pose.constants import legs as LEGS
LEGS=[leg for leg in LEGS if "J" not in leg]
LEGS




# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _centroid(pose: FlyPose) -> np.ndarray:
    """Body centroid = thorax (most stable keypoint)."""
    return pose.thorax


def _heading(pose: FlyPose) -> float:
    """
    Heading angle in radians [-π, π].
    Defined as the direction from abdomen to head (anterior axis).
    """
    vec = pose.head - pose.abdomen
    return np.arctan2(vec[1], vec[0])


def _body_axis(pose: FlyPose) -> np.ndarray:
    """Unit vector along the anterior axis (abdomen → head)."""
    vec = pose.head - pose.abdomen
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array([1.0, 0.0])
    return vec / norm


def _body_length(pose: FlyPose) -> float:
    return float(np.linalg.norm(pose.head - pose.abdomen))


def _wing_span(pose: FlyPose) -> float:
    return float(np.linalg.norm(pose.wing_l - pose.wing_r))


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Unsigned angle in [0, π] between two vectors."""
    cos = np.clip(np.dot(v1, v2) /
                  (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1)
    return float(np.arccos(cos))


def _signed_bearing(focal: FlyPose, other: FlyPose) -> float:
    """
    Signed angle (radians) between focal fly's heading and the vector
    pointing from focal toward other.  Positive = other is to the left.
    """
    to_other = _centroid(other) - _centroid(focal)
    heading = _body_axis(focal)
    cross = heading[0] * to_other[1] - heading[1] * to_other[0]   # z of 2D cross
    dot   = np.dot(heading, to_other)
    return float(np.arctan2(cross, dot))


# ---------------------------------------------------------------------------
# Pre-extraction: pull everything out of xarray in one shot
# ---------------------------------------------------------------------------

KEYPOINT_ORDER = ["head", "thorax", "abdomen", "proboscis", "lW", "rW"] + LEGS

def _preextract_arrays(pose_features, step_frames):
    """
    Pre-extract numpy arrays from xarray for fast multiprocessing.
    Now also extracts distance features.
    """
    positions = pose_features.position.values[::step_frames]  # (T, 2, keypoints, individuals)
    times = pose_features.time.values[::step_frames]
    frame_numbers = pose_features.frame_number.values[::step_frames]
    
    # Extract distance features (these are per-frame, may be per-individual or shared)
    food_distance = pose_features["food_distance"].values[::step_frames]
    notch_distance = pose_features["notch_distance"].values[::step_frames]
    edge_distance = pose_features["edge_distance"].values[::step_frames]
    food_cos = pose_features["food_cos"].values[::step_frames]
    food_sin = pose_features["food_sin"].values[::step_frames]

    food_distance = np.squeeze(food_distance)  # Removes all singleton dimensions
    notch_distance = np.squeeze(notch_distance)
    edge_distance = np.squeeze(edge_distance)
    food_cos = np.squeeze(food_cos)
    food_sin = np.squeeze(food_sin)

    # Get keypoint index and n_legs from pose_features
    individuals = pose_features.individuals.values
    n_individuals = len(individuals)
    keypoints = pose_features.keypoints.values
    kp_index = {kp: i for i, kp in enumerate(keypoints)}
    
    # Count legs (everything after lW and rW)
    n_legs = len([kp for kp in keypoints if kp.startswith('leg')])
    
    return (
        positions, times, frame_numbers,
        food_distance, notch_distance, edge_distance,
        food_cos, food_sin,
        n_individuals, kp_index, n_legs
    )

# ---------------------------------------------------------------------------
# Worker function (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _build_frame(args) -> GroupFrame:
    """
    Build a GroupFrame from pre-extracted numpy slices.
    args = (t_idx, time_val, frame_number_val, frame_positions, 
            food_dist, notch_dist, edge_dist, n_individuals, kp_index, n_legs)
    """
    (t_idx, time_val, frame_number_val, frame_positions,
     food_dist, notch_dist, edge_dist,
     food_cos, food_sin,
     n_individuals, kp_index, n_legs) = args
     
    leg_idx = [i for i, bp in enumerate(kp_index.keys()) if bp in LEGS]

    poses = []
    food_dist_list = []
    notch_dist_list = []
    edge_dist_list = []
    food_cos_list=[]
    food_sin_list=[]
    n_dims = 2
    
    for ind_idx in range(n_individuals):
        kps = frame_positions[..., ind_idx]  # (n_keypoints, 2)
        found_dims = len(kps[:, kp_index["head"]])
        assert found_dims == n_dims, f"{found_dims}!={n_dims}"

        pose = FlyPose(
            head      = kps[:, kp_index["head"]],
            thorax    = kps[:, kp_index["thorax"]],
            abdomen   = kps[:, kp_index["abdomen"]],
            proboscis = kps[:, kp_index["proboscis"]],
            wing_l    = kps[:, kp_index["lW"]],
            wing_r    = kps[:, kp_index["rW"]],
            legs      = [kps[:, leg_index] for leg_index in leg_idx],
        )
        poses.append(pose)
        
        # Handle distances per individual
        if isinstance(food_dist, np.ndarray) and food_dist.ndim > 0:
            food_dist_list.append(food_dist[ind_idx] if len(food_dist) > ind_idx else food_dist[0])
            notch_dist_list.append(notch_dist[ind_idx] if len(notch_dist) > ind_idx else notch_dist[0])
            edge_dist_list.append(edge_dist[ind_idx] if len(edge_dist) > ind_idx else edge_dist[0])
            food_cos_list.append(food_cos[ind_idx] if len(food_cos) > ind_idx else food_cos[0])
            food_sin_list.append(food_sin[ind_idx] if len(food_sin) > ind_idx else food_sin[0])
        else:
            food_dist_list.append(food_dist)
            notch_dist_list.append(notch_dist)
            edge_dist_list.append(edge_dist)
            edge_dist_list.append(edge_dist)
            food_cos_list.append(food_cos)
            food_sin_list.append(food_sin)

    return GroupFrame(
        flies=poses,
        t=time_val,
        frame_number=frame_number_val,
        food_distance=np.array(food_dist_list),
        notch_distance=np.array(notch_dist_list),
        edge_distance=np.array(edge_dist_list),
        food_cos=np.array(food_cos_list),
        food_sin=np.array(food_sin_list)
    )
