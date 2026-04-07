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

def _preextract_arrays(pose_features, step_frames: int) -> tuple:
    """
    Extract the full numpy array from xarray once, avoiding repeated .sel() calls.
    Returns:
        positions   : np.ndarray (n_times, n_individuals, n_keypoints, 2)
        times       : np.ndarray of selected time values
        individuals : list of individual labels
        kp_index    : dict mapping keypoint name → integer index into axis=2
    """
    times = pose_features.time.values[::step_frames]

    fn=pose_features.frame_number
    assert len(fn.shape)==1
    
    frame_numbers = fn.values[::step_frames]
    
    # Reindex keypoints into a known order so we can slice by integer index
    pose_reindexed = pose_features.sel(
        time=times,
        keypoints=KEYPOINT_ORDER,
    )

    # Single large array pull: (n_times, n_individuals, n_keypoints, 2)
    positions = pose_reindexed.position.values  # shape: (T, N, K, 2)

    individuals = list(pose_features.individuals.values)
    kp_index    = {kp: i for i, kp in enumerate(KEYPOINT_ORDER)}
    n_legs      = len(LEGS)

    return positions, times, frame_numbers, individuals, kp_index, n_legs


# ---------------------------------------------------------------------------
# Worker function (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _build_frame(args) -> GroupFrame:
    """
    Build a GroupFrame from pre-extracted numpy slices.
    args = (t_idx, time_val, frame_positions, n_legs, kp_index)
    frame_positions : np.ndarray (n_keypoints, 2, n_individuals)
    """
    t_idx, time_val, frame_number_val, frame_positions, n_legs, kp_index = args

    n_individuals = frame_positions.shape[2]
    n_leg_start   = 6  # head, thorax, abdomen, proboscis, lW, rW come first

    poses = []
    n_dims=2
    for ind_idx in range(n_individuals):
        kps = frame_positions[..., ind_idx]  # (n_keypoints, 2)
        found_dims=len(kps[:, kp_index["head"]])

        assert found_dims==n_dims, f"{found_dims}!={n_dims}"

        pose = FlyPose(
            head      = kps[:, kp_index["head"]],
            thorax    = kps[:, kp_index["thorax"]],
            abdomen   = kps[:, kp_index["abdomen"]],
            proboscis = kps[:, kp_index["proboscis"]],
            wing_l    = kps[:, kp_index["lW"]],
            wing_r    = kps[:, kp_index["rW"]],
            legs      = [kps[:, n_leg_start + i] for i in range(n_legs)],
        )
        poses.append(pose)

    return GroupFrame(flies=poses, t=time_val, frame_number=frame_number_val)

