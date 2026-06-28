from dataclasses import dataclass
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class FlyPose:
    """
    Pose of a single fly in one frame.
    All coordinates are (x, y) in pixels (or mm if calibrated).
    """
    head: np.ndarray        # (2,)
    thorax: np.ndarray      # (2,)  — used as body centroid
    abdomen: np.ndarray     # (2,)
    wing_l: np.ndarray      # (2,)
    wing_r: np.ndarray      # (2,)
    proboscis: np.ndarray   # (2,)
    legs: np.ndarray        # (6, 2)


from dataclasses import dataclass
from typing import List

@dataclass
class GroupFrame:
    """Container for a single timepoint of a fly group."""
    flies: List['FlyPose']
    t: float
    frame_number: int
    food_distance: np.ndarray      # (n_individuals,) or scalar
    notch_distance: np.ndarray     # (n_individuals,) or scalar
    edge_distance: np.ndarray      # (n_individuals,) or scalar

@dataclass
class Fly:
    """Single fly pose + derived features at one timepoint."""
    position: np.ndarray          # (2,) [x, y]
    velocity: np.ndarray          # (2,) [vx, vy]
    acceleration: np.ndarray      # (2,) [ax, ay]
    heading: float
    heading_change: float
    body_length: float
    wing_angle_l: float
    wing_angle_r: float
    wing_span: float
    sleep_state: int
    
    # NEW: Distance features
    food_distance: float
    notch_distance: float
    edge_distance: float

def flypose_to_fly(
    fly_idx: int,
    frame: GroupFrame,
    flypose: FlyPose,
    prev_flypose: FlyPose = None,
    prev_prev_flypose: FlyPose = None,
    sleep_labels: bool = False,
) -> Fly:
    """
    Convert a FlyPose (keypoints) to a Fly (kinematic features) at one frame.
    
    Parameters
    ----------
    flypose         : current frame's FlyPose
    prev_flypose    : previous frame's FlyPose (for velocity)
    prev_prev_flypose : two frames back (for acceleration)
    sleep_labels    : is this fly asleep?
    """
    
    # Position: center of mass of body keypoints
    body_kpts = np.array([
        flypose.head,
        flypose.thorax,
        flypose.abdomen,
    ])
    position = np.mean(body_kpts, axis=0)  # (2,)
    
    # Body orientation: thorax → abdomen vector
    body_vector = flypose.abdomen - flypose.thorax
    heading = np.arctan2(body_vector[1], body_vector[0])
    
    # Body length: thorax-to-abdomen distance
    body_length = np.linalg.norm(body_vector)
    
    # Velocity (backward difference)
    if prev_flypose is not None:
        prev_body_kpts = np.array([
            prev_flypose.head,
            prev_flypose.thorax,
            prev_flypose.abdomen,
        ])
        prev_position = np.mean(prev_body_kpts, axis=0)
        velocity = position - prev_position
    else:
        velocity = np.array([0.0, 0.0])
    
    # Acceleration (backward difference of velocity)
    if prev_prev_flypose is not None and prev_flypose is not None:
        prev_prev_body_kpts = np.array([
            prev_prev_flypose.head,
            prev_prev_flypose.thorax,
            prev_prev_flypose.abdomen,
        ])
        prev_prev_position = np.mean(prev_prev_body_kpts, axis=0)
        prev_velocity = prev_position - prev_prev_position
        acceleration = velocity - prev_velocity
    else:
        acceleration = np.array([0.0, 0.0])
    
    # Heading change (angular velocity)
    if prev_flypose is not None:
        prev_body_vector = prev_flypose.abdomen - prev_flypose.thorax
        prev_heading = np.arctan2(prev_body_vector[1], prev_body_vector[0])
        heading_change = np.arctan2(np.sin(heading - prev_heading), np.cos(heading - prev_heading))
    else:
        heading_change = 0.0
    
    # Wing angles
    # Wing angle: angle between wing and body
    lw_vector = flypose.wing_l - flypose.thorax
    rw_vector = flypose.wing_r - flypose.thorax
    wing_angle_l = np.arctan2(lw_vector[1], lw_vector[0]) - heading
    wing_angle_r = np.arctan2(rw_vector[1], rw_vector[0]) - heading
    wing_span = np.linalg.norm(flypose.wing_l - flypose.wing_r)
    
    food_dist = frame.food_distance[fly_idx] if isinstance(frame.food_distance, np.ndarray) else frame.food_distance
    notch_dist = frame.notch_distance[fly_idx] if isinstance(frame.notch_distance, np.ndarray) else frame.notch_distance
    edge_dist = frame.edge_distance[fly_idx] if isinstance(frame.edge_distance, np.ndarray) else frame.edge_distance

    return Fly(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        heading=heading,
        heading_change=heading_change,
        body_length=body_length,
        wing_angle_l=wing_angle_l,
        wing_angle_r=wing_angle_r,
        wing_span=wing_span,
        sleep_state=sleep_labels,
        food_distance=food_dist,      # ← NEW
        notch_distance=notch_dist,    # ← NEW
        edge_distance=edge_dist,      # ← NEW
    )


def frames_to_fly_list(
    frames: list[GroupFrame],
    fly_idx: int = 0,
    sleep_labels: np.ndarray = None,
) -> list[Fly]:
    """
    Extract a single fly's trajectory across all frames.
    
    Parameters
    ----------
    frames       : list of GroupFrame objects (from your build_frames_parallel)
    fly_idx      : which fly to extract (0 for first fly, etc.)
    sleep_labels : (n_frames,) array of sleep state (optional)
    
    Returns
    -------
    fly_trajectory : list of Fly objects, one per frame
    """
    fly_trajectory = []
    
    for t_idx, frame in enumerate(frames):
        if fly_idx >= len(frame.flies):
            continue
        
        curr_pose = frame.flies[fly_idx]
        prev_pose = frame.flies[fly_idx] if t_idx > 0 else None
        prev_prev_pose = frame.flies[fly_idx] if t_idx > 1 else None
        
        if t_idx > 0:
            prev_pose = frames[t_idx - 1].flies[fly_idx]
        if t_idx > 1:
            prev_prev_pose = frames[t_idx - 2].flies[fly_idx]
        
        asleep = sleep_labels[t_idx] if sleep_labels is not None else False
        
        fly = flypose_to_fly(fly_idx, frame, curr_pose, prev_pose, prev_prev_pose, asleep)
        fly_trajectory.append(fly)
    
    return fly_trajectory


def extract_individual_features(fly: Fly) -> dict:
    """
    Extract non-social features from a single Fly object.
    """
    return {
        # 'x': fly.position[0],                                                       # 0
        # 'y': fly.position[1],                                                       # 1
        'speed': np.linalg.norm(fly.velocity),                                      # 0
        'speed_x': fly.velocity[0],                                                 # 1
        'speed_y': fly.velocity[1],                                                 # 2
        'acceleration': np.linalg.norm(fly.acceleration),                           # 3
        'heading': fly.heading,                                                     # 4
        'heading_change': fly.heading_change,                                       # 5
        'heading_stability': 1.0 - np.clip(np.abs(fly.heading_change), 0, 1),       # 6
        'body_length': fly.body_length,                                             # 7
        'wing_angle_l': fly.wing_angle_l,                                           # 8
        'wing_angle_r': fly.wing_angle_r,                                           # 9
        'wing_span': fly.wing_span,                                                 # 10
        'wing_asymmetry': np.abs(fly.wing_angle_l - fly.wing_angle_r),              # 11
        'is_asleep': float(fly.sleep_state),                                        # 12
        # Arena-relative features (replaces x, y)
        'dist_to_food': fly.food_distance,                                          # 13
        'dist_to_notch': fly.notch_distance,                                        # 14
        'dist_to_edge': fly.edge_distance,                                          # 15
    }


# def extract_individual_timeseries_from_frames(
#     frames: list[GroupFrame],
#     fly_idx: int = 0,
#     sleep_labels: np.ndarray = None,
# ) -> tuple[np.ndarray, list[str]]:
#     """
#     Extract individual features for one fly across all frames.
    
#     Parameters
#     ----------
#     frames       : list of GroupFrame objects
#     fly_idx      : which fly in the group (0 for first, etc.)
#     sleep_labels : (n_frames,) array of sleep state
    
#     Returns
#     -------
#     X    : np.ndarray of shape (n_frames, n_features)
#     keys : list of feature names
#     """
#     # Convert frames to Fly trajectory
#     fly_trajectory = frames_to_fly_list(frames, fly_idx, sleep_labels)
    
#     # Extract features for each frame
#     rows = []
#     keys = None
#     for fly in fly_trajectory:
#         feat = extract_individual_features(fly)
#         if keys is None:
#             keys = list(feat.keys())
#         rows.append([feat[k] for k in keys])
    
#     X = np.array(rows)
#     return X, keys


def extract_individual_timeseries_from_frames(
    frames: list[GroupFrame],
    fly_idx: int = 0,
    sleep_labels: np.ndarray = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract individual features for one fly across all frames.
    """
    fly_trajectory = frames_to_fly_list(frames, fly_idx, sleep_labels)
    
    # Extract features for each frame
    rows = []
    keys = None
    for frame_idx, fly in enumerate(fly_trajectory):
        feat = extract_individual_features(fly)
        
        if keys is None:
            keys = list(feat.keys())
        
        # DEBUG: Check for non-scalar values
        row = []
        for k in keys:
            val = feat[k]
            if isinstance(val, (list, np.ndarray)):
                print(f"Frame {frame_idx}, Feature '{k}': is a sequence! Value: {val}")
                raise ValueError(f"Feature '{k}' returned a sequence instead of scalar: {val}")
            row.append(val)
        
        rows.append(row)
    
    X = np.array(rows)
    return X, keys