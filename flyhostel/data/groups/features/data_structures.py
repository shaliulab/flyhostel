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


@dataclass
class GroupFrame:
    """All fly poses in a single frame."""
    flies: list[FlyPose]    # length = n_flies (2–10)
    t: Optional[int] = None
    frame_number: Optional[int] = None
