"""
Parameters of the social-interaction pipelines.

Every threshold is in physical units (mm, s, mm/s, deg), never in pixels or frames, so
the same values apply to the 150 fps and the ~47 fps rigs. Values marked CALIBRATE are
first guesses: histogram the corresponding output column on hand-labelled events and
put the cut in the gap.
"""
from dataclasses import dataclass, field, asdict

from flyhostel.data.pose.constants import (
    DIST_MAX_MM,
    MIN_INTERACTION_DURATION,
    MIN_TIME_BETWEEN_INTERACTIONS,
    min_score as POSE_MIN_SCORE,
)
from flyhostel.data.interactions.touch_from_pose.constants import LONG_IMMOBILE_SECONDS


# Capsule radius of each skeleton segment kind (see geometry.SEGMENTS).
# Two flies touch when the distance between two segments minus both radii
# drops below ContactConfig.touch_mm.
DEFAULT_RADII_MM = {
    "head":      0.35,   # thorax->head capsule (anterior thorax + head)        CALIBRATE
    "abdomen":   0.45,   # thorax->abdomen capsule (the widest part of the fly) CALIBRATE
    "proboscis": 0.03,
    "wing":      0.10,
    "leg":       0.03,
}


@dataclass
class TrackConfig:
    fps: float = 30.0          # analysis rate; native frames are subsampled to this.   unit: Hz
    max_gap_s: float = 1.0     # centroid gaps up to this long are forward-filled
                               #   (chunk boundaries). Longer gaps stay NaN.              unit: s
    scale_attr: str = "pixels_per_mm"
                               # loader attribute giving px/mm. "pixels_per_mm" is the
                               #   metadata calibration (used by the PE pipeline);
                               #   "px_per_mm" is roi_width/60 and differs by ~10%.


@dataclass
class MovementConfig:
    smooth_s: float = 0.5          # speed = displacement across this window / window.   unit: s
    move_speed_mm_s: float = 1.0   # centroid speed above this = moving.                 unit: mm/s
                                   #   (same as PE BODY_MOVE_MM_S)                      CALIBRATE
    ignore_move_s: float = 0.5     # moving runs shorter than this do not break an
                                   #   immobility bout (tracking blips).                 unit: s
    min_sleep_s: float = 300.0     # immobility bout this long = sleep (5-min rule).     unit: s


@dataclass
class ProximityConfig:
    max_dist_mm: float = DIST_MAX_MM                     # centroid distance = interacting.  unit: mm
    min_duration_s: float = MIN_INTERACTION_DURATION     # shorter bouts are dropped.        unit: s
    merge_gap_s: float = MIN_TIME_BETWEEN_INTERACTIONS   # bouts closer than this merge.     unit: s
    approach_s: float = 2.0        # window before onset used to decide who approached
                                   #   and whether each fly was moving.                  unit: s


@dataclass
class ContactConfig:
    gate_mm: float = 5.5           # pose contact is only evaluated while centroids are
                                   #   closer than this (2 body lengths: legs can touch). unit: mm
    touch_mm: float = 0.10         # capsule gap below this = touching.                  unit: mm  CALIBRATE
    min_duration_s: float = 0.10   # shorter touches are dropped.                        unit: s
    merge_gap_s: float = 0.30      # touches closer than this merge into one bout.       unit: s
    radius_mm: dict = field(default_factory=lambda: dict(DEFAULT_RADII_MM))
    min_score: dict = field(default_factory=lambda: dict(POSE_MIN_SCORE))
                                   # keypoints below their SLEAP score are set to NaN
    block_frames: int = 5000       # frames per vectorised batch (bounds memory)
    gpu_block_frames: int = 50000  # same, when use_gpu (5-8x faster than CPU when warm)


@dataclass
class ResponseConfig:
    # --- which contacts count as "a moving fly approaches an immobile fly" ---
    min_pre_immobile_s: float = LONG_IMMOBILE_SECONDS  # recipient immobile at least this long
                                                       #   before contact.                unit: s
    approach_share_min: float = 0.6  # initiator contributes at least this fraction of the
                                     #   closing speed in the approach window.          unit: fraction

    # --- response grading (recipient) ---
    baseline_s: float = 8.0        # recipient window before the approach used as its own
                                   #   noise floor (ends approach_s before contact).      unit: s
    response_s: float = 3.0        # window after contact onset that is graded.          unit: s
    smooth_s: float = 0.2          # limb speed = path length over this window / window. unit: s
    limb_move_mm_s: float = 2.0    # body-frame limb speed above this = limb moved.      unit: mm/s  CALIBRATE
    baseline_q: float = 0.95       # per-limb threshold is also at least k * this
    baseline_k: float = 1.5        #   quantile of the limb's baseline speed (jitter floor)
    min_limb_s: float = 0.10       # limb activity shorter than this is ignored.         unit: s
    sustained_s: float = 0.50      # limb activity at least this long = postural.        unit: s
    turn_deg: float = 20.0         # heading change at least this = postural.            unit: deg  CALIBRATE
    shift_mm: float = 0.25         # thorax displacement at least this = postural.       unit: mm   CALIBRATE
    escape_mm: float = 1.00        # thorax displacement at least this = locomotor.      unit: mm   CALIBRATE
    min_coverage: float = 0.5      # graded only if the recipient thorax is tracked in at
                                   #   least this fraction of the response window.       unit: fraction

    # --- sleep after the interaction ---
    max_sleep_latency_s: float = 7200.0  # later sleep onsets are reported as censored.  unit: s


@dataclass
class PassByConfig:
    radius_mm: float = 5.0         # circle around the focal fly the passer enters and leaves. unit: mm
    focal_state: str = "asleep"    # focal fly at entry: "asleep" | "immobile" | "awake" | "any".
                                   #   "awake" is the negative control; "any" keeps both and
                                   #   lets you split on the focal_asleep column.
    min_focal_immobile_s: float = LONG_IMMOBILE_SECONDS  # for focal_state="immobile".   unit: s
    merge_gap_s: float = 0.3       # exits shorter than this (jitter on the circle edge) do
                                   #   not end the event.                                 unit: s
    max_inside_s: float = None     # optional: drop events where the passer stays inside
                                   #   longer than this (it stopped, not passed).         unit: s


@dataclass
class SocialConfig:
    tracks: TrackConfig = field(default_factory=TrackConfig)
    movement: MovementConfig = field(default_factory=MovementConfig)
    proximity: ProximityConfig = field(default_factory=ProximityConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    passby: PassByConfig = field(default_factory=PassByConfig)

    def to_dict(self):
        return asdict(self)
