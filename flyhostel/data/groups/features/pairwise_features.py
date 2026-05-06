import numpy as np
from .utils import (
    _body_axis,
    _centroid,
    _angle_between,
    _signed_bearing,
)
from .data_structures import (
    FlyPose,
)

# ---------------------------------------------------------------------------
# Pairwise features  (for a given pair i, j)
# ---------------------------------------------------------------------------

def pairwise_features(fi: FlyPose, fj: FlyPose) -> dict:
    """
    All features for an ordered pair (fi=focal, fj=other).
    Call twice with i,j swapped if you want directed features for both.
    """
    ci, cj = _centroid(fi), _centroid(fj)
    diff = cj - ci
    dist = float(np.linalg.norm(diff))

    ai, aj = _body_axis(fi), _body_axis(fj)

    # --- relative orientation ---
    heading_diff = _angle_between(ai, aj)           # [0, π]
    # antiparallel = π, parallel = 0

    # --- facing angle: is fi oriented toward fj? ---
    facing_angle = abs(_signed_bearing(fi, fj))     # [0, π]; 0 = fi faces fj

    # --- approach angle: from fj's frame, where is fi coming from? ---
    approach_angle = abs(_signed_bearing(fj, fi))   # [0, π]; 0 = fj faces fi

    # --- mutual facing: both look at each other ---
    mutual_facing = float(facing_angle < np.pi / 4 and approach_angle < np.pi / 4)

    # --- wing proximity to other's body ---
    wing_l_to_body_j = float(np.linalg.norm(fi.wing_l - cj))
    wing_r_to_body_j = float(np.linalg.norm(fi.wing_r - cj))
    min_wing_to_body = min(wing_l_to_body_j, wing_r_to_body_j)

    # --- proboscis to body (feeding / contact) ---
    prob_to_body_j = float(np.linalg.norm(fi.proboscis - cj))
    prob_to_head_j = float(np.linalg.norm(fi.proboscis - fj.head))

    # --- closest leg pair distance ---
    min_leg_dist = float(min(
        np.linalg.norm(li - lj)
        for li in fi.legs for lj in fj.legs
    ))

    # --- abdomen-to-head distance (relevant for posterior contact / copulation) ---
    abd_to_head_j = float(np.linalg.norm(fi.abdomen - fj.head))
    head_to_abd_j = float(np.linalg.norm(fi.head   - fj.abdomen))

    # --- parallel alignment (1=parallel, -1=antiparallel) ---
    parallel_score = float(np.dot(ai, aj))

    return {
        "dist_centroid":        dist,
        "heading_diff":         heading_diff,
        "facing_angle":         facing_angle,
        "approach_angle":       approach_angle,
        "mutual_facing":        mutual_facing,
        "min_wing_to_body":     min_wing_to_body,
        "prob_to_body":         prob_to_body_j,
        "prob_to_head":         prob_to_head_j,
        "min_leg_dist":         min_leg_dist,
        "abd_to_head":          abd_to_head_j,
        "head_to_abd":          head_to_abd_j,
        "parallel_score":       parallel_score,
    }


# ---------------------------------------------------------------------------
# Aggregation over all pairs → group-size-invariant vector
# ---------------------------------------------------------------------------

_AGG_KEYS = ["dist_centroid", "heading_diff", "facing_angle",
             "min_wing_to_body", "prob_to_body", "min_leg_dist",
             "abd_to_head", "parallel_score"]

def _aggregate_pairwise(all_pw: list[dict]) -> dict:
    """
    Given a list of pairwise feature dicts (both directions, all pairs),
    return min / mean / max / std for each scalar feature.
    This is permutation-invariant and group-size-invariant.
    """
    out = {}
    for key in _AGG_KEYS:
        vals = np.array([d[key] for d in all_pw])
        out[f"pw_{key}_min"]  = float(vals.min())
        out[f"pw_{key}_mean"] = float(vals.mean())
        out[f"pw_{key}_max"]  = float(vals.max())
        out[f"pw_{key}_std"]  = float(vals.std())
    # mutual_facing is boolean — fraction of pairs
    out["pw_mutual_facing_frac"] = float(
        np.mean([d["mutual_facing"] for d in all_pw])
    )
    return out
