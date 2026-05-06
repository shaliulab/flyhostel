import numpy as np
from .utils import (
    _body_axis,
    _centroid,
    _signed_bearing,
)
from .data_structures import (
    FlyPose,
)


# ---------------------------------------------------------------------------
# Individual-in-group-context features (aggregated over all flies)
# ---------------------------------------------------------------------------

def _individual_context_features(
        fly: FlyPose,
        others: list[FlyPose],
        contact_radius: float = 5.0,   # pixels (tune to your scale)
        local_radius:   float = 20.0,
) -> dict:
    """
    Features for one focal fly relative to the rest of the group.
    """
    if not others:
        return {}

    ci = _centroid(fly)
    dists = [float(np.linalg.norm(_centroid(o) - ci)) for o in others]
    bearings = [abs(_signed_bearing(fly, o)) for o in others]
    headings_others = [_body_axis(o) for o in others]

    nn_dist  = min(dists)
    nn_idx   = int(np.argmin(dists))
    nn_bearing = bearings[nn_idx]

    # Local density
    n_in_radius = sum(1 for d in dists if d < local_radius)

    # Heading alignment with nearest neighbour
    nn_alignment = float(np.dot(_body_axis(fly), headings_others[nn_idx]))

    # Mean alignment with ALL others
    mean_alignment = float(np.mean([
        np.dot(_body_axis(fly), ho) for ho in headings_others
    ]))

    # Is any other fly's wing within contact_radius?
    wing_contact = int(any(
        np.linalg.norm(fly.wing_l - _centroid(o)) < contact_radius or
        np.linalg.norm(fly.wing_r - _centroid(o)) < contact_radius
        for o in others
    ))

    # Is proboscis in contact range with any other?
    prob_contact = int(any(
        np.linalg.norm(fly.proboscis - _centroid(o)) < contact_radius
        for o in others
    ))

    # --- Isolation index (was described but not implemented) ---
    # Mean + std of distances to ALL others, normalised by group size.
    # High mean + high std → fly is far from everyone and the group is spread.
    # High mean + low std  → fly is uniformly distant (peripheral but not alone).
    # Low mean             → fly is embedded in the group.
    isolation_mean = float(np.mean(dists))
    isolation_std  = float(np.std(dists))
    # Harmonic mean of distances: down-weights very distant flies,
    # sensitive to whether even one neighbour is close
    isolation_harmonic = float(len(dists) / np.sum(1.0 / (np.array(dists) + 1e-9)))


    return {
        "nn_dist":            nn_dist,
        "nn_bearing":         nn_bearing,
        "n_in_radius":        n_in_radius,
        "nn_alignment":       nn_alignment,
        "mean_alignment":     mean_alignment,
        "wing_contact":       wing_contact,
        "prob_contact":       prob_contact,
        "isolation_mean":     isolation_mean,
        "isolation_std":      isolation_std,
        "isolation_harmonic": isolation_harmonic,
    }


def _aggregate_individual_context(per_fly: list[dict]) -> dict:
    """Aggregate individual-context features over all flies in the group."""
    if not per_fly:
        return {}
    out = {}
    _AGG_KEYS=[
        "nn_dist", "nn_bearing", "n_in_radius",
        "nn_alignment", "mean_alignment",
        "isolation_mean", "isolation_std", "isolation_harmonic"
    ]
    
    for key in _AGG_KEYS:
        vals = np.array([d[key] for d in per_fly])
        out[f"ctx_{key}_min"]  = float(vals.min())
        out[f"ctx_{key}_mean"] = float(vals.mean())
        out[f"ctx_{key}_max"]  = float(vals.max())
        out[f"ctx_{key}_std"]  = float(vals.std())
    out["ctx_wing_contact_frac"] = float(
        np.mean([d["wing_contact"] for d in per_fly])
    )
    out["ctx_prob_contact_frac"] = float(
        np.mean([d["prob_contact"] for d in per_fly])
    )
    return out
