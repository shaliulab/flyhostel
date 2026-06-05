# Feature Reference

This document describes every per-frame feature computed by `jaaba_features`,
explains what it means biologically, and shows how each scales when you go
from 2 flies to N flies.

## How features scale with number of flies

Features split into three categories by their dependency:

| Category       | Depends on        | # series per chamber for N flies |
|----------------|-------------------|----------------------------------|
| Per-fly        | one fly's pose    | N                                |
| Per-pair       | two flies' poses  | N · (N−1) directed pairs         |
| Closest-fly    | one focal fly + all others | N                       |

For your DGRP screen with **2M:2F (N = 4)**:
- Per-fly features: 4 time series per feature
- Per-pair features (all directed pairs): 12 time series per feature
- Per-pair features (just the 4 directed male→female pairs you care about): 4
- Closest-fly features: 4 time series per feature

`extract_features` handles all three. Use `other_fly_id` to restrict pair
features to a single target fly per call; otherwise it computes all directed
pairs and names columns `dcenter__vs2`, `dcenter__vs3`, etc.

## Units convention

- Lengths: **mm**
- Velocities: **mm/s**
- Accelerations: **mm/s²**
- Angles: **radians**, on [−π, π) by JAABA convention
- Angular velocities: **rad/s**
- Areas: **mm²**
- Counts: **unitless integers**

## Coordinate conventions

- `(x, y)` are image pixel coordinates converted to mm via `px_per_mm`
- `theta` is the body orientation, measured from the +x axis to the body-forward
  vector (thorax→head) using `arctan2(dy, dx)`. In image coordinates with y-down,
  `theta = 0` means the fly faces the right side of the frame; `theta = +π/2`
  means it faces the bottom of the frame.
- Body-relative angles (`anglefrom1to2`, `yaw`, etc.) are in fly1's body frame:
  **0 means directly ahead**, **+π/2 means to the fly's anatomical left** (in
  image coordinates with y-down), **−π/2 means to the right**, **±π means behind**.

---

## 1. Per-fly features

These depend on only one fly's pose. For N flies, you get N independent time series.

### Shape and appearance

| Feature | Formula | Biological meaning |
|---|---|---|
| `ecc` | b/a (ratio of semi-minor to semi-major axis) | Body "roundness." Closer to 1 when the fly is curled or in an unusual posture; ~0.3–0.4 for a typical extended fly. |
| `area_mm` | π · a · b | Body area as an ellipse. Largely constant per fly; deviations flag tracking errors or unusual postures. |
| `da`, `db` | d(a)/dt, d(b)/dt | Rate of change of body axes. Useful for catching mounting (sudden axis-length spikes) or tracking glitches. |
| `darea` | d(area)/dt | Rate of change of body area. |
| `decc` | d(ecc)/dt | Rate of change of body roundness. |

### Position and orientation

| Feature | Formula | Biological meaning |
|---|---|---|
| `phi` | direction of motion = `arctan2(dy, dx)` | Where the fly is heading, in world coordinates. Falls back to body orientation when stationary. |
| `yaw` | `phi − theta`, wrapped to [−π, π] | Difference between motion direction and body axis. **|yaw| ≈ 0** means walking forward, **|yaw| ≈ π** means walking backward, **|yaw| ≈ π/2** means sideways (sidestepping). Critical for catching reverse walking, a Murthy-lab signature of female feedback (Roemschied 2026). |

### Translational velocity

| Feature | Formula | Biological meaning |
|---|---|---|
| `velmag_ctr` | speed of centroid | Walking speed. ~0 when stationary, 5–20 mm/s during normal walking. |
| `velmag_nose` | speed of head point | Sensitive to head-bobbing and orienting movements. |
| `velmag_tail` | speed of abdomen point | Sensitive to abdomen-curling. |
| `du_ctr` | forward velocity component in body frame | **Positive** when walking forward, **negative** when walking backward. Use this rather than `velmag_ctr` if direction matters. |
| `dv_ctr` | sideways velocity component in body frame | Sideways/lateral motion. Large during circling or sidestepping. |
| `accmag` | acceleration magnitude of centroid | Useful for catching startles, escape responses, jumps. |

### Angular velocity

| Feature | Formula | Biological meaning |
|---|---|---|
| `dtheta` | d(theta)/dt, wrap-corrected | Body angular velocity. **Positive** = counter-clockwise (in math convention, which is clockwise in image-y-down). Critical for catching circling, which is a JAABA-canonical courtship sub-behavior. |
| `d2theta` | d²(theta)/dt² | Angular acceleration. Sharp peaks indicate sudden turns. |

### Wing features

These require `lW` and `rW` keypoints to be present. With anatomical wing
labels (your convention), `wing_anglel` is negative when the left wing is
spread out and `wing_angler` is positive when the right wing is spread out.

| Feature | Formula | Biological meaning |
|---|---|---|
| `max_wing_angle` | `max(−wing_anglel, wing_angler)` | **The strongest single-wing-extension signal.** Peaks when *either* wing is spread. ~30–45° during unilateral wing extension (singing), ~5–10° at rest. **This is what JAABA's `wing_extension` classifier mostly keys on.** |
| `min_wing_angle` | `min(−wing_anglel, wing_angler)` | The *other* wing's angle. Close to `max_wing_angle` during bilateral extension (rare in courtship; common during grooming). |
| `mean_wing_angle` | (`−wing_anglel + wing_angler`)/2 | Average spread. |
| `wing_angle_diff` | `wing_angler − wing_anglel` | Signed difference. **Positive** when right wing is spread more than left; **negative** for left. Distinguishes which wing is doing the singing. |
| `wing_angle_imbalance` | `\|wing_angler + wing_anglel\|` | How asymmetric the wing posture is. **Large** when one wing is spread and the other folded (singing); **near zero** when both are equally extended or equally folded. |

### Arena features

These require `arena_radius_mm` and `arena_center_mm` to be set on the `Trx`.

| Feature | Formula | Biological meaning |
|---|---|---|
| `arena_r` | distance from arena center | Position within the arena. |
| `dist2wall` | `arena_radius − arena_r` | Distance to the nearest point on the wall. Useful for catching wall-following ("centrophobia") behavior, which is sometimes a stress indicator. |
| `angle2wall` | angle from body axis to the nearest wall point | Where the wall is in the fly's frame. **0** = wall directly ahead. |
| `dangle2wall` | d(angle2wall)/dt | How fast the wall's bearing is changing. |

---

## 2. Per-pair features

These depend on two specific flies. Each is computed in fly1's frame and time
basis: the output is a time series of length `fly1.nframes`, with NaN at
frames where fly2 doesn't exist in the chamber.

For 2 flies you have 2 directed pairs: (fly0, fly1) and (fly1, fly0). For N
flies you have N · (N−1) directed pairs. In `extract_features`, use
`other_fly_id` to restrict computation to one target.

### Distances

| Feature | Definition | Biological meaning |
|---|---|---|
| `dcenter` | centroid-to-centroid distance | Standard inter-fly distance. |
| `dnose2tail` | distance from fly1's head to fly2's abdomen | **Critical for courtship.** Small when fly1 is following fly2 from behind ("chasing"). Approaches 0 during mounting. |
| `dnose2center` | distance from fly1's head to fly2's centroid | Like `dnose2tail` but more robust if abdomen tracking is noisy. |
| `dnose2ell` | min distance from fly1's nose to fly2's body ellipse | Captures actual head-to-body contact (e.g., during tapping). |
| `dell2nose` | min distance from a point on fly1's ellipse to fly2's nose | Symmetric counterpart. |
| `dell2ell` | min distance between points on the two ellipses | True body-to-body contact distance. **Zero or near-zero during mounting.** |

### Angles

| Feature | Definition | Biological meaning |
|---|---|---|
| `anglefrom1to2` | angle in fly1's body frame to fly2's centroid | **0** = fly2 directly ahead of fly1. **+π/2** = fly2 to fly1's left. The classic "facing-the-female" feature: small \|anglefrom1to2\| means fly1 is oriented toward fly2. |
| `anglesub` | angle subtended by fly2 in fly1's visual field | Effective angular size of fly2 as seen by fly1. **Combines distance and orientation:** a perpendicular fly close-by has a large subtended angle; a head-on fly far away has a small one. Used by Coen et al. 2016 for amplitude modulation with distance. Returns 0 if fly2 is outside fly1's field of view (default 270°). |

### Relative motion

| Feature | Definition | Biological meaning |
|---|---|---|
| `magveldiff` | magnitude of velocity difference, ‖v1 − v2‖ | How fast the two flies are diverging or converging. Small during sustained chasing (they move at the same speed); large during escape. |

---

## 3. Closest-fly features

These pick out, **at each frame**, the other fly that is currently closest to
the focal fly under some distance metric, and then report features relative to
that fly. For N flies you get one time series per focal fly per feature
(scales linearly).

You can choose the distance metric via the `closest_metric` argument to
`extract_features`. Available metrics: `"center"`, `"nose2ell"`, `"ell2nose"`,
`"ell2ell"`, `"nose2tail"`, `"anglesub"`.

| Feature | Definition | Biological meaning |
|---|---|---|
| `closest_fly_id` | id of the closest other fly at each frame | Useful for understanding who the focal fly is interacting with. **Changes mid-bout** if the focal fly switches its attention from one neighbor to another. |
| `closest_fly_dist` | distance to that closest fly under the chosen metric | The "current relevant" inter-fly distance. |
| `angle_on_closest` | angle in focal fly's frame to the closest fly | Where the closest fly is, in the focal fly's view. |
| `anglesub_on_closest` | angle subtended by the closest fly | Effective angular size of whichever fly is currently closest. |
| `nflies_close(threshold)` | count of other flies within `threshold_mm` | Local density. Distinguishes "fly is in a crowd" from "fly is one-on-one with another fly." |

---

## 4. How these features map to courtship sub-behaviors

This is the practical translation layer for your ethogram. None of these are
hard rules — they're starting points for hand-tuned thresholds or for
hand-labeling examples to train a JAABA-style classifier.

### Orienting toward female

- `anglefrom1to2` (male → female): \|angle\| < ~30° (small, near zero)
- `velmag_ctr` (male): low (not yet chasing)
- `dcenter` (male → female): can be anywhere from 2–15 mm

### Chasing / following

- `anglefrom1to2` (male → female): \|angle\| < ~30°
- `velmag_ctr` (male): > 5 mm/s
- `dnose2tail` (male → female): decreasing or sustained small
- `magveldiff` (male, female): small (matched velocities)

### Singing (wing extension)

- `max_wing_angle` (male): > ~20° (well above resting baseline)
- `wing_angle_imbalance` (male): high (unilateral, not bilateral)
- `dcenter` (male → female): typically 2–8 mm (close but not contacting)
- The male is usually behind or beside the female, so `anglefrom1to2` is small.

### Tapping

- `dnose2ell` (male → female): near zero, briefly
- `velmag_ctr` (male): low or briefly spiking
- Typically follows orienting/approach.

### Attempted copulation / mounting

- `dell2ell` (male → female): near zero, sustained
- `dcenter` (male → female): < ~2 mm
- `anglefrom1to2` (male → female): small (male approaching from behind)
- `da` (male): possibly spiking briefly as the male's body axis is occluded by the female's
- **NOTE:** during the mounted-blob phase, SLEAP often fails for at least one of
  the flies, so these features become NaN. Use the YORU-style object detector
  for the mounting state itself, then use these features to characterize the
  *approach* and the *post-mounting* behavior.

### Copulation (sustained mounting > 30 s)

- `dell2ell` (male → female): near zero, sustained
- `velmag_ctr` (both flies): low
- `dcenter` (male → female): small, sustained
- Duration ≥ 30–45 s (use `MateBook`-style persistence rule)

### Female receptive vs unreceptive

- Female `velmag_ctr` low + male `dcenter` small + male singing → likely receptive
- Female `velmag_ctr` high (running away) + male `dcenter` increasing → unreceptive

---

## 5. Window features (the missing layer JAABA uses)

The per-frame features in this module are the *inputs* to JAABA's full feature
set. The actual JAABA classifier doesn't use raw per-frame values — it uses
**window features**: rolling statistics over windows of varying lengths around
each frame.

For each per-frame feature, JAABA computes ~50+ window statistics (mean, std,
min, max, median, percentiles, harmonic mean, mode, change, diff, harmonic,
spectral) over multiple window sizes (typically powers of 2: 1, 2, 4, 8, 16,
32 frames) with multiple offsets. A typical JAABA classifier ends up with
~1,000+ features per frame per fly.

This module deliberately doesn't compute the full window-feature set, because
(a) it's straightforward to do with `numpy.lib.stride_tricks.sliding_window_view`
or `pandas.DataFrame.rolling`, and (b) the choice of which window features to
include is part of classifier design, not feature engineering.

A minimal window-feature recipe in Python:

```python
import pandas as pd
import numpy as np

# df is the output of extract_features. For each fly:
fly_df = df[df["fly_id"] == 0].set_index("frame").sort_index()

# Choose a per-frame feature (e.g., max_wing_angle)
x = fly_df["max_wing_angle"]

# Compute rolling stats over multiple window sizes
for w in [3, 7, 15, 31]:
    fly_df[f"max_wing_angle__mean{w}"] = x.rolling(w, center=True).mean()
    fly_df[f"max_wing_angle__std{w}"] = x.rolling(w, center=True).std()
    fly_df[f"max_wing_angle__max{w}"] = x.rolling(w, center=True).max()
    fly_df[f"max_wing_angle__min{w}"] = x.rolling(w, center=True).min()
```

Feed the resulting wide DataFrame (per-frame + window features) to your
classifier of choice.

---

## 6. Quick-reference table: which features for which classifier

Suggested feature subsets for common JAABA-style classifiers (per the DANCE
classifier set in Yadav et al. 2025 and Murthy lab practice):

| Classifier | Core per-frame features | Pair features |
|---|---|---|
| `wing_extension` (singing) | `max_wing_angle`, `min_wing_angle`, `wing_angle_diff`, `wing_angle_imbalance` | (uses focal fly only, but `dcenter` to nearest fly is also informative) |
| `following` / `chasing` | `velmag_ctr`, `dtheta`, `yaw`, `du_ctr` | `dnose2tail`, `dcenter`, `anglefrom1to2`, `magveldiff` |
| `orienting` | `velmag_ctr`, `dtheta` | `anglefrom1to2`, `anglesub`, `dcenter` |
| `tapping` | `velmag_nose`, `velmag_ctr` | `dnose2ell`, `dnose2center`, `anglefrom1to2` |
| `attempted_copulation` | `velmag_ctr`, `du_ctr`, `ecc` | `dell2ell`, `dcenter`, `anglefrom1to2`, `dnose2tail` |
| `copulation` | `velmag_ctr`, `ecc`, `area_mm` | `dell2ell`, `dcenter`, duration ≥ 30 s |
| `circling` | `dtheta`, `dv_ctr`, `velmag_ctr` | `dcenter`, `anglefrom1to2` |

This is a starting point. The actual classifier should be trained on hand-
labeled bouts from your real DGRP videos; the features above just constrain
the input space.
