# JAABA per-frame feature port — coverage map

This document maps every `.m` file in your `code.m` dump to its Python
equivalent (or to the reason it was deferred).

## Single-fly features (`jaaba_features/single_fly.py`)

| MATLAB file | Python function | Notes |
|---|---|---|
| `compute_ecc.m` | `ecc` | b/a ratio |
| `compute_area.m` / `compute_area_mm.m` | `area` / `area_mm` | ellipse area approximation |
| `compute_xnose_mm.m` / `compute_ynose_mm.m` | `xnose_mm` / `ynose_mm` | |
| `compute_phi.m` | `phi` | direction of motion, with theta fallback |
| `compute_smooththeta.m` | `smooththeta` | smoothing with a triangular filter; in MATLAB the filter is `trx.perframe_params.thetafil` |
| `compute_velmag_ctr.m` | `velmag_ctr` | |
| `compute_velmag_nose.m` | `velmag_nose` | |
| `compute_velmag_tail.m` | `velmag_tail` | |
| `compute_velmag.m` | `velmag` | falls back to `velmag_ctr` (skips COR computation) |
| `compute_accmag.m` | `accmag` | |
| `compute_veltoward.m` | `veltoward` | takes a second fly arg |
| `compute_dtheta.m` | `dtheta` | |
| `compute_d2theta.m` | `d2theta` | |
| `compute_smoothdtheta.m` | `smoothdtheta` | |
| `compute_smoothd2theta.m` | `smoothd2theta` | |
| `compute_dtheta_tail.m` | `dtheta_tail` | alias for `dtheta` (JAABA does not actually differ) |
| `compute_signdtheta.m` | `signdtheta` | |
| `compute_da.m` / `compute_db.m` / `compute_darea.m` / `compute_decc.m` | `da` / `db` / `darea` / `decc` | finite differences |
| `compute_du_ctr.m` / `compute_dv_ctr.m` / `compute_du_tail.m` / `compute_dv_tail.m` | `du_ctr` / `dv_ctr` / `du_tail` / `dv_tail` | body-frame velocity components |
| `compute_yaw.m` | `yaw` | |
| `compute_phisideways.m` | `phisideways` | |
| `compute_arena_r.m` / `compute_arena_angle.m` | `arena_r` / `arena_angle` | |
| `compute_dist2wall.m` | `dist2wall` | circular arena |
| `compute_angle2wall.m` | `angle2wall` | |
| `compute_dangle2wall.m` | `dangle2wall` | |

## All `compute_abs*` and `compute_d*` wrappers

JAABA has 23 `compute_abs*.m` files and 9 simple `compute_d*.m` files that are
one-liner wrappers around existing features. Instead of porting one Python
function per file, use the combinators:

```python
from jaaba_features.single_fly import absfeat, dfeat
absfeat(velmag_ctr(fly)[0])             # = compute_absvelmag_ctr
dfeat(dtheta(fly)[0], dt=fly.dt)        # = compute_d_dtheta (== compute_d2theta)
```

The `compute_abs__template.m` file is literally a template that does
`abs(trx(fly).(pFeatureName))`, so `absfeat` covers them all.

For derivatives of angular features (where you need wrap-around), use
`angular_diff` from `trx.py`, not `dfeat`.

## Wing features (`jaaba_features/wing_features.py`)

| MATLAB file | Python function |
|---|---|
| `compute_max_wing_angle.m` | `max_wing_angle` |
| `compute_min_wing_angle.m` | `min_wing_angle` |
| `compute_mean_wing_angle.m` | `mean_wing_angle` |
| `compute_wing_angle_diff.m` | `wing_angle_diff` |
| `compute_wing_angle_imbalance.m` | `wing_angle_imbalance` |
| `compute_max_wing_length.m` | `max_wing_length` |
| `compute_min_wing_length.m` | `min_wing_length` |
| `compute_mean_wing_length.m` | `mean_wing_length` |
| `compute_max_wing_area.m` | `max_wing_area` |
| `compute_min_wing_area.m` | `min_wing_area` |
| `compute_mean_wing_area.m` | `mean_wing_area` |
| `compute_length_inmost_wing.m` | `length_inmost_wing` |
| `compute_length_outmost_wing.m` | `length_outmost_wing` |
| `compute_area_inmost_wing.m` | `area_inmost_wing` |
| `compute_area_outmost_wing.m` | `area_outmost_wing` |
| `compute_angle_biggest_wing.m` | `angle_biggest_wing` |
| `compute_angle_smallest_wing.m` | `angle_smallest_wing` |
| `compute_dwing_angle_diff.m` | `dwing_angle_diff` |
| `compute_dmean_wing_angle.m` | `dmean_wing_angle` |
| `compute_dmax_wing_angle.m` | `dmax_wing_angle` |
| `compute_dmin_wing_angle.m` | `dmin_wing_angle` |
| `compute_dmax_wing_length.m` | `dmax_wing_length` |
| `compute_dmin_wing_length.m` | `dmin_wing_length` |
| `compute_dmean_wing_area.m` | `dmean_wing_area` |
| `compute_wing_areal_mm.m` / `compute_wing_arear_mm.m` | (just `fly.wing_areal_mm` / `fly.wing_arear_mm`) |
| `compute_wing_lengthl_mm.m` / `compute_wing_lengthr_mm.m` | (just `fly.wing_lengthl_mm` / `fly.wing_lengthr_mm`) |
| `compute_angle_longest_wing.m` / `compute_angle_shortest_wing.m` | identical to `angle_biggest_wing` / `angle_smallest_wing` (length-based instead of area-based — trivial swap) |
| `compute_max_absdwing_*.m` / `compute_min_absdwing_*.m` / `compute_min_dwing_angle_in.m` / etc | window stats / signed extrema; build with `absfeat`+`dfeat` combinators |
| `compute_dnwingsdetected.m` | not ported (requires a per-frame "number of wings detected" flag from the wing tracker) |

## Pair features (`jaaba_features/pair_features.py`)

| MATLAB file | Python function | Notes |
|---|---|---|
| `dcenter_pair.m` | `dcenter_pair` | centroid distance |
| `dnose2tail_pair.m` | `dnose2tail_pair` | |
| `dnose2center_pair.m` | `dnose2center_pair` | |
| `dcenter2nose_pair.m` (in some JAABA versions) | `dcenter2nose_pair` | |
| `dnose2ell_pair.m` | `dnose2ell_pair` | 20-sample ellipse approximation |
| `dell2nose_pair.m` | `dell2nose_pair` | |
| `dell2ell_pair.m` + `ellipse2ellipsedist_hack.m` | `dell2ell_pair` | combined into one |
| `anglesub_pair.m` | `anglesub_pair` | |
| `anglesubtended.m` + helpers (`eyeoffly1givenfly2`, `checkinborder`, `computetangentpoints`, `limitbyfov`) | `anglesubtended` (private helpers `_eye_of_fly1_given_fly2`, `_check_in_border`, `_compute_tangent_points`, `_limit_by_fov`) | exact port; geometric content preserved |
| `magveldiff_pair.m` (implicit in JAABA, via compute_magveldiff.m) | `magveldiff_pair` | |
| `anglefrom1to2` pair version | `anglefrom1to2_pair` | |
| `compute_anglesub.m` | use `anglesub_pair` + `closest_fly` from `closest.py` |
| `dnose2ell_anglerange_pair.m` | not ported (specialized; same idea as `dnose2ell_pair` but restricted to an angular range) |
| `dapt_pair.m` / `dapt2ctr_pair.m` / `dapt2ell_pair.m` | not ported (APT-specific) |
| `dell2nose_bounds.m` / `dnose2ell_bounds.m` / `dell2ell_bounds.m` | not ported (precomputed bounding boxes for early-exit; pure performance optimization, not a different feature) |
| `malah_dist.m` | not ported (Mahalanobis distance helper; only used in some special-case computations) |
| `isclose_pair.m` | use `dcenter_pair(fly1, fly2) < threshold` |

## Closest-fly features (`jaaba_features/closest.py`)

| MATLAB file | Python function |
|---|---|
| `compute_closestfly_center.m` | `closestfly_center` / `closest_fly(trx, fid, dcenter_pair)` |
| `compute_closestfly_nose2ell.m` | `closestfly_nose2ell` |
| `compute_closestfly_ell2nose.m` | `closestfly_ell2nose` |
| `compute_closestfly_ell2ell.m` | `closestfly_ell2ell` |
| `compute_closestfly_nose2tail.m` | `closestfly_nose2tail` |
| `compute_closestfly_anglesub.m` | `closestfly_anglesub` |
| `compute_dcenter.m` / `compute_dnose2ell.m` / `compute_dell2nose.m` / etc | the *distance to closest fly* is returned as the second tuple element by `closest_fly(...)` |
| `compute_angleonclosestfly.m` | `angle_on_closest_fly` |
| `compute_absangleonclosestfly.m` | `absfeat(angle_on_closest_fly(...))` |
| `compute_anglesub.m` (closest version) | `anglesub_on_closest_fly` |
| `compute_nflies_close.m` | `nflies_close` |
| `compute_dnose2ell_anglerange.m` / `compute_closestfly_nose2ell_anglerange.m` | not ported (specialized angular-range variants) |
| `compute_closestfly_apt2ctr.m` / `compute_closestfly_apt2ell.m` | not ported (APT-specific) |
| `compute_ddcenter.m` / `compute_ddell2nose.m` / `compute_ddnose2ell.m` | use `dfeat(...)` on the distance series |

## Center-of-rotation features

These are explicitly NOT ported. JAABA's center-of-rotation computation
(`center_of_rotation.m`, `center_of_rotation2.m`, `compute_corfrac_maj.m`,
`compute_corfrac_min.m`, `compute_du_cor.m`, `compute_dv_cor.m`,
`compute_corisonfly.m`, `compute_flipdv_cor.m`, `compute_absdv_cor.m`,
`compute_abscorfrac_min.m`) is rarely used by published JAABA courtship
classifiers and adds significant complexity. The `velmag` Python function falls
back to `velmag_ctr` as a substitute. If you find you actually need the COR
features, port the two helpers and the four `du_cor`/`dv_cor`/`corfrac_*`
features following the same template as the centroid-based ones.

## Spacetime (HOG/HOF) features

`compute_spacetime.m`, `compute_spacetime_gradient.m`,
`compute_spacetime_hoghof.m`, `compute_spacetime_mask.m`,
`compute_spacetime_transform.m` — NOT ported. These are image-based features
that require the raw video, and they call out to external HOG/HOF
implementations. If you need image features in your pipeline, use a recent
video-based classifier (e.g., a 3D CNN or video transformer) rather than
porting JAABA's HOG/HOF, which is a 2010-era pipeline.

## Rectangular-arena features

`compute_dist2wall_rect.m`, `compute_ddist2wall_rect.m`,
`compute_distnose2wall_rect.m`, `compute_distnose2wall_animaldir_rect.m`,
`compute_angle2wall_rect.m`, `compute_angle2corner_rect.m`,
`compute_dist2corner_rect.m`, `compute_dangle2*_rect.m`, etc — NOT ported.
You have a circular arena; the rectangular variants are easy to add following
the same template as `dist2wall` if you ever need them.

## ROI-2 features

`compute_angle2closestroi2.m`, `compute_dist2roi2.m`, `compute_mindist2roi2.m`,
`compute_dangle2closestroi2.m`, `compute_dmindist2roi2.m`,
`compute_closestroi2.m`, `compute_angle2roi2.m` — NOT ported. These compute
features relative to a secondary ROI in the chamber (e.g., a food patch).
Easy to add if your DGRP assay has such an ROI.

## APT (Animal Part Tracker) features

`compute_apt.m`, `compute_apt_distclosest.m`, `compute_apt_social.m`,
`dapt_pair.m`, `dapt2ctr_pair.m`, `dapt2ell_pair.m`,
`compute_closestfly_apt2ctr.m`, `compute_closestfly_apt2ell.m` — NOT ported.
These require APT-style keypoint output with a specific schema. Since you use
SLEAP, you'd implement equivalent keypoint-relative features directly on your
SLEAP keypoints rather than going through this layer.

## Smoothing / helper functions

| MATLAB file | Python equivalent |
|---|---|
| `modrange` (used everywhere) | `modrange` in `trx.py` |
| `SmoothAreaOutliers.m` | not ported; do this in a preprocessing step (e.g., `scipy.signal.medfilt`) |
| `LowPassFilterArea.m` | not ported; use `scipy.signal.butter` + `filtfilt` |
| `parseunits` | not ported; we return units as plain strings |
