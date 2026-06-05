# Using `jaaba_features` on your FlyHostel xarray dataset

Your dataset has this schema:

```
Dimensions:
  time:        1691008
  individuals: 2
  keypoints:   12        # 'head', 'thorax', ..., 'lW', 'rW'
  space:       2

Variables:
  pose_tracks  (time, individuals, keypoints, space)   float64, pixels
  confidence   (time, individuals, keypoints)           float32

Attributes:
  fps : 47.0
  source_software : SLEAP
```

The 12-keypoint convention used by movement / SLEAP for *Drosophila* in the
FlyHostel pipeline typically includes head, thorax, abdomen, and the four
legs/wings. The adapter only needs three keypoints to compute the JAABA
features faithfully (head, thorax, abdomen-or-tail), plus the two wing tips
(`lW`, `rW`) if you want wing features.

## Quickstart

```python
import xarray as xr
from jaaba_features.movement_adapter import movement_to_trx
from jaaba_features.extract import extract_features

# 1) Load your dataset
ds = xr.open_dataset("FlyHostel2_2X_2026-05-12_14-00-00__01.h5",
                     engine="h5netcdf")   # or however you load it

# 2) Convert pixels -> mm. If your chamber is, say, 60 mm in diameter
#    and is 1800 px across in your video, that's 30 px/mm:
px_per_mm = 30.0

# 3) Build a Trx
trx = movement_to_trx(
    ds,
    px_per_mm=px_per_mm,
    confidence_threshold=0.5,    # NaN-mask any pose below this confidence
    smooth_window=3,             # optional: 3-frame centered moving average
    arena_radius_mm=30.0,        # for arena features (dist2wall, angle2wall)
    arena_center_mm=(30.0, 30.0),
    keypoint_map={               # explicit name mapping in case auto-detect fails
        "head":    "head",
        "thorax":  "thorax",
        "abdomen": "abdomen",    # or "tail" if that's your name
        "wing_l":  "lW",
        "wing_r":  "rW",
    },
)

# trx is now a list of FlyTrack objects (one per individual)
print(f"{len(trx.flies)} flies, fps = {1.0/trx.flies[0].dt}")
for f in trx.flies:
    print(f"  fly {f.fly_id}: frames {f.firstframe}-{f.endframe} "
          f"(nframes={f.nframes}), median body length = "
          f"{2 * float(np.nanmedian(f.a_mm)):.2f} mm")

# 4) Extract features into a pandas DataFrame
import pandas as pd
df = extract_features(
    trx,
    include_pair=True,           # adds pairwise features (dcenter, anglesub, ...)
    include_closest=True,        # adds closest-fly features
    include_wing=True,           # adds wing-angle features
    include_arena=True,          # adds arena features (since you set arena_radius_mm)
    closest_metric="center",     # or "ell2ell", "nose2ell", ...
    other_fly_id=1,              # if 1M:1F, compute pair features vs the other fly explicitly
)
print(df.shape)                  # (n_flies * n_frames, n_features + 2)
print(df.columns.tolist())
```

## Specific to your dataset (2 individuals per chamber)

You said you ultimately want 2M:2F chambers; this dataset has 2 individuals,
which is one M-F pair. For that case:

```python
# Suppose individual 0 is the male and individual 1 is the female.
# The "directed" features for the male relative to the female are what JAABA
# uses for courtship classifiers.

male_df   = df[df["fly_id"] == 0].reset_index(drop=True)
female_df = df[df["fly_id"] == 1].reset_index(drop=True)

# male_df["dcenter"]      -> male's distance to female
# male_df["anglefrom1to2"] -> male's bearing to female (0 = ahead, +pi/2 = left)
# male_df["anglesub"]     -> female's angular size in male's visual field
# male_df["max_wing_angle"] -> max(|left wing|, right wing) -- larger when singing
```

For 2M:2F you'll want directed features for every (M_i, F_j) ordered pair (4
directed pairs total). Loop over the pair-feature columns with `other_fly_id`
set explicitly for each combination:

```python
all_pair_dfs = []
for male_id in [0, 1]:        # assuming flies 0,1 are males
    for female_id in [2, 3]:  # assuming flies 2,3 are females
        # Build a per-(male, female) DataFrame by extracting only the male's
        # features against the chosen female.
        df_pair = extract_features(
            trx, other_fly_id=female_id, include_closest=False,
        )
        # Keep only rows where fly_id == male_id
        df_pair = df_pair[df_pair["fly_id"] == male_id].copy()
        df_pair["female_id"] = female_id
        all_pair_dfs.append(df_pair)

all_pairs = pd.concat(all_pair_dfs, ignore_index=True)
# now you have, for each frame, four directed (male, female) feature rows
```

## Memory and chunking

Your dataset is 1.69M frames × 2 flies = 3.4M rows. With ~30 features that's
~100M floats ≈ 800 MB in pandas. If that's too much:

```python
# extract one fly at a time and stream to disk:
import pyarrow.parquet as pq
for fly_id in trx.fly_ids:
    df_one = extract_features(trx, ...)  # then filter to fly_id
    df_one = df_one[df_one["fly_id"] == fly_id]
    df_one.to_parquet(f"features_fly{fly_id}.parquet")
```

Or chunk the dataset by frame range before passing to the adapter (open the
xarray dataset with `chunks={"time": 200_000}` if you use `xarray + dask`).

## Calibrating `px_per_mm`

If you don't know the calibration off the top of your head:

```python
# pick a reference frame and a calibration object (e.g. the chamber wall)
import matplotlib.pyplot as plt
import numpy as np
# overlay the pose on a frame
# measure chamber diameter in pixels and divide by known diameter in mm
```

You can also estimate from the typical Drosophila body length (~2.5 mm). The
median head-to-abdomen distance for a given individual should be ~2.5 mm; if
you see ~75 px instead, `px_per_mm = 30`.

## Sanity checks after extraction

```python
# 1) velocity should be < ~50 mm/s for a fly (walking ~10 mm/s, flying ~100)
assert df["velmag_ctr"].quantile(0.99) < 50

# 2) distances should be in mm, not pixels
assert df["dcenter"].max() < 100  # chamber is ~60 mm; >100 mm impossible

# 3) angles should be in [-pi, pi]
assert df["anglefrom1to2"].abs().max() <= np.pi + 1e-6
```

## Adjusting for your wing-angle sign convention

`movement_adapter` assumes image coordinates (y down) and flips the sign on
wing angles so that JAABA's convention holds (`wing_anglel < 0` when spread).
If your `lW` is *not* the fly's anatomical left wing (e.g. if SLEAP labels
"left" by image side rather than by fly anatomy), you may need to swap
`keypoint_map["wing_l"]` and `["wing_r"]`. A quick check:

```python
# When the male is courting (spreading one wing toward the female):
#   max_wing_angle should peak at ~30-45 degrees
#   wing_angle_diff should be highly variable
# If you see max_wing_angle peaking at near-zero when courtship is obvious,
# your wing-side labels are swapped.
```

## Known limitations

- **No center-of-rotation features.** `velmag` falls back to `velmag_ctr`.
  Re-enable if you need them by porting JAABA's `center_of_rotation*.m`.
- **No spacetime (HOG/HOF) features.** Requires the raw video frames.
- **APT features not ported.** You're using SLEAP, so this is moot.
- **No JAABA window-feature computation here.** JAABA's classifier uses
  thousands of window features (means, std-devs, percentiles over multiple
  window sizes around each frame). Compute these downstream with
  `scipy.signal` or `numpy.lib.stride_tricks.sliding_window_view`. The
  per-frame features in this module are the inputs to that window-feature
  computation.
