INTERPOLATE=False
FILTER_BY_CONFIDENCE=False
LONG_IMMOBILE_SECONDS=10 # seconds


DEFAULT_TRAINABLE_PARAMS={
    "persist_sec": 0.05,  # minimum touch duration, the same at every frame rate (was persist_N: 2 frames)
    "median_kernel":1,
    "merge_gap_sec": 2,
    "app_app_thresh_BL": .2
}
