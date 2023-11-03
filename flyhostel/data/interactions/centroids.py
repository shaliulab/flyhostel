import ethoscopy as etho
from functools import partial
from ethoscopy.flyhostel import compute_xy_dist_log10x1000
import numpy as np

metadata_folder='/flyhostel_data/metadata'
meta_loc = f'{metadata_folder}/flyhostel.csv'
remote = '/flyhostel_data/videos'
local = '/flyhostel_data/videos'
flyhostel_cache='/flyhostel_data/cache'
time_window_length=10

def load_centroid_data(experiment, min_time=-float('inf'), max_time=+float('inf'), time_system="zt", n_jobs=20):
    meta = etho.link_meta_index(meta_loc, remote, local, source="flyhostel")
    meta=meta.loc[meta["id"].str.startswith(experiment[:26])]
    assert meta.shape[0]>0, f"Experiment not found in {meta_loc}"
    meta["experiment"]=experiment

    data = etho.load_flyhostel(
        meta, min_time = min_time, max_time = max_time, reference_hour = np.nan, cache = flyhostel_cache, n_jobs=n_jobs,
        time_system=time_system
    )

    assert data.shape[0] > 0, "No data found!"
    data.sort_values(["id", "t"], inplace=True)
    dt = etho.behavpy(data = data, meta = meta, check = True)
    return dt

def to_behavpy(*args, **kwargs):
    return etho.behavpy(*args, **kwargs)

sleep_annotation = partial(
    etho.sleep_annotation,
    time_window_length = time_window_length,
    min_time_immobile = 300,
    velocity_correction_coef=0.0048,
    optional_columns=["has_interacted", "frame_number"]
)


def downsample_centroid_data(dt, framerate):
    dt=dt.loc[(dt["frame_number"] % framerate) == 0]
    dt["xy_dist_log10x1000"]=compute_xy_dist_log10x1000(dt, min_distance=1/1000)
    return dt
