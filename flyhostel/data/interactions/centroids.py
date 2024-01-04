import logging
from functools import partial

from ethoscopy import link_meta_index, load_flyhostel, behavpy, flyhostel_sleep_annotation
from ethoscopy import flyhostel_sleep_annotation as flyhostel_sleep_annotation_primitive
from ethoscopy.flyhostel import compute_xy_dist_log10x1000

logger=logging.getLogger(__name__)


metadata_folder='/flyhostel_data/metadata'
meta_loc = f'{metadata_folder}/flyhostel.csv'
remote = '/flyhostel_data/videos'
local = '/flyhostel_data/videos'
flyhostel_cache='/flyhostel_data/cache'
time_window_length=10

def load_centroid_data(metadata=None, experiment=None, min_time=-float('inf'), max_time=+float('inf'), time_system="zt", n_jobs=20, verbose=False, **kwargs):
    if metadata is None:
        assert experiment is not None
        meta = link_meta_index(meta_loc, remote, local, source="flyhostel", verbose=verbose)
        meta=meta.loc[meta["id"].str.startswith(experiment[:26])]
        assert meta.shape[0]>0, f"Experiment not found in {meta_loc}"
        meta["experiment"]=experiment
    else:
        metadata["date"] = metadata["flyhostel_date"]
        metadata["machine_id"] = [f"FlyHostel{row['flyhostel_number']}" for _, row in metadata.iterrows()]
        count = metadata.groupby(["flyhostel_number", "flyhostel_date"]).count().iloc[:,:1].reset_index()
        count.columns=["flyhostel_number", "flyhostel_date", "number_of_animals"]
        metadata=metadata.merge(count, on=["flyhostel_number", "flyhostel_date"])
        metadata["machine_name"] = [f"{row['number_of_animals']}X" for _, row in metadata.iterrows()]
        metadata["input_date"]=metadata["date"]
        metadata.to_csv(meta_loc, index=None)
        
        meta = link_meta_index(meta_loc, remote, local, source="flyhostel", verbose=verbose)
        assert meta.shape[0]>0, f"Experiment not found in {meta_loc}"

    data, meta_info = load_flyhostel(
        meta, min_time = min_time, max_time = max_time, cache = flyhostel_cache, n_jobs=n_jobs,
        time_system=time_system, **kwargs
    )

    assert data.shape[0] > 0, "No data found!"
    data.sort_values(["id", "t"], inplace=True)
    dt = behavpy(data = data, meta = meta, check = True)
    return dt, meta_info

def to_behavpy(*args, **kwargs):
    return behavpy(*args, **kwargs)

flyhostel_sleep_annotation = partial(
    flyhostel_sleep_annotation_primitive,
    time_window_length = time_window_length,
    min_time_immobile = 300,
    velocity_correction_coef=0.0048,
    optional_columns=["has_interacted", "frame_number"]
)


def downsample_centroid_data(dt, framerate):
    dt=dt.loc[(dt["frame_number"] % framerate) == 0]
    dt["xy_dist_log10x1000"]=compute_xy_dist_log10x1000(dt, min_distance=1/1000)
    return dt