import ethoscopy as etho
from functools import partial

metadata_folder='/home/vibflysleep/FlySleepLab Dropbox/Antonio/FSLLab/Projects/ethoscopy/home/vibflysleep/metadata'
meta_loc = f'{metadata_folder}/2023_flyhostel.csv' 
remote = '/flyhostel_data/videos'
local = '/flyhostel_data/videos'
flyhostel_cache='/flyhostel_data/cache'
time_window_length=10

def load_centroid_data(experiment):
    meta = etho.link_meta_index(meta_loc, remote, local, source="flyhostel")
    meta=meta.loc[meta["id"].str.startswith(experiment[:26])]
    meta["experiment"]=experiment

    data = etho.load_flyhostel(
        meta, min_time = 0, max_time = 3600*36, reference_hour = 11.0, cache = flyhostel_cache,
    )
    assert data.shape[0] > 0
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
