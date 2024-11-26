import logging
import time
from functools import partial
import tempfile

import pandas as pd
import numpy as np
from ethoscopy import link_meta_index, load_flyhostel, behavpy, flyhostel_sleep_annotation
from ethoscopy import flyhostel_sleep_annotation as flyhostel_sleep_annotation_primitive
from ethoscopy.flyhostel import compute_xy_dist_log10x1000
from ethoscopy.load import update_metadata

logger=logging.getLogger(__name__)


metadata_folder='/flyhostel_data/metadata'
remote = '/flyhostel_data/videos'
local = '/flyhostel_data/videos'
flyhostel_cache='/flyhostel_data/cache'
time_window_length=10

def load_centroid_data(metadata=None, experiment=None, identity=None, min_time=None, max_time=None, time_system="zt", n_jobs=1, verbose=False, reference_hour=np.nan, **kwargs):
    
    meta_loc=tempfile.NamedTemporaryFile(suffix=".csv", prefix="flyhostel").name

    if metadata is None:
        assert experiment is not None
        update_metadata(meta_loc)
        meta = link_meta_index(meta_loc, remote, local, source="flyhostel", verbose=verbose)
        meta=meta.loc[meta["id"].str.startswith(experiment[:26])]
        if identity is not None:
            meta=meta.loc[meta["identity"]==str(int(identity))]
            n_after=meta.shape[0]
            if n_after==0:
                logger.warning("No metadata for experiment %s and identity %s", experiment, identity)
            if n_after>1:
                raise Exception("> 1 animals matches experiment %s and identity %s", experiment, identity)

        if meta.shape[0]==0:
            logger.warning("Experiment %s and identity %s not found in %s", experiment, identity, meta_loc)
            return None, {}

        meta["experiment"]=experiment
        meta=pd.DataFrame(meta)

    else:
        assert all([field in metadata.columns for field in ["flyhostel_date", "flyhostel_number", "number_of_animals"]])
        metadata["date"] = metadata["flyhostel_date"]
        metadata["input_date"]=metadata["date"]
        # machine_id = FlyHostelX
        metadata["machine_id"] = [f"FlyHostel{row['flyhostel_number']}" for _, row in metadata.iterrows()]
        # machine_name "NX"
        metadata["machine_name"] = [f"{row['number_of_animals']}X" for _, row in metadata.iterrows()]
        
        metadata.to_csv(meta_loc, index=None)
        
        meta = link_meta_index(meta_loc, remote, local, source="flyhostel", verbose=verbose)
        assert meta.shape[0]>0, f"Experiment {experiment} and identity {identity} not found in {meta_loc}"

    data, meta_info = load_flyhostel(
        meta, min_time = min_time, max_time = max_time, cache = flyhostel_cache, n_jobs=n_jobs,
        time_system=time_system, reference_hour=reference_hour, **kwargs
    )

    assert data.shape[0] > 0, "No data found!"
    if data.shape[0] == 1:
        logger.warning("Either cache or flyhostel.db of %s %s are corrupted. Ignoring", experiment, identity)
        return None, {}
    
    if identity is None:
        before=time.time()
        data.sort_values(["id", "t"], inplace=True)
        after=time.time()
        logger.debug("Sorting centroid data took %s seconds", after-before)
    
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
