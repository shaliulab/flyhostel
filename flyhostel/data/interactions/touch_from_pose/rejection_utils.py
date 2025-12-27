import glob
import logging

import numpy as np
import xarray as xr
import pandas as pd

from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.pose.ethogram.utils import annotate_bouts, annotate_bout_duration


logger=logging.getLogger(__name__)


def distance_per_second(disp: xr.DataArray) -> xr.DataArray:
    """
    disp: DataArray with dims (time, space, keypoints, individuals)
          and space = ['x', 'y'], containing frame-to-frame displacements.
    time: coordinate in seconds from start of recording.
    
    Returns:
        DataArray with dims (sec, keypoints, individuals) where `sec`
        is the integer second (0,1,2,...) and values are the total
        distance travelled in that second.
    """
    # 1) distance per frame: sqrt(dx^2 + dy^2)
    dx = disp.sel(space="x")
    dy = disp.sel(space="y")
    step_dist = np.hypot(dx, dy)   # dims: (time, keypoints, individuals)

    # 2) create integer-second coordinate for binning
    sec = np.floor(step_dist["time"]).astype("int64")
    step_dist = step_dist.assign_coords(sec=("time", sec.data))

    # 3) group by second and sum distances within each second
    dist_per_sec = step_dist.groupby("sec").sum(dim="time")

    return dist_per_sec

def add_animal_state(dist_per_sec: xr.DataArray,
                    state_df: pd.DataFrame,
                    time_col: str = "t",
                    animal_col: str = "animal",
                    state_col: str = "asleep",
                    state_name: str = None,
                    alignment_offset: int = 0,
                    ) -> xr.DataArray:
    """
    Attach a boolean `asleep` coordinate to dist_per_sec based on a
    DataFrame with columns [t, animal, asleep].

    Parameters
    ----------
    dist_per_sec : xr.DataArray
        Output of distance_per_second, dims: (sec, keypoints, individuals)
    state_df : pd.DataFrame
        Columns: time_col (seconds), animal_col (matches `individuals`),
        state_col (0/1 or False/True).
    time_col, animal_col, state_col : str
        Column names for time, animal ID and sleep state.

    Returns
    -------
    xr.DataArray
        Same as dist_per_sec, but with an `asleep` coordinate of
        dims (sec, individuals), broadcast over keypoints.
    """

    # 1) Convert time to integer seconds to match `sec` dim
    df = state_df.copy()
    df["sec"] = np.floor(df[time_col]).astype("int64")

    # 2) Keep only seconds present in dist_per_sec
    valid_secs = set(dist_per_sec["sec"].values.tolist())
    df = df[df["sec"].isin(valid_secs)]

    # 3) Build a Series indexed by (sec, individuals) and convert to xarray
    s = (
        df
        .rename(columns={animal_col: "individuals"})
        .set_index(["sec", "individuals"])[state_col]
    )
    state_da = s.to_xarray()  # dims: (sec, individuals)

    # Shift the *coordinates* of state_da so that
    # value at sec = t+offset is now labeled as sec = t
    if alignment_offset is not None and alignment_offset != 0:
        state_da = state_da.assign_coords(sec=state_da["sec"] - alignment_offset)
        
    # 4) Align with dist_per_sec (broadcast over keypoints)
    #    This will reindex state_da to match dist_per_sec's sec/individuals
    state_da, dist_per_sec_aligned = xr.align(state_da, dist_per_sec, join="right")


    if state_name is None:
        state_name=state_col

    # 5) Attach as a coordinate (dims: sec, individuals, broadcast over keypoints)
    dist_with_state = dist_per_sec_aligned.assign_coords({state_name: state_da})

    return dist_with_state