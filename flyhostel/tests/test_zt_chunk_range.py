"""
Selection of the chunks to validate from a ZT interval (integrate-human-annotations --zt0 --start-zt --end-zt)
It must match the idtrackerai pipeline (zt_chunk_range.py), which calls the same function
"""
import os

import numpy as np
import pytest

from flyhostel.tests import synthetic_experiment as synthetic
from flyhostel.utils.utils import get_chunk_range_from_zt, get_zt0_ms

# the folder name says the experiment starts at 14:00
ZT0 = "11:00"
START_ZT = 3


def chunk_zt(basedir):
    """ZT (h) of the first and last frame of every recorded chunk, with ZT0 at 11:00"""
    out = {}
    for chunk in synthetic.RECORDED_CHUNKS:
        frame_time = np.load(os.path.join(basedir, f"{chunk:06d}.npz"))["frame_time"]
        out[chunk] = (START_ZT + frame_time[0] / 3600 / 1000, START_ZT + frame_time[-1] / 3600 / 1000)
    return out


def test_zt0_is_last_lights_on_before_start(flyhostel_videos):
    assert get_zt0_ms(flyhostel_videos, "11:00") == -3 * 3600 * 1000
    assert get_zt0_ms(flyhostel_videos, 11) == -3 * 3600 * 1000
    assert get_zt0_ms(flyhostel_videos, "11:30:00") == -2.5 * 3600 * 1000
    # lights on after the start: ZT0 is on the day before
    assert get_zt0_ms(flyhostel_videos, "20:00") == -18 * 3600 * 1000


def test_open_interval_is_whole_experiment(flyhostel_videos):
    chunks = list(chunk_zt(flyhostel_videos))
    assert get_chunk_range_from_zt(flyhostel_videos, ZT0) == (chunks[0], chunks[-1])


def test_interval_selects_overlapping_chunks(flyhostel_videos):
    chunks = chunk_zt(flyhostel_videos)
    first, second = list(chunks)[:2]

    # an interval inside the second chunk selects only that chunk
    middle = sum(chunks[second]) / 2
    assert get_chunk_range_from_zt(flyhostel_videos, ZT0, start_zt=middle, end_zt=middle + 1e-9) == (second, second)

    # starting after the end of the first chunk skips it
    start, _ = get_chunk_range_from_zt(flyhostel_videos, ZT0, start_zt=chunks[first][1] + 1e-9)
    assert start == second

    # ending at the start of the second chunk leaves it out
    _, end = get_chunk_range_from_zt(flyhostel_videos, ZT0, end_zt=chunks[second][0])
    assert end == first


def test_interval_outside_experiment(flyhostel_videos):
    with pytest.raises(ValueError):
        get_chunk_range_from_zt(flyhostel_videos, ZT0, start_zt=100)
    with pytest.raises(ValueError):
        get_chunk_range_from_zt(flyhostel_videos, ZT0, end_zt=START_ZT)


def test_cli_frame_interval(flyhostel_videos):
    from flyhostel.data.bin.human_validation.integrate import get_parser, get_frame_interval
    chunks = chunk_zt(flyhostel_videos)
    second = list(chunks)[1]
    middle = sum(chunks[second]) / 2

    def parse(*argv):
        return get_parser().parse_args(["--experiment", synthetic.EXPERIMENT, *argv])

    zt_args = ["--start-zt", str(middle), "--end-zt", str(middle + 1e-6)]
    expected = (second * synthetic.CHUNKSIZE, (second + 1) * synthetic.CHUNKSIZE)
    assert get_frame_interval(parse("--zt0", ZT0, *zt_args)) == expected

    # without --zt0, the reference_hour of the metadata (11) is used
    assert synthetic.REFERENCE_HOUR == 11
    assert get_frame_interval(parse(*zt_args)) == expected

    assert get_frame_interval(parse("--fn-interval", "0", "10")) == [0, 10]

    with pytest.raises(ValueError):
        get_frame_interval(parse("--fn-interval", "0", "10", "--start-zt", "2"))


@pytest.mark.parametrize("created, first_frame_min, expected", [
    # the recording starts some minutes after the time in the folder name, and frame_time counts from the folder time
    ("2026-01-01 14:54:15", 54.25, "2026-01-01 14:00:00"),
    # even if the recording is created more than one hour later (the old rule, created_utc rounded down, gave 15:00)
    ("2026-01-01 15:00:41", 60.68, "2026-01-01 14:00:00"),
    ("2026-01-01 16:39:38", 159.6, "2026-01-01 14:00:00"),
    # renamed folder: the folder name does not match the recording, use created_utc rounded down
    ("2025-06-28 16:36:31", 36.52, "2025-06-28 16:00:00"),
])
def test_frame_time_origin(flyhostel_videos, monkeypatch, created, first_frame_min, expected):
    import datetime
    from flyhostel.utils import utils
    created = datetime.datetime.fromisoformat(created)
    monkeypatch.setattr(utils, "_load_recording_start", lambda *args, **kwargs: (created, first_frame_min * 60000))
    assert utils.get_frame_time_origin(flyhostel_videos) == datetime.datetime.fromisoformat(expected)


def test_dbfile_and_imgstore_agree(flyhostel_videos):
    """The pipeline (no dbfile yet) and the loaders (dbfile) must find the same ZT"""
    from flyhostel.utils import utils
    dbfile = synthetic.dbfile_path(flyhostel_videos)
    assert utils._load_recording_start(flyhostel_videos) == utils._load_recording_start(flyhostel_videos, dbfile=dbfile)
    assert get_zt0_ms(flyhostel_videos, ZT0) == get_zt0_ms(flyhostel_videos, ZT0, dbfile=dbfile)
    assert utils.load_meta_info(dbfile, 1)["t_after_ref"] == -get_zt0_ms(flyhostel_videos, ZT0) / 1000


def test_zt_from_chunk_matches_chunk_selection(flyhostel_videos):
    from flyhostel.utils.utils import get_zt_from_chunk
    chunks = chunk_zt(flyhostel_videos)
    second = list(chunks)[1]
    zt = get_zt_from_chunk(synthetic.EXPERIMENT, second)
    assert zt == round(chunks[second][0], 2)
