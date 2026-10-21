"""
Tests for loading the data sources of a FlyHostel experiment (FlyHostelLoader, flyhostel.utils, ethoscopy):
experiment properties and metadata, centroids, the store index, compiled pose and identity tables.

Everything is loaded from the synthetic experiment (synthetic_experiment.py), where the true position
and timing of every fly is known.
"""
import os
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

import flyhostel.data.pose.loaders.centroids as centroids
import flyhostel.utils.cvat as cvat
from flyhostel.data.pose.constants import bodyparts as BODYPARTS
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.tests import synthetic_experiment as synthetic
from flyhostel.tests.test_pose_compile import code, compiled_path, predictions_path, write_predictions
from flyhostel.utils import (
    get_chunksize,
    get_first_frame,
    get_framerate,
    get_last_frame,
    get_local_identities,
    get_number_of_animals,
    get_pixels_per_mm,
    get_square_width,
    get_wavelet_downsample,
    load_meta_info,
    load_metadata,
)
from flyhostel.utils.pose_export import recreate_pose_file


def frame_time_s(frame_numbers):
    return np.array([synthetic.frame_time(fn) for fn in frame_numbers]) / 1000


@pytest.fixture
def no_cvat(monkeypatch):
    """Fail if validation status is looked up in CVAT (network)"""
    def get_project_id_from_name(*args, **kwargs):
        raise AssertionError("CVAT should not be queried")
    monkeypatch.setattr(cvat, "get_project_id_from_name", get_project_id_from_name)


@pytest.fixture
def centroid_source(tmp_path, monkeypatch):
    """
    Point the centroid loader (ethoscopy) to a FLYHOSTEL_VIDEOS folder and a metadata sheet.
    Returns a function that takes that folder and optionally the reference hour of the sheet
    """
    def configure(videos, reference_hour=synthetic.REFERENCE_HOUR):
        header, rows = synthetic.ethoscope_metadata_rows()
        rows = [row[:-3] + [str(reference_hour)] + row[-2:] for row in rows[:2]] + rows[2:]
        sheet = tmp_path / "metadata_sheet.csv"
        # the sheet has no index column
        sheet.write_text("\n".join(",".join(row[1:]) for row in [header] + rows) + "\n")
        monkeypatch.setattr(centroids, "update_metadata", lambda path: shutil.copy(sheet, path))
        monkeypatch.setattr(centroids, "remote", videos)
        monkeypatch.setattr(centroids, "local", videos)
        cache = tmp_path / "cache"
        cache.mkdir(exist_ok=True)
        monkeypatch.setattr(centroids, "flyhostel_cache", str(cache))
    return configure


@pytest.fixture
def loader(flyhostel_videos, no_cvat):
    return FlyHostelLoader(synthetic.EXPERIMENT, identity=1)


def test_loader_reads_the_experiment_properties(loader, flyhostel_videos):
    assert loader.basedir == flyhostel_videos
    assert loader.dbfile == synthetic.dbfile_path(flyhostel_videos)
    assert loader.framerate == synthetic.FRAMERATE
    assert loader.chunksize == synthetic.CHUNKSIZE
    assert loader.pixels_per_mm == synthetic.PIXELS_PER_CM // 10
    assert loader.square_width == loader.square_height == synthetic.SQUARE_WIDTH
    assert loader.roi_width == synthetic.ROI_WIDTH
    assert loader.wavelet_downsample == 5
    assert loader.number_of_animals == synthetic.NUMBER_OF_ANIMALS
    assert loader.datasetnames == [synthetic.EXPERIMENT + "__01"]
    assert loader.ids == ["FlyHostel1_2X_2026-01-01_1|01"]
    assert str(loader) == synthetic.EXPERIMENT + "__01"


def test_validated_experiments_are_loaded_from_the_validated_tables(loader):
    assert loader.is_validated
    assert loader.identity_table == "IDENTITY_VAL"
    assert loader.roi_0_table == "ROI_0_VAL"


def test_metadata_of_the_fly(loader):
    assert loader.metadata.shape[0] == 1
    assert loader.metadata["identity"].item() == "1"
    assert loader.genotype == "CS"
    assert loader.sex == "FEM"
    assert loader.meta_info == {"t_after_ref": synthetic.T_AFTER_REF, "start_time": synthetic.T_AFTER_REF}


@pytest.fixture
def unvalidated_experiment(experiment_copy, monkeypatch):
    with sqlite3.connect(synthetic.dbfile_path(experiment_copy)) as conn:
        for table in ("ROI_0_VAL", "IDENTITY_VAL", "CONCATENATION_VAL"):
            conn.execute(f"DROP TABLE {table}")
    shutil.rmtree(os.path.join(experiment_copy, "flyhostel", "validation"))
    # not in CVAT either
    monkeypatch.setattr(cvat, "get_project_id_from_name", lambda *args, **kwargs: None)
    return experiment_copy


def test_unvalidated_multi_animal_experiments_are_refused(unvalidated_experiment):
    with pytest.raises(ValueError, match="not validated"):
        FlyHostelLoader(synthetic.EXPERIMENT, identity=1)

    loader = FlyHostelLoader(synthetic.EXPERIMENT, identity=1, ensure_validated=False)
    assert (loader.identity_table, loader.roi_0_table) == ("IDENTITY", "ROI_0")


@pytest.mark.xfail(strict=True, reason="experiment_is_validated looks for a ROI_VAL table, but validation writes ROI_0_VAL, "
                                       "so without the annotations zip it falls back to querying CVAT")
def test_validation_is_detected_from_the_validated_tables(experiment_copy, no_cvat):
    shutil.rmtree(os.path.join(experiment_copy, "flyhostel", "validation"))
    assert cvat.experiment_is_validated(synthetic.EXPERIMENT)


@pytest.mark.parametrize("identity", [1, 2])
def test_centroids(flyhostel_videos, no_cvat, centroid_source, identity):
    centroid_source(os.environ["FLYHOSTEL_VIDEOS"])
    loader = FlyHostelLoader(synthetic.EXPERIMENT, identity=identity)
    loader.load_centroid_data()
    dt = loader.dt.sort_values("frame_number")

    frame_numbers = np.arange(synthetic.N_FRAMES)
    assert dt["frame_number"].tolist() == frame_numbers.tolist()
    assert (dt["id"] == f"FlyHostel1_2X_2026-01-01_1|{str(identity).zfill(2)}").all()

    expected = np.array([synthetic.fly_xy(identity, fn) for fn in frame_numbers])
    np.testing.assert_allclose(dt[["center_x", "center_y"]].values, expected)
    # x and y are normalized to the width of the ROI
    np.testing.assert_allclose(dt["x"].values, expected[:, 0] / synthetic.ROI_WIDTH)
    # top left corner of the square cropped around the fly (exactness tested below)
    np.testing.assert_allclose(dt["tl_x_arena_pixels"].values, expected[:, 0] - synthetic.SQUARE_WIDTH // 2, atol=1)
    np.testing.assert_allclose(dt["tl_y_arena_pixels"].values, expected[:, 1] - synthetic.SQUARE_WIDTH // 2, atol=1)

    # t is seconds since ZT0
    np.testing.assert_allclose(dt["t"].values, synthetic.T_AFTER_REF + frame_time_s(frame_numbers))
    assert loader.meta_info["t_after_ref"] == synthetic.T_AFTER_REF


@pytest.mark.xfail(strict=True, reason="center_x = (x / roi_width) * roi_width is truncated with astype(int), "
                                       "so float round-off (57.9999) moves the anchor stored in the pose file 1 px")
def test_arena_anchor_is_exact(flyhostel_videos, no_cvat, centroid_source):
    centroid_source(os.environ["FLYHOSTEL_VIDEOS"])
    loader = FlyHostelLoader(synthetic.EXPERIMENT, identity=1)
    loader.load_centroid_data()
    dt = loader.dt.sort_values("frame_number")
    expected = np.array([synthetic.fly_xy(1, fn)[0] for fn in dt["frame_number"]])
    np.testing.assert_array_equal(dt["tl_x_arena_pixels"].values, expected - synthetic.SQUARE_WIDTH // 2)


def test_centroids_in_a_time_window(flyhostel_videos, no_cvat, centroid_source):
    centroid_source(os.environ["FLYHOSTEL_VIDEOS"])
    loader = FlyHostelLoader(synthetic.EXPERIMENT, identity=1)
    # not on a frame time (multiples of 1/150 s), so float round-off cannot move frames in or out of the window
    min_time, max_time = synthetic.T_AFTER_REF + 0.201, synthetic.T_AFTER_REF + 0.501
    loader.load_centroid_data(min_time=min_time, max_time=max_time)
    t = loader.dt["t"]
    assert t.min() >= min_time and t.max() <= max_time
    expected = [fn for fn in range(synthetic.N_FRAMES) if min_time <= synthetic.T_AFTER_REF + synthetic.frame_time(fn) / 1000 <= max_time]
    assert sorted(loader.dt["frame_number"].tolist()) == expected


@pytest.mark.xfail(strict=True, reason="ethoscopy's write_query reads ROI_0_VAL for multi-animal experiments "
                                       "regardless of roi_0_table, so non validated data cannot be loaded")
def test_centroids_of_non_validated_experiments(unvalidated_experiment, centroid_source):
    centroid_source(os.environ["FLYHOSTEL_VIDEOS"])
    loader = FlyHostelLoader(synthetic.EXPERIMENT, identity=1, ensure_validated=False)
    loader.load_centroid_data()
    assert loader.dt.shape[0] == synthetic.N_FRAMES


def set_reference_hour(basedir, reference_hour):
    """Change the reference hour of the flies in the METADATA of the dbfile"""
    dbfile = synthetic.dbfile_path(basedir)
    with sqlite3.connect(dbfile) as conn:
        (csv,), = conn.execute("SELECT value FROM METADATA WHERE field = 'ethoscope_metadata'").fetchall()
        header, *rows = [line.split(",") for line in csv.strip().split("\n")]
        column = header.index("reference_hour")
        for row in rows:
            row[column] = str(reference_hour)
        csv = "\n".join(",".join(row) for row in [header] + rows) + "\n"
        conn.execute("UPDATE METADATA SET value = ? WHERE field = 'ethoscope_metadata'", (csv,))


# 15: the recording starts before ZT0, so ZT is counted from the ZT0 of the day before
@pytest.mark.parametrize("reference_hour", [synthetic.REFERENCE_HOUR, 15])
def test_centroids_and_store_index_agree_on_time(experiment_copy, no_cvat, centroid_source, reference_hour):
    set_reference_hour(experiment_copy, reference_hour)
    centroid_source(os.environ["FLYHOSTEL_VIDEOS"], reference_hour=reference_hour)
    loader = FlyHostelLoader(synthetic.EXPERIMENT, identity=1)
    store_index = loader.load_store_index_v2()
    loader.load_centroid_data()
    merged = loader.dt[["frame_number", "t"]].merge(store_index[["frame_number", "t"]], on="frame_number", suffixes=("_centroid", "_index"))
    assert merged.shape[0] == synthetic.N_FRAMES
    np.testing.assert_allclose(merged["t_centroid"], merged["t_index"])


def test_store_index_v2(loader):
    store_index = loader.load_store_index_v2()
    assert store_index.shape[0] == synthetic.N_RECORDED_FRAMES
    np.testing.assert_allclose(store_index["frame_time"].values, frame_time_s(range(synthetic.N_RECORDED_FRAMES)))
    np.testing.assert_allclose(store_index["t"].values, synthetic.T_AFTER_REF + store_index["frame_time"].values)

    # min_time and max_time are in seconds since ZT0
    min_time, max_time = synthetic.T_AFTER_REF + 0.105, synthetic.T_AFTER_REF + 0.195
    window = loader.load_store_index_v2(min_time=min_time, max_time=max_time)
    expected = [fn for fn in range(synthetic.N_RECORDED_FRAMES) if 105 <= synthetic.frame_time(fn) <= 195]
    assert window["frame_number"].tolist() == expected
    assert window["t"].between(min_time, max_time).all()


def test_store_index_from_the_recording(loader):
    loader.load_store_index()
    assert loader.store_index["frame_number"].tolist() == list(range(synthetic.N_RECORDED_FRAMES))
    np.testing.assert_allclose(loader.store_index["frame_time"].values, frame_time_s(range(synthetic.N_RECORDED_FRAMES)))


@pytest.fixture
def compiled_pose(experiment_copy, tmp_path):
    for chunk in synthetic.CHUNKS:
        for local_identity in (1, 2):
            write_predictions(predictions_path(experiment_copy, local_identity, chunk), local_identity, chunk, nodes=BODYPARTS)
    recreate_pose_file(synthetic.EXPERIMENT, 1, output=str(tmp_path))
    return compiled_path(str(tmp_path), 1)


def test_pose(compiled_pose, no_cvat, centroid_source):
    loader = FlyHostelLoader(synthetic.EXPERIMENT, identity=1)
    loader.load_pose_data(files=[(compiled_pose,)])
    pose = loader.pose

    assert pose["frame_number"].tolist() == list(range(synthetic.N_FRAMES))
    assert (pose["id"] == loader.ids[0]).all()
    assert (pose["identity"] == 1).all()
    expected = np.concatenate([
        code(synthetic.local_identity(1, chunk), range(chunk * synthetic.CHUNKSIZE, (chunk + 1) * synthetic.CHUNKSIZE))
        for chunk in synthetic.CHUNKS
    ])
    for bodypart in BODYPARTS:
        np.testing.assert_array_equal(pose[f"{bodypart}_x"].values, expected)
        assert (pose[f"{bodypart}_y"] == BODYPARTS.index(bodypart)).all()
    np.testing.assert_allclose(pose["thorax_likelihood"].values, np.repeat([0.5, 0.6], synthetic.CHUNKSIZE))

    # pose gets its time from the centroid data
    centroid_source(os.environ["FLYHOSTEL_VIDEOS"])
    loader.load_centroid_data()
    pose = loader.annotate_t_in_dataset(pose.drop("t", axis=1))
    np.testing.assert_allclose(pose["t"].values, synthetic.T_AFTER_REF + frame_time_s(range(synthetic.N_FRAMES)))


def test_experiment_properties_from_the_dbfile(flyhostel_videos):
    assert get_number_of_animals(synthetic.EXPERIMENT) == synthetic.NUMBER_OF_ANIMALS
    assert get_framerate(synthetic.EXPERIMENT) == synthetic.FRAMERATE
    assert get_chunksize(synthetic.EXPERIMENT) == synthetic.CHUNKSIZE
    assert get_pixels_per_mm(synthetic.EXPERIMENT) == 15
    assert get_square_width(synthetic.EXPERIMENT) == synthetic.SQUARE_WIDTH
    # flies of an arena get a wider square, whatever the resolution
    assert get_square_width(synthetic.EXPERIMENT + "_arena1") == 260
    assert get_wavelet_downsample(synthetic.EXPERIMENT) == 5

    dbfile = synthetic.dbfile_path(flyhostel_videos)
    assert get_first_frame(dbfile) == 0
    assert get_last_frame(dbfile) == synthetic.N_FRAMES


def test_local_identities(flyhostel_videos):
    dbfile = synthetic.dbfile_path(flyhostel_videos)
    table = get_local_identities(dbfile, [0, 65])
    table = table.sort_values(["chunk", "identity"])
    assert table[["chunk", "identity", "local_identity", "frame_number"]].values.tolist() == [
        [0, 1, 1, 0], [0, 2, 2, 0], [1, 1, 2, 60], [1, 2, 1, 60],
    ]


def test_load_metadata_and_meta_info(flyhostel_videos):
    dbfile = synthetic.dbfile_path(flyhostel_videos)
    metadata = load_metadata(dbfile, 2)
    assert metadata["identity"].tolist() == ["2"]
    assert metadata["number_of_animals"].tolist() == [str(synthetic.NUMBER_OF_ANIMALS)]
    assert load_meta_info(dbfile, 2)["t_after_ref"] == synthetic.T_AFTER_REF
    # the reference hour can be overriden (in seconds)
    assert load_meta_info(dbfile, 2, reference_hour=12 * 3600)["t_after_ref"] == 2 * 3600
    assert load_meta_info(dbfile, 2, reference_hour=11.5 * 3600)["t_after_ref"] == 2.5 * 3600
    # lights on after the start of the recording: ZT0 of the day before
    assert load_meta_info(dbfile, 2, reference_hour=15 * 3600)["t_after_ref"] == 23 * 3600
    with pytest.raises(Exception):
        load_metadata(dbfile, 3)


def test_centroid_cache_with_another_t_after_ref_is_shifted(experiment_copy, no_cvat, centroid_source):
    """Caches saved before a change in how ZT is computed must not keep the old t"""
    centroid_source(os.environ["FLYHOSTEL_VIDEOS"], reference_hour=synthetic.REFERENCE_HOUR)
    loader = FlyHostelLoader(synthetic.EXPERIMENT, identity=1)
    loader.load_centroid_data()
    cached = loader.dt[["frame_number", "t"]].sort_values("frame_number")

    # same cache, lights on 4 h later (and the day before)
    set_reference_hour(experiment_copy, 15)
    centroid_source(os.environ["FLYHOSTEL_VIDEOS"], reference_hour=15)
    loader = FlyHostelLoader(synthetic.EXPERIMENT, identity=1)
    loader.load_centroid_data()
    shifted = loader.dt[["frame_number", "t"]].sort_values("frame_number")
    assert loader.meta_info["t_after_ref"] == 23 * 3600
    np.testing.assert_allclose(shifted["t"].values, cached["t"].values + (23 - 3) * 3600)
