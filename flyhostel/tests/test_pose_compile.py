"""
Tests for compiling the per chunk SLEAP output of each local identity into one pose file per fly
(compile-pose, flyhostel.utils.pose_export).

Each synthetic chunk file stores in its tracks the value 1000 * local_identity + frame_number,
so the compiled file tells which local identity and frame every pose came from.
"""
import os

import h5py
import numpy as np
import pandas as pd
import pytest

from flyhostel.tests import synthetic_experiment as synthetic
from flyhostel.utils.pose_export import (
    estimate_body_length_mm,
    generate_single_file,
    get_first_frame_number,
    load_concatenation_table,
    load_file,
    load_files,
    load_frame_numbers,
    pipeline,
    recreate_pose_file,
)

NODES = ["thorax", "abdomen", "head", "proboscis"]


def code(local_identity, frame_numbers):
    return 1000 * local_identity + np.asarray(frame_numbers, dtype=float)


def write_predictions(path, local_identity, chunk, chunksize=synthetic.CHUNKSIZE, nodes=NODES, instance_scores=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame_numbers = np.arange(chunk * chunksize, chunk * chunksize + chunksize)
    tracks = np.zeros((1, 2, len(nodes), chunksize))
    tracks[0, 0] = code(local_identity, frame_numbers)
    tracks[0, 1] = np.arange(len(nodes))[:, None]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("node_names", data=np.array([n.encode() for n in nodes]))
        handle.create_dataset("tracks", data=tracks)
        handle.create_dataset("point_scores", data=np.full((1, len(nodes), chunksize), 0.5 + chunk / 10))
        if instance_scores:
            handle.create_dataset("instance_scores", data=np.full((1, chunksize), float(local_identity)))
    return path


def predictions_path(basedir, local_identity, chunk):
    return os.path.join(basedir, "flyhostel", "single_animal", str(local_identity).zfill(3), f"{str(chunk).zfill(6)}.mp4.predictions.h5")


@pytest.fixture
def experiment_with_pose(experiment_copy):
    for chunk in synthetic.CHUNKS:
        for local_identity in (1, 2):
            write_predictions(predictions_path(experiment_copy, local_identity, chunk), local_identity, chunk)
    return experiment_copy


def compiled_path(output, identity):
    animal = f"{synthetic.EXPERIMENT}__{str(identity).zfill(2)}"
    return os.path.join(output, animal, animal + ".h5")


@pytest.mark.parametrize("identity", [1, 2])
def test_compiled_pose_follows_the_identity_across_chunks(experiment_with_pose, tmp_path, identity):
    recreate_pose_file(synthetic.EXPERIMENT, identity, output=str(tmp_path))
    with h5py.File(compiled_path(str(tmp_path), identity), "r") as handle:
        assert [n.decode() for n in handle["node_names"][:]] == NODES
        tracks = handle["tracks"][:]
        assert tracks.shape == (1, 2, len(NODES), synthetic.N_FRAMES)

        expected = np.concatenate([
            code(synthetic.local_identity(identity, chunk), np.arange(chunk * synthetic.CHUNKSIZE, (chunk + 1) * synthetic.CHUNKSIZE))
            for chunk in synthetic.CHUNKS
        ])
        for node in range(len(NODES)):
            np.testing.assert_array_equal(tracks[0, 0, node], expected)
            assert (tracks[0, 1, node] == node).all()

        files = [f.decode() for f in handle["files"][:]]
        assert files == [predictions_path(experiment_with_pose, synthetic.local_identity(identity, chunk), chunk) for chunk in synthetic.CHUNKS]

        point_scores = handle["point_scores"][:]
        assert point_scores.shape == (1, len(NODES), synthetic.N_FRAMES)
        np.testing.assert_allclose(point_scores[0, 0, ::synthetic.CHUNKSIZE], [0.5, 0.6])

        instance_scores = handle["instance_scores"][:]
        assert instance_scores.shape == (1, synthetic.N_FRAMES)
        assert instance_scores[0].tolist() == [float(synthetic.local_identity(identity, chunk)) for chunk in synthetic.CHUNKS for _ in range(synthetic.CHUNKSIZE)]

    path = compiled_path(str(tmp_path), identity)
    assert get_first_frame_number(path, synthetic.CHUNKSIZE) == 0
    assert load_frame_numbers(path, synthetic.CHUNKSIZE).tolist() == list(range(synthetic.N_FRAMES))


def centroid_data(frame_numbers):
    return pd.DataFrame({
        "frame_number": frame_numbers,
        "tl_x_arena_pixels": np.asarray(frame_numbers) + 0.5,
        "tl_y_arena_pixels": -np.asarray(frame_numbers, dtype=float),
        "t": synthetic.T_AFTER_REF + np.asarray(frame_numbers) / synthetic.FRAMERATE,
    })


def test_compiled_pose_stores_anchor_and_time_of_every_frame(experiment_with_pose, tmp_path):
    dt = centroid_data(np.arange(synthetic.N_FRAMES))
    recreate_pose_file(synthetic.EXPERIMENT, 1, output=str(tmp_path), dt=dt)
    with h5py.File(compiled_path(str(tmp_path), 1), "r") as handle:
        assert handle["anchor"].shape == (synthetic.N_FRAMES, 2)
        np.testing.assert_array_equal(handle["anchor"][:], dt[["tl_x_arena_pixels", "tl_y_arena_pixels"]].values)
        # float64, otherwise the timestamps get quantized (see SleepLoader.load_data_for_social_regression)
        assert handle["t"].dtype == np.float64
        np.testing.assert_array_equal(handle["t"][:], dt["t"].values)


def test_frames_without_centroid_get_nan_anchor_and_time(experiment_with_pose, tmp_path):
    dt = centroid_data(np.arange(10, synthetic.N_FRAMES))
    recreate_pose_file(synthetic.EXPERIMENT, 1, output=str(tmp_path), dt=dt)
    with h5py.File(compiled_path(str(tmp_path), 1), "r") as handle:
        assert handle["anchor"].shape == (synthetic.N_FRAMES, 2)
        assert np.isnan(handle["t"][:10]).all()
        assert not np.isnan(handle["t"][10:]).any()


def test_only_requested_chunks_are_compiled(experiment_with_pose, tmp_path):
    recreate_pose_file(synthetic.EXPERIMENT, 2, chunks=[1], output=str(tmp_path))
    path = compiled_path(str(tmp_path), 2)
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["tracks"][0, 0, 0], code(1, np.arange(60, 120)))
    assert load_frame_numbers(path, synthetic.CHUNKSIZE).tolist() == list(range(60, 120))


def concatenation(experiment_with_pose, chunks, identity=1):
    """Concatenation table of one fly, as passed to pipeline, available only in the given chunks"""
    rows = []
    for chunk in chunks:
        local = synthetic.local_identity(identity, chunk)
        path = predictions_path(experiment_with_pose, local, chunk)
        if not os.path.exists(path):
            write_predictions(path, local, chunk)
        rows.append({"chunk": chunk, "identity": identity, "local_identity": local, "dfile": path})
    return pd.DataFrame(rows)


def test_strict_pipeline_refuses_missing_chunks(experiment_with_pose, tmp_path):
    table = concatenation(experiment_with_pose, [0, 2])
    with pytest.raises(Exception, match="Chunks missing"):
        pipeline(synthetic.EXPERIMENT, 1, table, chunks=[0, 1, 2], output=str(tmp_path), strict=True)


def test_non_strict_pipeline_stops_at_the_first_gap(experiment_with_pose, tmp_path):
    table = concatenation(experiment_with_pose, [0, 2])
    pipeline(synthetic.EXPERIMENT, 1, table, chunks=[0, 1, 2], output=str(tmp_path), strict=False)
    with h5py.File(compiled_path(str(tmp_path), 1), "r") as handle:
        assert handle["tracks"].shape[3] == synthetic.CHUNKSIZE


@pytest.mark.xfail(strict=True, reason="after truncating at a gap, interval is still computed from the requested chunks, "
                                       "so anchor and t are longer than tracks")
def test_non_strict_pipeline_anchor_matches_tracks_after_a_gap(experiment_with_pose, tmp_path):
    table = concatenation(experiment_with_pose, [0, 2])
    dt = centroid_data(np.arange(3 * synthetic.CHUNKSIZE))
    pipeline(synthetic.EXPERIMENT, 1, table, chunks=[0, 1, 2], output=str(tmp_path), strict=False, dt=dt)
    with h5py.File(compiled_path(str(tmp_path), 1), "r") as handle:
        assert handle["anchor"].shape[0] == handle["tracks"].shape[3]


@pytest.mark.xfail(strict=True, raises=IndexError,
                   reason="when the missing chunks are at the end there is no gap to find, and .iloc[0] of an empty selection raises")
def test_non_strict_pipeline_with_missing_chunks_at_the_end(experiment_with_pose, tmp_path):
    table = concatenation(experiment_with_pose, [0, 1])
    pipeline(synthetic.EXPERIMENT, 1, table, chunks=[0, 1, 2], output=str(tmp_path), strict=False)


@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason="the diagnostic print indexes .loc with positional rows, so it raises KeyError before the ValueError")
def test_concatenation_with_missing_chunks_is_rejected(experiment_with_pose):
    import sqlite3
    with sqlite3.connect(synthetic.dbfile_path(experiment_with_pose)) as conn:
        conn.execute("DELETE FROM CONCATENATION_VAL WHERE chunk = 1")
        conn.execute("INSERT INTO CONCATENATION_VAL (chunk, local_identity, identity) VALUES (2, 1, 1)")
        with pytest.raises(ValueError, match="Missing chunks"):
            load_concatenation_table(conn.cursor(), experiment_with_pose, concatenation_table="CONCATENATION_VAL")


def test_load_file_checks_the_number_of_frames(tmp_path):
    path = write_predictions(str(tmp_path / "x.h5"), 1, 0, chunksize=50)
    with pytest.raises(AssertionError, match="missing pose estimates"):
        load_file(path, chunksize=60)
    node_names, tracks, score, inst, file = load_file(path, chunksize=50)
    assert node_names == NODES and tracks.shape == (1, 2, 4, 50) and inst.shape == (1, 50)


def test_load_file_without_instance_scores_fills_nan(tmp_path):
    path = write_predictions(str(tmp_path / "x.h5"), 1, 0, instance_scores=False)
    _, _, _, inst, _ = load_file(path)
    assert inst.shape == (1, synthetic.CHUNKSIZE) and np.isnan(inst).all()


def test_load_files_refuses_missing_files_and_inconsistent_skeletons(tmp_path):
    good = write_predictions(str(tmp_path / "a.h5"), 1, 0)
    with pytest.raises(ValueError, match="could not be loaded"):
        load_files([good, str(tmp_path / "missing.h5")], chunksize=synthetic.CHUNKSIZE)

    other = write_predictions(str(tmp_path / "b.h5"), 1, 1, nodes=["thorax", "head", "abdomen", "proboscis"])
    with pytest.raises(AssertionError):
        load_files([good, other], chunksize=synthetic.CHUNKSIZE)


def test_generate_single_file_concatenates_in_order(tmp_path):
    files = [write_predictions(str(tmp_path / f"{chunk}.h5"), 1, chunk) for chunk in (3, 4)]
    node_names, datasets, point_scores, inst_scores = load_files(files, chunksize=synthetic.CHUNKSIZE)
    dest = generate_single_file(node_names, datasets, point_scores, inst_scores, files, str(tmp_path / "out.h5"))
    with h5py.File(dest, "r") as handle:
        np.testing.assert_array_equal(handle["tracks"][0, 0, 0], code(1, np.arange(180, 300)))


@pytest.mark.parametrize("estimator,expected", [("median", 2.0), ("max", 10.0), ("mean", 3.6), ("percentile", 10.0)])
def test_body_length_scalar_estimators(estimator, expected):
    px = np.array([20, 20, 20, 100, 20, np.nan, 500])
    conf = np.array([1, 1, 1, 1, 1, 1, 0])
    out = estimate_body_length_mm(px, conf, pixels_per_mm=10, estimator=estimator, percentile=100, min_confidence=0.5)
    assert out.shape == px.shape
    np.testing.assert_allclose(out, expected)


def test_body_length_per_frame_masks_low_confidence():
    out = estimate_body_length_mm(np.array([20.0, 30.0, np.nan]), np.array([1, 0, 1]), pixels_per_mm=10, estimator="per_frame", min_confidence=0.5)
    np.testing.assert_array_equal(out, [2.0, np.nan, np.nan])


def test_body_length_without_valid_frames_is_nan():
    assert np.isnan(estimate_body_length_mm([1.0, 2.0], [0, 0], 10, min_confidence=0.5)).all()
    with pytest.raises(ValueError):
        estimate_body_length_mm([1.0], [1], 10, estimator="mode")
