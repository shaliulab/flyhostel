"""
Tests for the analysis of pose into behavior (motionmapper wavelets + classifier + rules, flyhostel.data.pose.ethogram).

The classifier is replaced by a deterministic toy model (the class with the largest feature wins),
so what is tested is the plumbing around it: feature loading, probabilities, removal of short bouts,
the rules that produce prediction2 and the bout annotation.
"""
import os
import pickle

import h5py
import numpy as np
import pandas as pd
import pytest

import flyhostel.data.pose.ethogram.ethogram as ethogram
from flyhostel.data.pose.constants import BEHAVIOR_IDX_MAP
from flyhostel.data.pose.ethogram.inference import inference_
from flyhostel.data.pose.ethogram.rules import apply_proboscis_requirement, apply_sequence_rules
from flyhostel.data.pose.ethogram.utils import annotate_bout_duration, annotate_bouts, most_common_behavior_vectorized
from flyhostel.data.pose.loaders.wavelets import WaveletLoader

FPS = 30
CLASSES = np.array(sorted(BEHAVIOR_IDX_MAP.keys()))
FEATURES = [f"f_{behavior}" for behavior in CLASSES]


class ArgmaxModel:
    """Predicts the class whose feature is largest"""
    classes_ = CLASSES

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        return x / x.sum(axis=1, keepdims=True)


class IdentityScaler:
    def transform(self, x):
        return x


@pytest.fixture
def model_path(tmp_path):
    path = str(tmp_path / "model.pkl")
    with open(path, "wb") as handle:
        pickle.dump((ArgmaxModel(), FEATURES, IdentityScaler()), handle)
    return path


@pytest.fixture
def bout_lengths(monkeypatch):
    """Shortest bout (in frames) kept for every behavior, normally computed from the ground truth (DEG) database"""
    lengths = {behavior: 1 for behavior in CLASSES}
    lengths["groom"] = 10
    monkeypatch.setattr(ethogram, "get_bout_length_percentile_from_project", lambda *args, **kwargs: lengths)
    return lengths


def bouts_dataset(bouts, id="fly|01"):
    """
    Pose features of a fly built from bouts of (behavior, number of frames, extra columns).
    The feature of the behavior is 0.9 and the rest share 0.1
    """
    rows = []
    for behavior, n, extra in bouts:
        for _ in range(n):
            row = {feature: 0.1 / (len(FEATURES) - 1) for feature in FEATURES}
            row[f"f_{behavior}"] = 0.9
            row.update({"proboscis": 0.5, "head_proboscis_distance": 1.0, "food_distance": 1.0})
            row.update(extra)
            rows.append(row)
    dataset = pd.DataFrame(rows)
    dataset.insert(0, "id", id)
    dataset.insert(1, "frame_number", np.arange(len(dataset)) * 5)
    dataset.insert(2, "t", dataset["frame_number"] / 150)
    return dataset


def run_inference(dataset, model_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return inference_(dataset, "behavior", model_path=model_path, fps=FPS, deg_folder=None)


def test_probabilities_and_prediction(model_path, bout_lengths, tmp_path, monkeypatch):
    dataset = bouts_dataset([("walk", 300, {}), ("inactive", 300, {})])
    preds = run_inference(dataset, model_path, tmp_path, monkeypatch)

    assert preds.shape[0] == dataset.shape[0]
    assert preds["frame_number"].tolist() == dataset["frame_number"].tolist()
    for behavior in CLASSES:
        assert f"{behavior}_prob" in preds.columns
    np.testing.assert_allclose(preds[[f"{b}_prob" for b in CLASSES]].sum(axis=1), 1)
    assert preds["prediction"].tolist() == ["walk"] * 300 + ["inactive"] * 300
    assert preds["prediction2"].tolist() == preds["prediction"].tolist()
    np.testing.assert_allclose(preds["score"], 0.9)
    # the temporary file is cleaned up
    assert not os.path.exists(tmp_path / "backup.feather")


def test_bout_annotation_of_prediction2(model_path, bout_lengths, tmp_path, monkeypatch):
    preds = run_inference(bouts_dataset([("walk", 300, {}), ("inactive", 600, {})]), model_path, tmp_path, monkeypatch)
    assert preds["duration_pred"].tolist() == [10.0] * 300 + [20.0] * 600
    assert preds["bout_in_pred"].iloc[[0, 299, 300]].tolist() == [1, 300, 1]
    assert preds["bout_out_pred"].iloc[[0, 299, 300]].tolist() == [300, 1, 600]


def test_bouts_shorter_than_the_ground_truth_percentile_are_removed(model_path, bout_lengths, tmp_path, monkeypatch):
    # a 5 frame groom bout is shorter than the 10 frames required. It becomes background,
    # and a background bout shorter than 5 s becomes micromovement (rule 1.5)
    dataset = bouts_dataset([("inactive", 300, {}), ("groom", 5, {}), ("inactive", 300, {}), ("groom", 300, {})])
    preds = run_inference(dataset, model_path, tmp_path, monkeypatch)
    assert preds["prediction"].iloc[300:305].tolist() == ["background"] * 5
    assert preds["prediction2"].iloc[300:305].tolist() == ["inactive+micromovement"] * 5
    assert (preds["rule"].iloc[300:305] == 1.5).all()
    # long groom bouts are kept
    assert (preds["prediction2"].iloc[-300:] == "groom").all()


def test_proboscis_extension_requires_a_visible_proboscis(model_path, bout_lengths, tmp_path, monkeypatch):
    dataset = bouts_dataset([("inactive+pe", 60, {"proboscis": 0}), ("inactive+pe", 60, {"proboscis": 0.8}), ("walk", 300, {})])
    preds = run_inference(dataset, model_path, tmp_path, monkeypatch)
    assert preds["prediction"].iloc[:60].tolist() == ["inactive"] * 60
    assert preds["prediction"].iloc[60:120].tolist() == ["inactive+pe"] * 60


def test_long_proboscis_extension_is_feeding(model_path, bout_lengths, tmp_path, monkeypatch):
    # inactive+pe for more than 5 s becomes feed (rule 3)
    dataset = bouts_dataset([("inactive+pe", 6 * FPS, {"proboscis": 0.8}), ("walk", 300, {})])
    preds = run_inference(dataset, model_path, tmp_path, monkeypatch)
    assert preds["prediction"].iloc[:180].tolist() == ["inactive+pe"] * 180
    assert preds["prediction2"].iloc[:180].tolist() == ["feed"] * 180
    assert (preds["rule"].iloc[:180] == 3).all()


def test_inactive_with_proboscis_on_the_food_is_feeding(model_path, bout_lengths, tmp_path, monkeypatch):
    dataset = bouts_dataset([
        ("inactive", 300, {"proboscis": 0.8, "food_distance": 0.0}),
        ("inactive", 300, {"proboscis": 0.8, "food_distance": 1.0}),
    ])
    preds = run_inference(dataset, model_path, tmp_path, monkeypatch)
    assert preds["prediction2"].iloc[:300].tolist() == ["feed"] * 300
    assert preds["prediction2"].iloc[300:].tolist() == ["inactive"] * 300


def test_missing_features_are_filled_from_neighbouring_frames(model_path, bout_lengths, tmp_path, monkeypatch):
    dataset = bouts_dataset([("walk", 300, {})])
    dataset.loc[10:20, FEATURES] = np.nan
    preds = run_inference(dataset, model_path, tmp_path, monkeypatch)
    assert (preds["prediction"] == "walk").all()


def test_feed_far_from_the_food_is_proboscis_extension():
    df = pd.DataFrame({
        "prediction": ["feed"] * 4, "prediction2": ["feed", "feed", "inactive+pe", "inactive+pe"],
        "food_distance": [1.0, 0.0, 0.0, 1.0], "proboscis": 1.0, "duration_pred": 1.0,
    })
    out = apply_sequence_rules(df, "inactive+micromovement")
    assert out["prediction2"].tolist() == ["inactive+pe", "feed", "feed", "inactive+pe"]


def test_proboscis_requirement_only_changes_pe_and_feed():
    df = pd.DataFrame({
        "prediction": ["inactive+pe", "feed", "walk", "inactive+pe"],
        "proboscis": [0, 0.9, 0, 0.9], "head_proboscis_distance": [1, 0, 1, 1],
    })
    assert apply_proboscis_requirement(df, "inactive")["prediction"].tolist() == ["inactive", "inactive", "walk", "inactive+pe"]


def test_annotate_bouts():
    df = pd.DataFrame({"behavior": ["a", "a", "a", "b", "a", "a"]})
    out = annotate_bout_duration(annotate_bouts(df, "behavior"), fps=2)
    assert out["bout_count"].tolist() == [1, 1, 1, 2, 3, 3]
    assert out["bout_in"].tolist() == [1, 2, 3, 1, 1, 2]
    assert out["bout_out"].tolist() == [3, 2, 1, 1, 2, 1]
    assert out["duration"].tolist() == [1.5, 1.5, 1.5, 0.5, 1.0, 1.0]


def test_most_common_behavior_per_second():
    df = pd.DataFrame({
        "id": "fly", "t": np.arange(6) / 3, "frame_number": np.arange(6) * 50,
        "behavior": ["walk", "walk", "groom", "groom", "groom", "inactive"],
    })
    out = most_common_behavior_vectorized(df, time_window_length=1)
    assert out[["t", "behavior", "fraction", "frame_number"]].values.tolist() == [[0, "walk", 0.67, 0], [1, "groom", 0.67, 150]]


@pytest.fixture
def wavelet_loader():
    loader = WaveletLoader()
    loader.experiment = "FlyHostel1_2X_2026-01-01_14-00-00"
    loader.identity = 1
    loader.ids = ["FlyHostel1_2X_2026-01-01_1|01"]
    return loader


def test_load_wavelets(wavelet_loader, tmp_path):
    wavelet_loader.basedir = str(tmp_path)
    matfile = wavelet_loader.get_matfile()
    assert matfile == os.path.join(
        str(tmp_path), "motionmapper", "01", "wavelets/FlyHostel_long_timescale_analysis/Wavelets",
        "FlyHostel1_2X_2026-01-01_14-00-00__01-pcaModes-wavelets.mat",
    )
    os.makedirs(os.path.dirname(matfile))
    names = ["lW_x_0.5", "lW_y_0.5", "lW_x_1.0", "lW_y_1.0"]
    wavelets = np.arange(40, dtype=float).reshape(10, 4)
    with h5py.File(matfile, "w") as handle:
        handle.create_dataset("wavelets", data=wavelets)
        handle.create_dataset("freq_names", data=[n.encode() for n in names])
        handle.create_dataset("f", data=[0.5, 1.0])

    loaded, (frequencies, freq_names) = wavelet_loader.load_wavelets()
    assert freq_names == names
    assert frequencies.tolist() == [0.5, 1.0]
    assert loaded.columns.tolist() == names
    assert (loaded.index == wavelet_loader.ids[0]).all()
    np.testing.assert_array_equal(loaded.values, wavelets)

    subset, _ = wavelet_loader.load_wavelets(frames=[2, 5])
    np.testing.assert_array_equal(subset.values, wavelets[[2, 5]])


def test_missing_wavelets(wavelet_loader, tmp_path):
    wavelet_loader.basedir = str(tmp_path)
    assert wavelet_loader.load_wavelets() == (None, (None, None))


@pytest.mark.parametrize("framerate,downsample", [(150, 5), (47, 1), (47.1, 1)])
def test_wavelet_downsample_matches_the_motionmapper_profile(monkeypatch, framerate, downsample):
    PROFILES = pytest.importorskip("motionmapperpy").PROFILES
    import flyhostel.utils.utils as utils
    monkeypatch.setattr(utils, "get_framerate", lambda experiment: framerate)
    profile = utils.get_wavelet_profile("experiment")
    assert utils.get_wavelet_downsample("experiment") == downsample == PROFILES[profile]["wavelet_downsample"]
    assert PROFILES[profile]["samplingFreq"] == framerate


def write_wavelets_with_motionmapper(tmp_path, monkeypatch, projections, node_names, profile="default"):
    mmpy = pytest.importorskip("motionmapperpy")
    # motionmapperpy writes config.yaml and a status file to the working directory
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "project"
    (project / "Wavelets").mkdir(parents=True)
    path = str(tmp_path / "animal.h5")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("projections", data=projections.T)
        handle.create_dataset("node_names", data=[n.encode() for n in node_names])
    parameters = mmpy.setRunParameters()
    parameters.update(mmpy.PROFILES[profile])
    parameters.projectPath = str(project)
    parameters.waveletDecomp = True
    # numProcessors=1 references an undefined multiprocessing pool
    parameters.numProcessors = 2
    mmpy.motionmapper.calc_and_write_wavelets_ls(path, parameters)
    with h5py.File(str(project / "Wavelets" / "animal-wavelets.mat"), "r") as handle:
        return handle["wavelets"][:], [n.decode() for n in handle["freq_names"][:]], handle["f"][:]


def test_feature_names_match_the_motionmapper_wavelet_names(tmp_path, monkeypatch):
    from flyhostel.data.pose.ethogram.train import get_frequencies_bps
    from motionmapperpy import PROFILES
    nodes = ["lW", "fRL"]
    _, freq_names, _ = write_wavelets_with_motionmapper(tmp_path, monkeypatch, np.random.default_rng(0).normal(size=(600, 4)), nodes)
    profile = PROFILES["default"]
    features = get_frequencies_bps(profile["minF"], profile["maxF"], profile["numPeriods"], bodyparts_xy=["lW_x", "lW_y", "fRL_x", "fRL_y"])
    assert sorted(features) == sorted(freq_names)


@pytest.mark.xfail(strict=True, reason="motionmapperpy stores the amplitudes of each projection in a contiguous block of "
                                       "frequencies, but names them interleaving x and y at every frequency, "
                                       "so wavelet columns carry the name of another body part axis and frequency")
def test_wavelet_columns_are_named_after_their_projection_and_frequency(tmp_path, monkeypatch):
    framerate = 150
    t = np.arange(1500) / framerate
    projections = np.zeros((len(t), 4))
    # only lW_x moves, at 5 Hz
    projections[:, 0] = np.sin(2 * np.pi * 5 * t)
    wavelets, freq_names, frequencies = write_wavelets_with_motionmapper(tmp_path, monkeypatch, projections, ["lW", "fRL"])
    strongest = freq_names[int(np.argmax(wavelets.sum(axis=0)))]
    assert strongest.startswith("lW_x_")
    assert abs(float(strongest.split("_")[-1]) - 5) < 1
