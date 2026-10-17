"""
Fixtures that build the synthetic FlyHostel experiment in flyhostel/tests/synthetic_experiment.py.

The raw recording and its dbfile are built once per session (writing the mp4 chunks is the slow part).
Tests that change files on disk must work on a copy (experiment_copy).
"""
import os

import pytest

from flyhostel.tests import synthetic_experiment as synthetic


@pytest.fixture(scope="session")
def raw_experiment(tmp_path_factory):
    """FLYHOSTEL_VIDEOS folder with the raw recording, before the dbfile is made"""
    root = str(tmp_path_factory.mktemp("raw_videos"))
    synthetic.build_raw_experiment(root)
    return root


@pytest.fixture(scope="session")
def exported_experiment(raw_experiment, tmp_path_factory):
    """FLYHOSTEL_VIDEOS folder with the recording and its dbfile (validated). Do not modify"""
    root = str(tmp_path_factory.mktemp("videos"))
    basedir = synthetic.copy_experiment(raw_experiment, os.path.join(root, "videos"))
    cwd = os.getcwd()
    with pytest.MonkeyPatch.context() as monkeypatch:
        synthetic.export_dbfile(basedir, monkeypatch)
    os.chdir(cwd)
    synthetic.validate_dbfile(basedir)
    return os.path.join(root, "videos")


@pytest.fixture
def experiment_copy(exported_experiment, tmp_path, monkeypatch):
    """A private copy of the exported experiment, set as FLYHOSTEL_VIDEOS. Returns its basedir"""
    root = str(tmp_path / "videos")
    basedir = synthetic.copy_experiment(exported_experiment, root)
    monkeypatch.setenv("FLYHOSTEL_VIDEOS", root)
    return basedir


@pytest.fixture
def flyhostel_videos(exported_experiment, monkeypatch):
    """Point FLYHOSTEL_VIDEOS to the shared, read-only exported experiment. Returns its basedir"""
    monkeypatch.setenv("FLYHOSTEL_VIDEOS", exported_experiment)
    return os.path.join(exported_experiment, synthetic.RELPATH)
