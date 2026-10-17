"""
Tests for the production of FlyHostel dbfiles (fh export, flyhostel.data.sqlite3).

The synthetic recording in synthetic_experiment.py is exported once per session and every table
is checked against the ground truth used to draw the video.
"""
import calendar
import datetime
import json
import os
import sqlite3
import time

import cv2
import numpy as np
import pandas as pd
import pytest

from flyhostel.data.sqlite3 import IdtrackeraiExporter
from flyhostel.data.sqlite3.concatenation import ConcatenationExporter
from flyhostel.data.sqlite3.constants import PRESETS
from flyhostel.tests import synthetic_experiment as synthetic


@pytest.fixture(scope="module")
def dbfile(exported_experiment):
    return synthetic.dbfile_path(os.path.join(exported_experiment, synthetic.RELPATH))


def read_table(dbfile, table):
    with sqlite3.connect(dbfile) as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)


def metadata(dbfile):
    return read_table(dbfile, "METADATA").set_index("field")["value"].to_dict()


def test_dbfile_is_named_after_the_experiment(dbfile):
    assert os.path.basename(dbfile) == synthetic.EXPERIMENT + ".db"


def test_all_default_and_trajectory_tables_are_written(dbfile):
    with sqlite3.connect(dbfile) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in PRESETS["default_and_trajectory"]:
        assert table in tables
        assert read_table(dbfile, table).shape[0] > 0, f"{table} is empty"


def test_metadata(dbfile):
    meta = metadata(dbfile)
    assert float(meta["framerate"]) == synthetic.FRAMERATE
    assert float(meta["data_framerate"]) == synthetic.FRAMERATE
    assert int(float(meta["chunksize"])) == synthetic.CHUNKSIZE
    assert int(meta["pixels_per_cm"]) == synthetic.PIXELS_PER_CM
    assert int(meta["frame_width"]) == int(meta["frame_height"]) == synthetic.IMG_SIZE
    # chunks spans the whole recording, not only the exported chunks
    assert meta["chunks"] == "0,2"
    assert int(meta["first_time"]) == 0
    assert meta["machine_name"] == f"FlyHostel{synthetic.FLYHOSTEL_ID}"
    assert json.loads(meta["idtrackerai_conf"])["_number_of_animals"]["value"] == synthetic.NUMBER_OF_ANIMALS
    assert meta["camera_metadata"].startswith("# camera settings")

    with open(os.path.join(os.path.dirname(dbfile), "metadata.csv")) as handle:
        assert meta["ethoscope_metadata"] == handle.read()


def test_metadata_date_time_is_the_utc_start_of_the_store(dbfile):
    created = datetime.datetime.strptime(synthetic.CREATED_UTC, "%Y-%m-%dT%H:%M:%S")
    assert float(metadata(dbfile)["date_time"]) == calendar.timegm(created.timetuple())


@pytest.fixture
def local_timezone():
    """Run the test with the computer set to the given timezone"""
    original = os.environ.get("TZ")

    def set_timezone(tz):
        os.environ["TZ"] = tz
        time.tzset()

    yield set_timezone
    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


@pytest.mark.parametrize("tz", [
    "UTC", "Europe/Brussels", "Asia/Tokyo",
    pytest.param("America/New_York", marks=pytest.mark.xfail(
        strict=True,
        reason="write_metadata_table adds utcoffset(dt).seconds, which is 86400 - |offset| west of UTC, "
               "so date_time lands one day late")),
])
def test_metadata_date_time_does_not_depend_on_the_timezone_of_the_computer(raw_experiment, tmp_path, tz, local_timezone):
    local_timezone(tz)
    exporter = IdtrackeraiExporter(os.path.join(raw_experiment, synthetic.RELPATH))
    dbfile = str(tmp_path / "metadata.db")
    exporter.init_metadata_table(dbfile)
    exporter.write_metadata_table(dbfile)
    created = datetime.datetime.strptime(synthetic.CREATED_UTC, "%Y-%m-%dT%H:%M:%S")
    assert float(metadata(dbfile)["date_time"]) == calendar.timegm(created.timetuple())


def test_store_index_matches_the_recording_index(dbfile):
    store_index = read_table(dbfile, "STORE_INDEX")
    with sqlite3.connect(os.path.join(os.path.dirname(dbfile), "index.db")) as conn:
        frames = pd.read_sql("SELECT chunk, frame_number, frame_time FROM frames", conn)

    assert store_index.shape[0] == synthetic.N_RECORDED_FRAMES
    pd.testing.assert_frame_equal(
        store_index[["chunk", "frame_number", "frame_time"]].astype(np.int64),
        frames.astype(np.int64),
    )
    # half_second flags one frame every half second
    expected = (store_index["frame_number"] % (synthetic.FRAMERATE // 2) == 0).astype(int)
    assert (store_index["half_second"] == expected).all()


def test_roi_0_contains_the_centroid_of_every_fly_in_every_frame(dbfile):
    roi_0 = read_table(dbfile, "ROI_0")
    assert roi_0.shape[0] == synthetic.N_FRAMES * synthetic.NUMBER_OF_ANIMALS
    assert not roi_0.duplicated(["frame_number", "in_frame_index"]).any()

    for _, row in roi_0.iterrows():
        identity = row["in_frame_index"] + 1
        assert (row["x"], row["y"]) == synthetic.fly_xy(identity, row["frame_number"])
    assert (roi_0["area"] == round(np.pi * synthetic.FLY_RADIUS ** 2)).all()


def test_roi_0_flags_frames_modified_by_yolov7(dbfile):
    roi_0 = read_table(dbfile, "ROI_0")
    modified = roi_0.loc[roi_0["modified"] == 1]
    assert modified[["frame_number", "in_frame_index", "class_name"]].values.tolist() == [[synthetic.MODIFIED_FRAME, 0, "fly"]]
    assert roi_0.loc[roi_0["modified"] == 0, "class_name"].isna().all()
    assert read_table(dbfile, "AI").values.tolist() == [[synthetic.MODIFIED_FRAME, "YOLOv7"]]


def test_identity_maps_local_identities_to_identities_across_chunks(dbfile):
    identity = read_table(dbfile, "IDENTITY")
    assert identity.shape[0] == synthetic.N_FRAMES * synthetic.NUMBER_OF_ANIMALS

    true_identity = identity["in_frame_index"] + 1
    chunk = identity["frame_number"] // synthetic.CHUNKSIZE
    assert (identity["identity"] == true_identity).all()
    expected_local = [synthetic.local_identity(i, c) for i, c in zip(true_identity, chunk)]
    assert identity["local_identity"].tolist() == expected_local
    # the swap between chunks is really there, otherwise this test would not test the mapping
    assert identity.loc[chunk == 1, "local_identity"].tolist() != identity.loc[chunk == 1, "identity"].tolist()


def test_concatenation(dbfile):
    concatenation = read_table(dbfile, "CONCATENATION")
    assert concatenation[["chunk", "local_identity", "identity"]].values.tolist() == [
        [0, 1, 1], [0, 2, 2], [1, 2, 1], [1, 1, 2],
    ]


def test_qc_table_parses_the_idtrackerai_logs(dbfile):
    qc = read_table(dbfile, "QC").sort_values("chunk")
    assert qc["chunk"].tolist() == synthetic.CHUNKS
    assert qc["global_fragments"].tolist() == [3, 4]
    assert qc["accuracy"].tolist() == [0.901, 0.911]
    # no fragments.npy in the synthetic sessions
    assert qc["fragments"].isna().all()


def test_snapshots_are_the_first_frame_of_each_chunk(dbfile):
    snapshots = read_table(dbfile, "IMG_SNAPSHOTS")
    assert snapshots["frame_number"].tolist() == [chunk * synthetic.CHUNKSIZE for chunk in synthetic.CHUNKS]
    for _, row in snapshots.iterrows():
        img = cv2.imdecode(np.frombuffer(row["img"], np.uint8), cv2.IMREAD_GRAYSCALE)
        expected = synthetic.make_frame(row["frame_number"])
        assert img.shape == expected.shape
        # jpeg quality 50
        assert np.abs(img.astype(int) - expected).mean() < 5


def test_roi_map(dbfile):
    roi_map = read_table(dbfile, "ROI_MAP")
    assert roi_map[["x", "y", "w", "h"]].values.tolist() == [[20, 20, synthetic.ROI_WIDTH, synthetic.ROI_WIDTH]]
    mask = cv2.imdecode(np.frombuffer(roi_map["mask"].iloc[0], np.uint8), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (synthetic.IMG_SIZE, synthetic.IMG_SIZE)
    assert mask[200, 200] > 200 and mask[5, 5] < 50


def test_var_map_declares_the_distance_variables(dbfile):
    var_map = read_table(dbfile, "VAR_MAP").set_index("var_name")["functional_type"]
    # ethoscopy divides these columns by the roi width
    assert var_map[var_map == "distance"].index.tolist() == ["x", "y"]


def test_landmarks(dbfile):
    landmarks = read_table(dbfile, "LANDMARKS")
    assert landmarks["shape"].tolist() == ["food", "notch"]
    assert landmarks["specification"].iloc[0] == "{'center': [200, 200], 'axes': [20, 20], 'angle': 0}"


@pytest.mark.xfail(strict=True, reason="write_environment_table does not reset data between chunks, so chunk 0 is inserted again with chunk 1")
def test_environment_rows_are_not_duplicated(dbfile):
    environment = read_table(dbfile, "ENVIRONMENT")
    assert not environment.duplicated().any()
    assert environment.shape[0] == 2 * len(synthetic.CHUNKS)


def test_export_at_lower_framerate_keeps_every_nth_frame_and_all_modified_frames(raw_experiment, tmp_path, monkeypatch):
    basedir = synthetic.copy_experiment(raw_experiment, str(tmp_path / "videos"))
    monkeypatch.setattr("flyhostel.data.sqlite3.idtrackerai.ListOfBlobs", synthetic.FakeListOfBlobs)
    monkeypatch.chdir(basedir)

    from flyhostel.data.sqlite3 import export_dataset
    export_dataset(os.path.join(basedir, "metadata.yaml"), synthetic.CHUNKS, framerate=15, tables=["default_and_trajectory"])
    dbfile = synthetic.dbfile_path(basedir)

    assert float(metadata(dbfile)["data_framerate"]) == 15
    frames = sorted(read_table(dbfile, "ROI_0")["frame_number"].unique())
    step = synthetic.FRAMERATE // 15
    expected = sorted(set(range(0, synthetic.N_FRAMES, step)) | {synthetic.MODIFIED_FRAME})
    assert frames == expected


@pytest.mark.xfail(strict=True, reason="init_index_table runs CREATE INDEX without IF NOT EXISTS, so updating an existing dbfile without --reset crashes")
def test_tables_can_be_rewritten_without_reset(experiment_copy, monkeypatch):
    monkeypatch.chdir(experiment_copy)
    from flyhostel.data.sqlite3 import export_dataset
    export_dataset(os.path.join(experiment_copy, "metadata.yaml"), synthetic.CHUNKS, reset=False, tables=["index"])


@pytest.mark.xfail(strict=True, reason="--tables defaults to the string 'all' instead of ['all'], so export_dataset iterates over its characters")
def test_cli_tables_default_is_a_preset_list():
    from flyhostel.data.bin.export import get_parser
    args = get_parser().parse_args(["--metadata", "metadata.yaml", "--chunks", "0"])
    assert args.tables == ["all"]


class SingleAnimalConcatenation(ConcatenationExporter):
    def __init__(self, basedir):
        self._basedir = basedir


@pytest.mark.xfail(strict=True, raises=TypeError, reason="missing comma between cmd and the row tuple: 'str' object is not callable")
def test_single_animal_concatenation_without_csv(tmp_path):
    basedir = tmp_path / "FlyHostel1" / "1X" / "2026-01-01_14-00-00"
    (basedir / "idtrackerai").mkdir(parents=True)
    dbfile = str(tmp_path / "1X.db")
    exporter = SingleAnimalConcatenation(str(basedir))
    exporter.init_concatenation_table(dbfile)
    exporter.write_concatenation_table(dbfile, chunks=[0, 1])
    assert read_table(dbfile, "CONCATENATION")["chunk"].tolist() == [0, 1]


def test_multi_animal_concatenation_without_csv_fails_loudly(tmp_path):
    basedir = tmp_path / "FlyHostel1" / "2X" / "2026-01-01_14-00-00"
    (basedir / "idtrackerai").mkdir(parents=True)
    dbfile = str(tmp_path / "2X.db")
    exporter = SingleAnimalConcatenation(str(basedir))
    exporter.init_concatenation_table(dbfile)
    with pytest.raises(FileNotFoundError):
        exporter.write_concatenation_table(dbfile, chunks=[0, 1])


def test_unknown_table_names_are_rejected(raw_experiment, tmp_path):
    exporter = IdtrackeraiExporter(os.path.join(raw_experiment, synthetic.RELPATH))
    with pytest.raises(Exception, match="Do you have a typo"):
        exporter.init_tables(str(tmp_path / "x.db"), ["ROI0"])


def test_blob_rows():
    exporter = IdtrackeraiExporter.__new__(IdtrackeraiExporter)
    blob = synthetic.make_blob(identity=1, frame_number=synthetic.CHUNKSIZE + 5)
    r0, identity, modified = exporter.add_blob(blob, id_mapping={2: 1, 1: 2})
    assert r0 == (65, 0, 165.0, 150.0, 11, 201, False, None)
    # in chunk 1, fly 1 has local identity 2
    assert identity == (65, 0, 2, 1)
    assert modified is None

    # blobs without identity (crossings) get identity 0
    blob.final_identities = [None]
    assert exporter.write_blob_identity(blob, id_mapping={2: 1, 1: 2}) == (65, 0, 0, 0)
