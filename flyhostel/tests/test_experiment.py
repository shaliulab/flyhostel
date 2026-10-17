import os

import pytest

from flyhostel.utils.experiment import (
    experiment_to_relpath,
    get_experiment_identifier,
    get_id_prefix,
    make_id,
    match_experiment,
    parse_experiment_name,
    split_arena,
)

SINGLE="FlyHostel1_6X_2023-05-23_14-00-00"
ARENA="FlyHostel1_4X_2026-10-13_17-00-00_arena2"


def test_relpath():
    assert experiment_to_relpath(SINGLE) == os.path.join("FlyHostel1", "6X", "2023-05-23_14-00-00")
    assert experiment_to_relpath(ARENA) == os.path.join("FlyHostel1", "4X", "2026-10-13_17-00-00", "arenas", "2")


@pytest.mark.parametrize("experiment", [SINGLE, ARENA])
def test_roundtrip(experiment):
    basedir=os.path.join("/flyhostel_data/videos", experiment_to_relpath(experiment))
    assert get_experiment_identifier(basedir) == experiment
    assert get_experiment_identifier(basedir + os.path.sep) == experiment


def test_split_arena():
    assert split_arena(SINGLE) == (SINGLE, None)
    assert split_arena(ARENA) == ("FlyHostel1_4X_2026-10-13_17-00-00", 2)


def test_parse():
    fields=parse_experiment_name(ARENA)
    assert fields["flyhostel_id"] == 1
    assert fields["number_of_animals"] == 4
    assert fields["date_time"] == "2026-10-13_17-00-00"
    assert fields["arena"] == 2
    assert parse_experiment_name(SINGLE)["arena"] is None


def test_match():
    assert match_experiment(SINGLE + "__01") == SINGLE
    assert match_experiment(ARENA + "__01") == ARENA
    assert match_experiment(ARENA + "_000010_001") == ARENA
    assert match_experiment(SINGLE + "_000010_001") == SINGLE
    assert match_experiment("split.yaml") is None


def test_ids():
    assert make_id(SINGLE, 1) == "FlyHostel1_6X_2023-05-23_1|01"
    assert make_id(ARENA, 1) == "FH1_4X_261013_17-00_arena2|01"
    assert len(get_id_prefix(SINGLE)) == len(get_id_prefix(ARENA)) == 26


def test_deg_entries():
    from flyhostel.data.deg import parse_entry, parse_experiment, parse_chunk, parse_local_identity
    entry=ARENA + "_000010_003"
    assert parse_experiment(entry) == ARENA
    assert parse_chunk(entry) == 10
    assert parse_local_identity(entry) == 3
    assert parse_experiment(SINGLE + "_000010_003") == SINGLE

    rejections=ARENA + "_rejections_141000_141100_002_03"
    ok, tokens=parse_entry(rejections, chunksize=14100)
    assert ok
    assert "_".join(tokens[:-2]) == ARENA
    assert tokens[-2:] == [10, 2]


def test_experiment_from_id():
    from flyhostel.utils.experiment import experiment_from_id
    assert experiment_from_id(make_id(ARENA, 1)) == ARENA
    assert experiment_from_id("FH1_4X_261013_17-00_arena2") == ARENA


def test_experiment_from_single_arena_id(tmp_path, caplog):
    from flyhostel.utils.experiment import experiment_from_id
    id=make_id(SINGLE, 1)
    assert experiment_from_id(id, videos_folder=str(tmp_path)) is None

    (tmp_path / "FlyHostel1" / "6X" / "2023-05-23_14-00-00").mkdir(parents=True)
    assert experiment_from_id(id, videos_folder=str(tmp_path)) == SINGLE

    (tmp_path / "FlyHostel1" / "6X" / "2023-05-23_16-00-00").mkdir(parents=True)
    assert experiment_from_id(id, videos_folder=str(tmp_path)) == "FlyHostel1_6X_2023-05-23_16-00-00"
    assert "Using the latest one" in caplog.text


def test_metadata_arena_column(caplog):
    import pandas as pd
    from googleapi.download.flyhostel import add_arena_column, filter_by_arena
    df=add_arena_column(pd.DataFrame({"flyhostel_number": [1, 1]}))
    assert (df["arena"] == 0).all()
    assert "no arena column" in caplog.text
    df=pd.DataFrame({"arena": ["1", "2"]})
    assert filter_by_arena(df, arena=2)["arena"].tolist() == ["2"]
    assert filter_by_arena(df).shape[0] == 2


def test_arena_zero_is_standard_basedir():
    from flyhostel.utils.experiment import make_experiment_name, experiment_from_id, normalize_arena
    arena0=SINGLE + "_arena0"
    assert experiment_to_relpath(arena0) == experiment_to_relpath(SINGLE)
    assert split_arena(arena0) == (SINGLE, None)
    assert parse_experiment_name(arena0)["arena"] is None
    assert make_id(arena0, 1) == make_id(SINGLE, 1)
    assert make_experiment_name(1, 6, "2023-05-23_14-00-00", arena=0) == SINGLE
    assert make_experiment_name(1, 4, "2026-10-13_17-00-00", arena=2) == ARENA
    assert experiment_from_id("FH1_6X_230523_14-00_arena0|01") == SINGLE
    assert normalize_arena(float("nan")) is None
    assert normalize_arena("2") == 2


def _arena_basedir(root, experiment):
    basedir=root / experiment_to_relpath(experiment)
    basedir.mkdir(parents=True)
    return basedir


def test_number_of_animals_in_arena(tmp_path, monkeypatch):
    import json
    import sqlite3
    from flyhostel.utils import get_identities, get_number_of_animals

    monkeypatch.setenv("FLYHOSTEL_VIDEOS", str(tmp_path))
    # the NX token of single arena experiments is the number of animals
    assert get_number_of_animals(SINGLE) == 6

    # arena 1 has 1 fly, read from its idtrackerai conf
    arena1=ARENA.replace("arena2", "arena1")
    basedir=_arena_basedir(tmp_path, arena1)
    (basedir / "2026-10-13_17-00-00.conf").write_text(json.dumps({"_number_of_animals": {"value": 1}}))
    assert get_number_of_animals(arena1) == 1
    assert get_identities(arena1) == [0]

    # arena 2 has 2 flies, read from its dbfile because there is no conf
    basedir=_arena_basedir(tmp_path, ARENA)
    with sqlite3.connect(basedir / f"{ARENA}.db") as conn:
        conn.execute("CREATE TABLE METADATA (field TEXT, value TEXT);")
        conn.execute(
            "INSERT INTO METADATA VALUES ('idtrackerai_conf', ?);",
            (json.dumps({"_number_of_animals": {"value": 2}}),)
        )
    assert get_number_of_animals(ARENA) == 2
    assert get_identities(ARENA) == [1, 2]
