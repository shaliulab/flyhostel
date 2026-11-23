"""The CVAT project list cache (index_cvat_projects.json) follows $FLYHOSTEL_VIDEOS."""
import json
import os
import time

import flyhostel.utils.cvat as cvat


class _Response:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _Session:
    calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, **kwargs):
        type(self).calls += 1
        return _Response({"next": None, "previous": None, "results": [{"name": "FlyHostel1_1X_2025-01-01_10-00-00", "id": 7}]})


def test_projects_json_follows_flyhostel_videos(tmp_path, monkeypatch):
    monkeypatch.delenv("FLYHOSTEL_VIDEOS", raising=False)
    assert cvat.projects_json() == "/flyhostel_data/videos/index_cvat_projects.json"
    monkeypatch.setenv("FLYHOSTEL_VIDEOS", str(tmp_path))
    assert cvat.projects_json() == str(tmp_path / "index_cvat_projects.json")


def test_update_project_list_writes_cache_in_flyhostel_videos(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYHOSTEL_VIDEOS", str(tmp_path))
    monkeypatch.setattr(cvat, "_require_cvat", lambda: None)
    monkeypatch.setattr(cvat, "cvat_auth", lambda session: None)
    monkeypatch.setattr(cvat.requests, "Session", _Session)
    _Session.calls = 0

    assert cvat.get_project_id_from_name("FlyHostel1_1X_2025-01-01_10-00-00") == 7
    cache = tmp_path / "index_cvat_projects.json"
    assert json.loads(cache.read_text())["results"][0]["id"] == 7
    assert [p.name for p in tmp_path.iterdir()] == ["index_cvat_projects.json"]  # no temp file left

    # fresh cache: read from disk, not fetched again
    assert cvat.get_project_id_from_name("FlyHostel1_1X_2025-01-01_10-00-00") == 7
    assert _Session.calls == 1

    # stale cache: fetched and rewritten
    old = time.time() - 120
    os.utime(cache, (old, old))
    cvat.update_project_list()
    assert _Session.calls == 2
    assert cache.stat().st_mtime > old + 60
