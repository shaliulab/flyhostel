"""
Synthetic tests for the unassigned label (flyhostel.data.human_validation.cvat.cvat_integration.UNASSIGNED_LABEL):
boxes pre-populated by yolov7-cvat and left UNASSIGNED are ignored, those given an identity are kept.
"""
import pandas as pd

import flyhostel.data.human_validation.cvat.cvat_integration as cvat_integration

CHUNKSIZE = 14100


def _annotations(monkeypatch):
    monkeypatch.setattr(cvat_integration, "load_original_resolution", lambda basedir: [2100, 2100])
    images = pd.DataFrame({
        "id": [1, 2],
        "file_name": [f"FlyHostel1_3X_2026-09-22_14-00-00_000003_0000047803_{i:06d}.png" for i in range(2)],
    })
    categories = pd.DataFrame({"id": [1, 2, 3, 4], "name": ["1", "UNASSIGNED", "FMB", "0"]})
    box = lambda id, image_id, category_id, x: dict(
        id=id, image_id=image_id, category_id=category_id, segmentation=[], bbox=[x, 100, 20, 10])
    annotations = [
        box(1, 1, 1, 100), box(2, 1, 2, 500),  # frame 47803: identity 1 and an unassigned prediction
        box(3, 2, 2, 100), box(4, 2, 2, 500),  # frame 47804: only unassigned predictions
        box(5, 2, 3, 300),                     # frame 47804: a tag
        box(6, 2, 4, 700),                     # frame 47804: identity 0, the fly of a single-fly experiment
    ]
    return cvat_integration.load_task_annotations(annotations, images, categories, basedir=None, chunksize=CHUNKSIZE)


def test_unassigned_predictions_are_ignored(monkeypatch):
    parsed, contours = _annotations(monkeypatch)
    assert len(parsed) == len(contours) == 3
    identity = parsed.loc[parsed["local_identity"].notna()]
    assert identity["frame_number"].tolist() == [47803, 47804]
    assert identity["local_identity"].tolist() == [1, 0]
    assert parsed.loc[parsed["text"].notna(), "text"].tolist() == ["FMB"]
