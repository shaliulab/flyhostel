import datetime

import pandas as pd

from flyhostel.data.human_validation.cvat.cvat_integration import drop_frames_annotated_by_newer_tasks

CHUNKSIZE = 1000
OLD = datetime.datetime(2026, 1, 1, 10)
NEW = datetime.datetime(2026, 1, 1, 12)


def task(frames_annotated, image_frames, mtime):
    df = pd.DataFrame({"frame_number": frames_annotated, "idx": range(len(frames_annotated))})
    return (df, [None] * len(df), mtime, set(image_frames))


def kept_frames(result, i):
    return sorted(result[i][0]["frame_number"].tolist())


def test_unannotated_newer_task_does_not_override_annotations():
    # task 1 annotates frames 10-12; task 2 is newer and shows them, but annotates only 13-14
    out = [
        task([10, 11, 12], [10, 11, 12], OLD),
        task([13, 14], range(10, 15), NEW),
    ]
    result = drop_frames_annotated_by_newer_tasks(out, [1, 2], CHUNKSIZE)
    assert kept_frames(result, 0) == [10, 11, 12]
    assert kept_frames(result, 1) == [13, 14]


def test_newer_task_wins_when_both_annotate():
    out = [
        task([10, 11], [10, 11], OLD),
        task([10, 11], [10, 11], NEW),
    ]
    result = drop_frames_annotated_by_newer_tasks(out, [1, 2], CHUNKSIZE)
    assert kept_frames(result, 0) == []
    assert kept_frames(result, 1) == [10, 11]


def test_task_id_breaks_ties():
    out = [
        task([10], [10], NEW),
        task([10], [10], NEW),
    ]
    result = drop_frames_annotated_by_newer_tasks(out, [5, 7], CHUNKSIZE)
    assert kept_frames(result, 0) == []
    assert kept_frames(result, 1) == [10]
