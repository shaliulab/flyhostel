"""
Synthetic tests telling apart a fly walking into a COURTSHIP box from a spurious identity of the copulating blob
"""
import pandas as pd

from flyhostel.data.human_validation.cvat.courtship import (
    report_intruders_in_courtship_boxes,
    remove_blobs_associated_to_courtship,
)

CHUNKSIZE=1000
PIXELS_PER_MM=10
FRAMERATE=10
BOX=(100.0, 100.0, 200.0, 200.0)


def _intervals():
    x1, y1, x2, y2=BOX
    return pd.DataFrame({
        "interval_id": 1, "frame_number": range(100, 200),
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
    })


def _row(fn, lid, x, y):
    x1, y1, x2, y2=BOX
    inside=x1 <= x <= x2 and y1 <= y <= y2 and 100 <= fn < 200
    return {
        "interval_id": 1 if inside else None, "frame_number": fn, "local_identity": lid,
        "x": x, "y": y, "courtship": inside, "validated_fragment": False, "fragment": 1.0,
    }


def _data():
    rows=[]
    for fn in range(90, 210):
        # 2 and 3 are the courting pair
        rows += [_row(fn, 2, 150.0, 150.0), _row(fn, 3, 150.0, 150.0)]
        # 4 is another fly, far from the box, but its identity jumps onto the copulating blob in frames 120-124
        rows.append(_row(fn, 4, 150.0, 150.0) if 120 <= fn < 125 else _row(fn, 4, 600.0, 600.0))
        # 5 walks into the box (frames 140-149) coming from and going back to its edge
        if 135 <= fn < 155:
            rows.append(_row(fn, 5, 190.0 if 140 <= fn < 150 else 205.0, 150.0))
        # the copulating blob is also tracked as an unknown fly
        if 160 <= fn < 163:
            rows.append(_row(fn, 0, 150.0, 150.0))
    return pd.DataFrame(rows)


def test_classify_and_discard_intruders(tmp_path, monkeypatch):
    # remove_blobs_associated_to_courtship writes courtship_discarded.csv to the working directory
    monkeypatch.chdir(tmp_path)
    data=_data()
    engaged={1: {"engaged_per_chunk": {0: {2, 3}}}}

    runs=report_intruders_in_courtship_boxes(
        data, engaged, CHUNKSIZE, path=tmp_path / "intruders.csv",
        intervals=_intervals(), pixels_per_mm=PIXELS_PER_MM, framerate=FRAMERATE,
    )
    kinds=dict(zip(runs["local_identity"], runs["kind"]))
    assert kinds == {0: "unidentified", 4: "spurious", 5: "intruder"}

    out=remove_blobs_associated_to_courtship(data, engaged, CHUNKSIZE, intruders=runs)
    inside=out.loc[out["courtship"] == True]
    assert not (inside["local_identity"] == 0).any()
    assert not (inside["local_identity"] == 4).any()
    # the fly that walked in is kept
    assert (inside["local_identity"] == 5).sum() == 10
    # identity 4 is kept outside the box
    assert (out["local_identity"] == 4).sum() == 120 - 5
