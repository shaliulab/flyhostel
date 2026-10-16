"""
Synthetic tests for the NEW-FRAGMENT tag (flyhostel.data.human_validation.cvat.fragments.reassign_fragments).
"""
import numpy as np
import pandas as pd

from flyhostel.data.human_validation.cvat.fragments import reassign_fragments

CHUNKSIZE = 100


def _machine(rows):
    df = pd.DataFrame(rows, columns=["frame_number", "in_frame_index", "x", "y", "fragment"])
    df["fragment"] = df["fragment"].astype(float)
    return df


def test_sequential_collision():
    # fragment 3 is fly A in frames 0-4 and fly B in frames 5-9; fragment 7 is another fly
    rows = [(fn, 0, 10.0 if fn < 5 else 80.0, 10.0, 3) for fn in range(10)]
    rows += [(fn, 1, 50.0, 50.0, 7) for fn in range(10)]
    machine = _machine(rows)
    human = pd.DataFrame({"frame_number": [2, 6], "x": [10.0, 80.0], "y": [10.0, 10.0], "fragment": [3.0, 3.0]})

    machine, human = reassign_fragments(machine, human, [(5, 3.0, 80.0, 10.0)], CHUNKSIZE)

    frag3 = machine.loc[machine["in_frame_index"] == 0].set_index("frame_number")["fragment"]
    assert (frag3.loc[:4] == 3).all()
    assert (frag3.loc[5:] == 8).all()
    assert (machine.loc[machine["in_frame_index"] == 1, "fragment"] == 7).all()
    assert human["fragment"].tolist() == [3.0, 8.0]


def test_simultaneous_collision():
    # fragment 3 is given to two flies at the same time, one on the left, one on the right
    rows = []
    for fn in range(10):
        rows.append((fn, 0, 10.0 + fn, 10.0, 3))
        rows.append((fn, 1, 90.0 - fn, 10.0, 3))
    machine = _machine(rows)
    human = pd.DataFrame({"frame_number": [4, 4], "x": [14.0, 86.0], "y": [10.0, 10.0], "fragment": [3.0, 3.0]})

    # tag the right fly in frame 2
    machine, human = reassign_fragments(machine, human, [(2, 3.0, 88.0, 10.0)], CHUNKSIZE)

    right = machine.loc[machine["in_frame_index"] == 1].set_index("frame_number")["fragment"]
    assert (right.loc[:1] == 3).all()
    assert (right.loc[2:] == 4).all()
    assert (machine.loc[machine["in_frame_index"] == 0, "fragment"] == 3).all()
    assert human["fragment"].tolist() == [3.0, 4.0]


def test_two_tags_same_fragment_and_other_chunk_untouched():
    # A -> B -> A in fragment 3 of chunk 0; chunk 1 also has a fragment 3
    rows = [(fn, 0, 80.0 if 3 <= fn < 6 else 10.0, 10.0, 3) for fn in range(10)]
    rows += [(CHUNKSIZE + fn, 0, 80.0, 10.0, 3) for fn in range(5)]
    machine = _machine(rows)
    human = pd.DataFrame(columns=["frame_number", "x", "y", "fragment"], dtype=float)

    machine, _ = reassign_fragments(machine, human, [(6, 3.0, 10.0, 10.0), (3, 3.0, 80.0, 10.0)], CHUNKSIZE)

    frag = machine.set_index("frame_number")["fragment"]
    assert frag.loc[[0, 1, 2]].tolist() == [3, 3, 3]
    assert frag.loc[[3, 4, 5]].tolist() == [4, 4, 4]
    assert frag.loc[[6, 7, 8, 9]].tolist() == [5, 5, 5, 5]
    assert (frag.loc[CHUNKSIZE:] == 3).all()


def test_tag_without_fragment_is_ignored():
    machine = _machine([(0, 0, 10.0, 10.0, 3)])
    human = pd.DataFrame({"frame_number": [0], "x": [10.0], "y": [10.0], "fragment": [3.0]})
    machine2, human2 = reassign_fragments(machine, human, [(0, None, 10.0, 10.0), (0, np.nan, 10.0, 10.0)], CHUNKSIZE)
    pd.testing.assert_frame_equal(machine, machine2)
    pd.testing.assert_frame_equal(human, human2)
