"""
Synthetic tests for the report of CVAT frames that still have problems after validation
"""
import pandas as pd

from flyhostel.data.human_validation.cvat.report import find_problem_frames, find_problem_blocks


def _out(rows):
    return pd.DataFrame(rows, columns=["frame_number", "identity", "x", "y"])


def test_find_problem_frames():
    rows=[]
    for fn in range(10):
        rows.append((fn, 1, 10.0, 10.0))
        # identity 2 is missing in frame 3 and duplicated in frame 4
        if fn != 3:
            rows.append((fn, 2, 50.0, 50.0))
        if fn == 4:
            rows.append((fn, 2, 60.0, 60.0))
    # identity 1 jumps in frame 7, and a blob has no identity in frame 8
    rows=[(fn, i, 90.0 if (fn == 7 and i == 1) else x, y) for fn, i, x, y in rows]
    rows.append((8, None, 0.0, 0.0))

    problems=find_problem_frames(_out(rows), number_of_animals=2, max_jump=5)
    found=set(map(tuple, problems.values.tolist()))
    assert found == {
        (3, "identity 2 missing"),
        (4, "identity 2 duplicated"),
        (7, "identity 1 jumps"),
        (8, "identity 1 jumps"),
        (8, "blob without identity"),
    }


def test_find_problem_blocks():
    # task 1 has frames 100-104 (task frames 0-4), task 2 has frames 200-201
    cvat_frames={100+i: [1, 11, i] for i in range(5)}
    cvat_frames.update({200+i: [2, 21, i] for i in range(2)})
    problems=pd.DataFrame({
        "frame_number": [101, 102, 102, 104, 200, 201, 500],
        "problem": ["identity 1 missing", "identity 1 missing", "identity 2 jumps", "no flies", "no flies", "no flies", "no flies"],
    })
    blocks=find_problem_blocks(problems, cvat_frames)
    assert blocks[["task", "job", "task_frame", "first_frame_number", "last_frame_number", "n_frames"]].values.tolist() == [
        [1, 11, 1, 101, 102, 2],
        [1, 11, 4, 104, 104, 1],
        [2, 21, 0, 200, 201, 2],
    ]
    assert blocks["problems"].iloc[0] == "identity 1 missing (2 frames), identity 2 jumps (1 frame)"


def test_no_problem_blocks():
    blocks=find_problem_blocks(pd.DataFrame({"frame_number": [], "problem": []}), {1: [1, 1, 0]})
    assert blocks.shape[0] == 0
