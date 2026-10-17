"""
Tests for the production of single animal .mp4 videos used for pose estimation (fh-make-video, flyhostel.data.video).

The videos are made from the synthetic recording and dbfile (synthetic_experiment.py) and every frame is checked
to be a crop of the raw frame centered on the fly with that local identity.
"""
import os

import cv2
import numpy as np
import pandas as pd
import pytest

from flyhostel.data.bin.video import get_parser, main, save_csv
from flyhostel.data.video.reader import MP4Reader
from flyhostel.tests import synthetic_experiment as synthetic
from flyhostel.utils.filters import one_pass_filter_1d

HALF = synthetic.SQUARE_WIDTH // 2


def make_videos(basedir, monkeypatch, chunks, n_jobs=1):
    monkeypatch.chdir(basedir)
    args = get_parser().parse_args([
        "--experiment", synthetic.EXPERIMENT, "--dataset", synthetic.EXPERIMENT + ".db",
        "--n-jobs", str(n_jobs), "--chunks", *[str(chunk) for chunk in chunks],
    ])
    main(args)


def video_path(basedir, local_identity, chunk, ext=".mp4"):
    return os.path.join(basedir, "flyhostel", "single_animal", str(local_identity).zfill(3), str(chunk).zfill(6) + ext)


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames)


def expected_crop(identity, frame_number):
    x, y = synthetic.fly_xy(identity, frame_number)
    return synthetic.make_frame(frame_number)[y - HALF:y + HALF, x - HALF:x + HALF]


@pytest.fixture
def videos_chunk_1(experiment_copy, monkeypatch):
    make_videos(experiment_copy, monkeypatch, chunks=[1])
    return experiment_copy


def test_one_video_per_local_identity_with_a_full_chunk(videos_chunk_1):
    for local_identity in (1, 2):
        cap = cv2.VideoCapture(video_path(videos_chunk_1, local_identity, 1))
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == synthetic.CHUNKSIZE
        assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (synthetic.SQUARE_WIDTH,) * 2
        cap.release()

        # progress file used to skip chunks that are done
        with open(video_path(videos_chunk_1, local_identity, 1, ".txt")) as handle:
            assert int(handle.read()) == synthetic.CHUNKSIZE

        index = np.load(video_path(videos_chunk_1, local_identity, 1, ".npz"))
        assert index["frame_number"].tolist() == list(range(synthetic.CHUNKSIZE, 2 * synthetic.CHUNKSIZE))


def test_every_frame_is_centered_on_the_fly_with_that_local_identity(videos_chunk_1):
    chunk = 1
    for local_identity in (1, 2):
        # local identities swap in chunk 1: local identity 1 is fly 2
        identity = 3 - local_identity
        assert synthetic.local_identity(identity, chunk) == local_identity

        frames = read_video(video_path(videos_chunk_1, local_identity, chunk))
        for i, frame in enumerate(frames):
            frame_number = chunk * synthetic.CHUNKSIZE + i
            # the fly is a dark disk in the center of the crop
            assert frame[HALF - 2:HALF + 2, HALF - 2:HALF + 2].mean() < 80
            # and the rest of the crop matches the raw frame (mp4 is lossy)
            assert np.abs(frame.astype(int) - expected_crop(identity, frame_number)).mean() < 6


@pytest.mark.xfail(strict=True, reason="init_video_writer opens the imgstore in 'w' mode (truncating the mp4) before "
                                       "reading the .txt progress file, and the writer stays registered, so the rest of "
                                       "the chunk is written again: the video ends up with chunksize-1 frames")
def test_rerunning_a_finished_chunk_keeps_its_video(videos_chunk_1, monkeypatch):
    path = video_path(videos_chunk_1, 1, 1)
    make_videos(videos_chunk_1, monkeypatch, chunks=[1])
    assert read_video(path).shape[0] == synthetic.CHUNKSIZE
    with open(video_path(videos_chunk_1, 1, 1, ".txt")) as handle:
        assert int(handle.read()) == synthetic.CHUNKSIZE


def test_save_csv_writes_the_coordinates_used_to_crop(experiment_copy, monkeypatch):
    monkeypatch.chdir(experiment_copy)
    args = get_parser().parse_args([
        "--experiment", synthetic.EXPERIMENT, "--dataset", synthetic.EXPERIMENT + ".db", "--chunks", "1",
    ])
    save_csv(args)
    for local_identity in (1, 2):
        csv = pd.read_csv(video_path(experiment_copy, local_identity, 1, ".csv"))
        identity = 3 - local_identity
        assert csv["frame_number"].tolist() == list(range(synthetic.CHUNKSIZE, 2 * synthetic.CHUNKSIZE))
        expected = [synthetic.fly_xy(identity, fn) for fn in csv["frame_number"]]
        assert list(zip(csv["x"], csv["y"])) == expected


@pytest.mark.xfail(strict=True, reason="imgstore's writer seeds the store with the first frame when first_chunk=0, "
                                       "so the chunk 0 video has 1 frame. validate_video never runs because "
                                       "_make_single_video always returns capfn=None")
def test_video_of_chunk_0_has_a_full_chunk(experiment_copy, monkeypatch):
    make_videos(experiment_copy, monkeypatch, chunks=[0])
    assert read_video(video_path(experiment_copy, 1, 0)).shape[0] == synthetic.CHUNKSIZE


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="_make_single_video closes the writers after each chunk but does not reset them, "
                          "so with --n-jobs 1 the second chunk is written to a closed store")
def test_several_chunks_in_a_single_process(experiment_copy, monkeypatch):
    make_videos(experiment_copy, monkeypatch, chunks=[0, 1])


@pytest.fixture
def reader():
    """MP4Reader whose cropping logic can be used without a recording"""
    reader = MP4Reader.__new__(MP4Reader)
    reader.width = reader.height = 20
    reader.background_color = 255
    return reader


def numbered_image(height=60, width=80):
    return (np.arange(height * width) % 250).reshape(height, width).astype(np.uint8)


@pytest.mark.parametrize("centroid", [
    (40, 30),       # inside
    (0, 0),         # top left corner
    (79, 59),       # bottom right corner
    (5, 50),        # left edge
    (75, 3),        # top edge
    (-5, 30),       # outside, as can happen with bad tracking
])
def test_crop_is_always_square_and_centered_on_the_centroid(reader, centroid):
    img = numbered_image()
    crop = reader.crop_image(img, centroid)
    assert crop.shape == (20, 20)

    x, y = centroid
    for row in range(20):
        for col in range(20):
            src_y, src_x = y - 10 + row, x - 10 + col
            if 0 <= src_y < img.shape[0] and 0 <= src_x < img.shape[1]:
                assert crop[row, col] == img[src_y, src_x]
            else:
                assert crop[row, col] == reader.background_color


def test_crop_larger_than_the_frame(reader):
    reader.width = reader.height = 100
    crop = reader.crop_image(numbered_image(), (40, 30))
    assert crop.shape == (100, 100)
    assert crop[0, 0] == crop[-1, -1] == 255


def test_one_pass_filter_removes_one_bout_jitter_only():
    # a bout surrounded by two bouts of the same value is overwritten
    assert one_pass_filter_1d([10, 10, 11, 10, 10]).tolist() == [10] * 5
    assert one_pass_filter_1d([10, 10, 11, 11, 11, 10]).tolist() == [10] * 6
    # real movement is untouched
    assert one_pass_filter_1d([10, 11, 12, 13]).tolist() == [10, 11, 12, 13]
    assert one_pass_filter_1d([10, 10, 12, 12]).tolist() == [10, 10, 12, 12]
    # values are rounded to whole pixels
    assert one_pass_filter_1d([10.4, 10.6, 11.2]).tolist() == [10, 11, 11]
