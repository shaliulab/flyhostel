"""
Videos of sleep bouts, cut from the per-fly movie produced by the motionmapper pipeline:

    <basedir>/motionmapper/<NN>/<experiment>__<NN>.feather   its first frame is the start of the movie
    <basedir>/motionmapper/<NN>/movie/movie.m3u8              HLS playlist of ~10 s .ts segments

Time conventions:
    frame_number   frames since the beginning of the recording (chunk 0)
    zt             seconds since ZT0 of the first day of recording
                   = frame_time / 1000 + t_after_ref  (see flyhostel.utils.load_meta_info)
    movie_time     seconds since the start of the fly's movie
                   = (frame_number - first_chunk * chunksize) / framerate

Usage:
    loader = FlyHostelLoader(experiment="FlyHostel1_6X_2025-10-02_16-00-00", identity=1)
    loader.record_all_sleep_bouts()
    # -> <basedir>/flyhostel/videos/<fly>/<fly>_sleep_bout_t0-XXXX_t1-YYYY.mp4

The idtrackerai-validator-server ("Capture all sleep bouts" button) uses the same functions.
"""
import os
import os.path
import sqlite3
import logging
import tempfile
import subprocess
from contextlib import closing

import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Seconds of movie added before and after a sleep bout.
SLEEP_BOUT_PADDING = 5
# Two asleep rows further apart than this many seconds belong to different bouts.
SLEEP_BOUT_MAX_GAP = 2


# ── movie ───────────────────────────────────────────────────────────────────────

def fly_motionmapper_dir(basedir, identity):
    return os.path.join(basedir, "motionmapper", str(int(identity)).zfill(2))


def movie_dir(basedir, identity):
    return os.path.join(fly_motionmapper_dir(basedir, identity), "movie")


def has_movie(basedir, identity):
    return os.path.exists(os.path.join(movie_dir(basedir, identity), "movie.m3u8"))


def movie_first_chunk(basedir, experiment, identity, chunksize):
    """Chunk at which the fly's movie starts: that of the first frame in its feather file.
    None if there is no feather file."""
    fly = f"{experiment}__{str(int(identity)).zfill(2)}"
    feather_path = os.path.join(fly_motionmapper_dir(basedir, identity), f"{fly}.feather")
    if not os.path.exists(feather_path):
        return None
    frame_numbers = pd.read_feather(feather_path, columns=["frame_number"])["frame_number"]
    if not len(frame_numbers):
        return None
    return int(frame_numbers.iloc[0]) // int(chunksize)


def read_playlist(directory):
    """[(segment path, start in movie time, duration)] of <directory>/movie.m3u8.

    The playlist has #EXT-X-DISCONTINUITY tags (timestamps restart in every
    segment), so ffmpeg cannot seek in the .m3u8: segments are addressed directly."""
    segments = []
    start = 0.0
    duration = None
    with open(os.path.join(directory, "movie.m3u8")) as playlist:
        for line in playlist:
            line = line.strip()
            if line.startswith("#EXTINF:"):
                duration = float(line[len("#EXTINF:"):].split(",")[0])
            elif line and not line.startswith("#") and duration is not None:
                segments.append((os.path.join(directory, line), start, duration))
                start += duration
                duration = None
    return segments


def playlist_duration(segments):
    return segments[-1][1] + segments[-1][2] if segments else 0.0


def segments_between(segments, start, end):
    """Segments overlapping [start, end] of movie time."""
    return [seg for seg in segments if seg[1] + seg[2] > start and seg[1] <= end]


def run_ffmpeg(*args):
    """Run ffmpeg; returns its stdout, raises RuntimeError with its stderr on failure."""
    process = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode(errors="replace").strip())
    return process.stdout


def _run_ffmpeg_with_progress(args, duration, on_progress):
    """Run ffmpeg, calling on_progress(fraction) as it writes `duration` s of output."""
    process = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-progress", "pipe:1", "-nostats", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    for line in process.stdout:
        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and value.lstrip("-").isdigit() and duration > 0:
            on_progress(min(1.0, max(0.0, int(value) / 1e6 / duration)))
    stderr = process.stderr.read()
    if process.wait() != 0:
        raise RuntimeError(stderr.strip())


def cut_movie(segments, start, end, path, on_progress=None):
    """Write to `path` an MP4 spanning [start, end] s of movie time.

    Whole segments are concatenated with stream copy: fast even for clips of hours,
    but it can only cut at keyframes, so the clip may include up to one segment
    (~10 s) of extra context on each side. That guarantees [start, end] is included.

    Raises ValueError if the movie does not cover [start, end], RuntimeError if ffmpeg fails.
    """
    duration = playlist_duration(segments)
    if end < 0 or start > duration:
        raise ValueError(f"[{start:.1f}, {end:.1f}] s is not covered by the movie (0-{duration:.1f} s)")
    selected = segments_between(segments, max(0.0, start), min(duration, end))

    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False) as handle:
        handle.write("\n".join(["ffconcat version 1.0"] + [f"file '{segment}'" for segment, _, _ in selected]) + "\n")
        list_path = handle.name
    try:
        # -f mp4: `path` may not end in .mp4 (e.g. a .part file renamed when complete)
        args = ["-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", "-f", "mp4", "-movflags", "+faststart", path]
        if on_progress is None:
            run_ffmpeg(*args)
        else:
            _run_ffmpeg_with_progress(args, sum(d for _, _, d in selected), on_progress)
    finally:
        os.unlink(list_path)


# ── sleep bouts ─────────────────────────────────────────────────────────────────

def group_sleep_bouts(asleep_frames, framerate, max_gap=SLEEP_BOUT_MAX_GAP):
    """[(first_frame, end_frame)] of every sleep bout, sorted. end_frame is exclusive.

    asleep_frames: frame numbers of the rows where the fly is asleep (about one per second).
    Rows further apart than `max_gap` seconds belong to different bouts; each row
    stands for one second, so a bout ends 1 s after its last row.
    """
    bouts = []
    for frame_number in sorted(int(f) for f in asleep_frames):
        if bouts and frame_number - bouts[-1][1] <= max_gap * framerate:
            bouts[-1][1] = frame_number
        else:
            bouts.append([frame_number, frame_number])
    return [(start, int(end + framerate)) for start, end in bouts]


def find_sleep_bout(bouts, frame_number, direction="current"):
    """Pick a bout relative to `frame_number`; None if there is none.

    current: the bout ongoing at frame_number, or else the next one
    next:    the first bout starting after frame_number (skips an ongoing bout)
    prev:    the last bout starting before the ongoing bout, or before
             frame_number when the fly is awake
    """
    if direction == "current":
        return next(((s, e) for s, e in bouts if e > frame_number), None)
    if direction == "next":
        return next(((s, e) for s, e in bouts if s > frame_number), None)
    if direction == "prev":
        ongoing = next(((s, e) for s, e in bouts if s <= frame_number < e), None)
        reference = ongoing[0] if ongoing else frame_number
        return next(((s, e) for s, e in reversed(bouts) if s < reference), None)
    raise ValueError(f"direction must be current, next or prev, not {direction}")


def bout_video_name(fly, t0, t1):
    """<fly>_sleep_bout_t0-XXXX_t1-YYYY.mp4, with t0 / t1 in seconds since ZT0."""
    return f"{fly}_sleep_bout_t0-{round(t0)}_t1-{round(t1)}.mp4"


# ── FlyHostelLoader mixin ───────────────────────────────────────────────────────

class SleepVideoRecorder:
    """Adds loader.record_all_sleep_bouts() to FlyHostelLoader."""

    basedir = None
    dbfile = None
    experiment = None
    identity = None
    framerate = None
    chunksize = None
    meta_info = None
    sleep = None

    def __init__(self, *args, **kwargs):
        super(SleepVideoRecorder, self).__init__(*args, **kwargs)

    @property
    def fly(self):
        return f"{self.experiment}__{str(self.identity).zfill(2)}"

    def sleep_bouts(self, min_time_immobile=300, max_gap=SLEEP_BOUT_MAX_GAP):
        """[(first_frame, end_frame)] of every sleep bout of the fly (end exclusive).

        Reads the per-second sleep annotation (load_sleep_data with bin_size=None).
        self.sleep is left as it was.
        """
        previous = self.sleep
        try:
            self.load_sleep_data(min_time_immobile=min_time_immobile, bin_size=None, errors="warning")
            sleep = self.sleep
        finally:
            self.sleep = previous
        if sleep is None or sleep.empty or "asleep" not in sleep.columns:
            return []
        asleep = sleep.loc[sleep["asleep"] == True, "frame_number"].dropna()
        return group_sleep_bouts(asleep, float(self.framerate), max_gap=max_gap)

    def frame_to_zt(self, frame_number):
        """Seconds since ZT0 of the first day of recording at `frame_number`."""
        with closing(sqlite3.connect(f"file:{self.dbfile}?mode=ro", uri=True)) as conn:
            row = conn.execute(
                "SELECT frame_time FROM STORE_INDEX WHERE frame_number = ?", (int(frame_number),)
            ).fetchone()
        if row is None:
            raise ValueError(f"Frame {frame_number} not found in {self.dbfile}")
        return row[0] / 1000 + self.meta_info["t_after_ref"]

    def sleep_bout_video_name(self, bout):
        start, end = bout
        t0 = self.frame_to_zt(start)
        try:
            t1 = self.frame_to_zt(end)
        except ValueError:   # the bout ends with the recording
            t1 = t0 + (end - start) / float(self.framerate)
        return bout_video_name(self.fly, t0, t1)

    def sleep_videos_dir(self):
        return os.path.join(self.basedir, "flyhostel", "videos", self.fly)

    def record_sleep_bout(self, bout, output_dir=None, padding=SLEEP_BOUT_PADDING, on_progress=None,
                          _movie=None):
        """Write the movie of one sleep bout (first_frame, end_frame) to output_dir.

        Returns the path of the video. Raises ValueError if the fly's movie does not
        cover the bout, RuntimeError if ffmpeg fails.
        """
        output_dir = output_dir or self.sleep_videos_dir()
        os.makedirs(output_dir, exist_ok=True)
        segments, first_frame = _movie or self._sleep_video_movie()

        start, end = bout
        framerate = float(self.framerate)
        movie_start = (start - first_frame) / framerate - padding
        movie_end = (end - first_frame) / framerate + padding

        name = self.sleep_bout_video_name(bout)
        partial = os.path.join(output_dir, f".{name}.part")
        try:
            cut_movie(segments, movie_start, movie_end, partial, on_progress=on_progress)
        except (ValueError, RuntimeError):
            if os.path.exists(partial):
                os.unlink(partial)
            raise
        path = os.path.join(output_dir, name)
        os.replace(partial, path)
        return path

    def _sleep_video_movie(self):
        """(playlist segments, frame number at which the movie starts)"""
        if not has_movie(self.basedir, self.identity):
            raise FileNotFoundError(f"{self.fly} has no movie in {movie_dir(self.basedir, self.identity)}")
        first_chunk = movie_first_chunk(self.basedir, self.experiment, self.identity, self.chunksize)
        if first_chunk is None:
            raise FileNotFoundError(f"Cannot place frames in the movie of {self.fly}: no feather file")
        return read_playlist(movie_dir(self.basedir, self.identity)), first_chunk * int(self.chunksize)

    def record_all_sleep_bouts(self, output_dir=None, padding=SLEEP_BOUT_PADDING, max_gap=SLEEP_BOUT_MAX_GAP,
                               min_time_immobile=300, on_progress=None):
        """One video per sleep bout of the fly, cut from its motionmapper movie.

        Same as "Capture all sleep bouts" with "save on server" ticked in the
        idtrackerai-validator-server.

        Arguments:
            output_dir: default <basedir>/flyhostel/videos/<fly>/
            padding: seconds of movie added before and after each bout
            max_gap: asleep rows further apart than this (s) belong to different bouts
            min_time_immobile: passed to load_sleep_data
            on_progress: callable(bout_index, fraction) called while each video is
                         written. Default: a tqdm progress bar.

        Returns:
            paths of the videos written. Bouts the movie does not cover, or that
            ffmpeg fails on, are skipped with a warning.
        """
        bouts = self.sleep_bouts(min_time_immobile=min_time_immobile, max_gap=max_gap)
        if not bouts:
            logger.warning("%s has no sleep bouts", self.fly)
            return []
        movie = self._sleep_video_movie()

        bar = None
        if on_progress is None:
            bar = tqdm(total=len(bouts), desc=f"{self.fly} sleep bouts", unit="video",
                       bar_format="{l_bar}{bar}| {n:.2f}/{total} [{elapsed}<{remaining}]")

        written = []
        try:
            for i, bout in enumerate(bouts):
                if bar is not None:
                    progress = (lambda fraction, i=i: bar.update(i + fraction - bar.n))
                else:
                    progress = (lambda fraction, i=i: on_progress(i, fraction))
                try:
                    written.append(self.record_sleep_bout(bout, output_dir, padding, progress, _movie=movie))
                except (ValueError, RuntimeError) as error:
                    logger.warning("Skipping sleep bout %s-%s of %s: %s", bout[0], bout[1], self.fly, error)
                if bar is not None:
                    bar.update(i + 1 - bar.n)
        finally:
            if bar is not None:
                bar.close()

        if not written:
            raise RuntimeError(f"No sleep bout video could be made for {self.fly}")
        return written
