
import os
import json
import cv2
import hashlib
import logging
import pandas as pd

from imgstore.interface import VideoCapture
from idtrackerai_validator_server.backend import (
    draw_frame,
    annotate_frame
)

logger = logging.getLogger(__name__)


def md5_checksum(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Stream MD5 to avoid reading large videos into RAM."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


class BaseManifest:
    header: list[dict] = [{"version": "1.0"}]  # subclasses extend
    manifest_type: str = "unknown"

    def _write_jsonl(self, jsonl_path: str, records: list[dict]) -> None:
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")


class VideoManifestWriter(BaseManifest):
    header = [{"version": "1.0"}, {"type": "video"}]
    manifest_type = "video"

    def __init__(self, video_path: str, writer: cv2.VideoWriter, compute_checksum: bool = False):
        self.video_path = video_path
        self.writer = writer
        self.compute_checksum = compute_checksum
        self.width = None
        self.height = None
        self._released = False

    def write(self, frame):
        if self.height is None:
            self.height, self.width = frame.shape[:2]
        self.writer.write(frame)

    def release(self):
        if self._released:
            return
        self._released = True

        self.writer.release()
        self._write_manifest()

    def _write_manifest(self):
        checksum = md5_checksum(self.video_path) if self.compute_checksum and os.path.exists(self.video_path) else 0
        item = {
            "name": os.path.splitext(os.path.basename(self.video_path))[0],
            "extension": os.path.splitext(self.video_path)[1],
            "width": self.width,
            "height": self.height,
            "meta": {"related_images": []},
            "checksum": checksum,
        }
        self._write_jsonl(self.video_path + ".jsonl", self.header + [item])


class ImageSequenceWriter(BaseManifest):
    header = [{"version": "1.0"}, {"type": "images"}]
    manifest_type = "images"

    def __init__(
        self,
        basedir: str,
        fn0: int,
        extension: str,
        compute_checksum: bool = True,
        zfill: int = 10,
    ):
        self.basedir = basedir
        self.fn0 = int(fn0)
        self.extension = extension
        self.compute_checksum = compute_checksum
        self.zfill = zfill

        os.makedirs(self.basedir, exist_ok=True)

        self.width = None
        self.height = None
        self._count = 0
        self._items: list[dict] = []
        self._released = False

    def write(self, frame):
        if self.height is None:
            self.height, self.width = frame.shape[:2]

        # Use deterministic filenames
        frame_id = self.fn0 + self._count
        fname = f"{os.path.basename(self.basedir.rstrip(os.path.sep))}_{str(frame_id).zfill(self.zfill)}{self.extension}"
        dest_path = os.path.join(self.basedir, fname)

        ok = cv2.imwrite(dest_path, frame)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {dest_path}")

        checksum = md5_checksum(dest_path) if self.compute_checksum else 0
        name_no_ext, _ = os.path.splitext(fname)

        # Match your earlier convention: name includes folder + stem
        self._items.append(
            {
                "name": os.path.join(os.path.basename(self.basedir), name_no_ext),
                "extension": self.extension,
                "width": self.width,
                "height": self.height,
                "meta": {"related_images": []},
                "checksum": checksum,
            }
        )

        self._count += 1

    def release(self):
        if self._released:
            return
        self._released = True

        jsonl_path = os.path.join(os.path.dirname(self.basedir), os.path.basename(self.basedir) + ".jsonl")
        self._write_jsonl(jsonl_path, self.header + self._items)


def generate_validation_video(
    store_path, row, df, number_of_animals, chunksize, framerate,
    output_folder=".", format=".mp4", field="identity",
    compute_checksum=False,
):
    vw = None
    cap = None
    output_video = None
    output_path_csv = None

    try:
        cap = VideoCapture(store_path, 50)
        frame_number_0 = int(row["frame_number"])
        frame_number_last = int(df["frame_number"].iloc[-1])
        experiment = row["experiment"]

        cap.set(1, frame_number_0)

        chunk = frame_number_0 // chunksize
        output_path_csv = os.path.join(output_folder, f"{experiment}_{str(chunk).zfill(6)}_{str(frame_number_0).zfill(10)}")

        if format == ".mp4":
            output_path_csv += ".mp4.csv"
        else:
            output_path_csv += ".csv"

        accum = 0

        while True:
            ret, frame = cap.read()
            if not ret or cap.frame_number > frame_number_last:
                break

            frame_number = cap.frame_number
            tracking_data = df.loc[df["frame_number"] == frame_number]

            frame = draw_frame(frame, tracking_data, number_of_animals, field=field)
            frame = annotate_frame(frame, row)

            if vw is None:
                suffix = f"{str(frame_number_0 // chunksize).zfill(6)}_{str(frame_number_0).zfill(10)}"

                if format == ".mp4":
                    output_video = os.path.join(output_folder, f"{experiment}_{suffix}.mp4")
                    logger.debug("Saving %s", output_video)

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # safer for mp4 container
                    writer = cv2.VideoWriter(
                        output_video,
                        fourcc,
                        framerate,
                        frameSize=(frame.shape[1], frame.shape[0]),  # (width, height)
                        isColor=True,
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"cv2.VideoWriter failed to open: {output_video}")

                    vw = VideoManifestWriter(output_video, writer, compute_checksum=compute_checksum)

                elif format in [".png", ".jpg"]:
                    output_video = os.path.join(output_folder, f"{experiment}_{suffix}")
                    logger.debug("Saving %s", output_video)

                    vw = ImageSequenceWriter(
                        basedir=output_video,
                        fn0=frame_number_0,
                        extension=format,
                        compute_checksum=compute_checksum,
                    )
                else:
                    raise ValueError(f"Unsupported format: {format}")

            vw.write(frame)
            accum += 1

        logger.info("Wrote %s frames", accum)
        row["nframes"]=accum

        # Write metadata CSV if we produced something meaningful
        if output_video is not None and (os.path.exists(output_video) or os.path.isdir(output_video)):
            pd.DataFrame([row]).to_csv(output_path_csv, index=False)

    except Exception:
        logger.exception("Failed generating validation video for store_path=%s", store_path)

    finally:
        try:
            if vw is not None:
                vw.release()
        finally:
            if cap is not None:
                cap.release()

    return output_video