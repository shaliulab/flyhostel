"""
Build a tiny but complete FlyHostel recording on disk, so the pipeline can be run end to end without real data.

Two flies are filmed for three chunks of CHUNKSIZE frames at FRAMERATE fps, and the first two are tracked
(like a real recording, the last chunk is not analysed):
identity 1 walks to the right along y=FLY1_Y, identity 2 sits still at FLY2_XY.
idtrackerai assigns local identities per chunk, and they swap between chunks
(chunk 0: local 1 -> identity 1, chunk 1: local 1 -> identity 2), so code that confuses
local_identity, in_frame_index and identity gives wrong answers.

The recording folder contains what the dbfile exporter expects to find:
metadata.yaml + mp4 chunks + index.db (written by imgstore), the idtrackerai .conf and .toml,
a .pfs camera file, metadata.csv, landmarks.toml, chunk snapshots (.png), environment
data (.extra.json), idtrackerai session folders and concatenation-overlap.csv.
"""
import json
import os
import pickle
import shutil
import sqlite3
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

FLYHOSTEL_ID = 1
NUMBER_OF_ANIMALS = 2
DATE_TIME = "2026-01-01_14-00-00"
EXPERIMENT = f"FlyHostel{FLYHOSTEL_ID}_{NUMBER_OF_ANIMALS}X_{DATE_TIME}"
RELPATH = os.path.join(f"FlyHostel{FLYHOSTEL_ID}", f"{NUMBER_OF_ANIMALS}X", DATE_TIME)

FRAMERATE = 150
CHUNKSIZE = 60
RECORDED_CHUNKS = [0, 1, 2]
CHUNKS = [0, 1]
N_RECORDED_FRAMES = CHUNKSIZE * len(RECORDED_CHUNKS)
N_FRAMES = CHUNKSIZE * len(CHUNKS)
IMG_SIZE = 400
# < 250 px/cm makes get_square_width return 100 px
PIXELS_PER_CM = 150
SQUARE_WIDTH = 100
BACKGROUND = 220
FLY_INTENSITY = 30
FLY_RADIUS = 8
ROI = [[20, 20], [380, 20], [380, 380], [20, 380]]
# cv2.boundingRect of ROI is inclusive of both edges
ROI_WIDTH = 361
# area excluded from the ROI (negative ROI), away from both flies
NEGATIVE_ROI = [[300, 40], [360, 40], [360, 80], [300, 80]]

# store created at 14:00 UTC, the time in the folder name (DATE_TIME), from which frame_time counts.
# lights on (ZT0) at 11:00 UTC
CREATED_UTC = "2026-01-01T14:00:00"
REFERENCE_HOUR = 11
T_AFTER_REF = (14 - REFERENCE_HOUR) * 3600

FLY1_Y = 150
FLY2_XY = (300, 280)
# off the grid of frames kept when exporting at a lower framerate
MODIFIED_FRAME = 73


def fly_xy(identity, frame_number):
    if identity == 1:
        return (100 + frame_number, FLY1_Y)
    return FLY2_XY


def local_identity(identity, chunk):
    """idtrackerai local identity of each fly; it swaps between chunks"""
    if chunk % 2 == 0:
        return identity
    return 3 - identity


def in_frame_index(identity):
    """Blobs are listed left to right and fly 1 is always left of fly 2"""
    return identity - 1


def make_frame(frame_number):
    img = np.full((IMG_SIZE, IMG_SIZE), BACKGROUND, np.uint8)
    for identity in (1, 2):
        cv2.circle(img, fly_xy(identity, frame_number), FLY_RADIUS, FLY_INTENSITY, -1)
    return img


def frame_time(frame_number):
    """ms since the start of the recording"""
    return round(1000 * frame_number / FRAMERATE)


def write_recording(basedir):
    """
    Write an imgstore (metadata.yaml, one .mp4 + .npz per chunk) as the FlyHostel recorder leaves it.
    The files are written directly rather than with imgstore's writer, whose first frame seeds the store
    """
    import imgstore
    for chunk in RECORDED_CHUNKS:
        frame_numbers = list(range(chunk * CHUNKSIZE, (chunk + 1) * CHUNKSIZE))
        path = os.path.join(basedir, str(chunk).zfill(6))
        writer = cv2.VideoWriter(path + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), FRAMERATE, (IMG_SIZE, IMG_SIZE), isColor=False)
        for frame_number in frame_numbers:
            writer.write(make_frame(frame_number))
        writer.release()
        np.savez(path + ".npz", frame_number=frame_numbers, frame_time=[frame_time(fn) for fn in frame_numbers])

    metadata = {"__store": {
        "chunksize": CHUNKSIZE, "class": "VideoImgStore", "created_utc": CREATED_UTC, "encoding": None,
        "extension": ".mp4", "format": "mp4v/mp4", "framerate": FRAMERATE, "imgdtype": "uint8",
        "imgshape": [IMG_SIZE, IMG_SIZE], "timezone_local": "Etc/UTC", "uuid": "0" * 32, "version": 2,
        "pixels_per_cm": PIXELS_PER_CM,
    }}
    metadata_path = os.path.join(basedir, "metadata.yaml")
    with open(metadata_path, "w") as handle:
        yaml.safe_dump(metadata, handle)

    # imgstore builds index.db the first time the store is opened for reading
    imgstore.new_for_filename(metadata_path).close()
    assert os.path.exists(os.path.join(basedir, "index.db"))


def write_idtrackerai_conf(basedir):
    """DATE_TIME.toml (idtrackerai v5) and DATE_TIME.conf (made from it by conf-convert)"""
    with open(os.path.join(basedir, f"{DATE_TIME}.toml"), "w") as handle:
        handle.write(f"number_of_animals = {NUMBER_OF_ANIMALS}\nroi_list = [\n")
        handle.write(f'    "+ Polygon {[[float(v) for v in point] for point in ROI]}",\n')
        handle.write(f'    "- Polygon {[[float(v) for v in point] for point in NEGATIVE_ROI]}",\n')
        handle.write("]\n")

    conf = {
        "_number_of_animals": {"value": NUMBER_OF_ANIMALS},
        "_roi": {"value": [
            [", ".join(str(point) for point in ROI) + ", "],
            ["- " + ", ".join(str(point) for point in NEGATIVE_ROI) + ", "],
        ]},
    }
    with open(os.path.join(basedir, f"{DATE_TIME}.conf"), "w") as handle:
        json.dump(conf, handle)


def ethoscope_metadata_rows():
    """One row per fly as found in the FlyHostel metadata sheet, plus a fly of another experiment"""
    header = [
        "", "machine_id", "machine_name", "date", "flyhostel_number", "number_of_animals",
        "flyhostel_date", "flyhostel_time", "identity", "region_id", "reference_hour", "genotype", "sex",
    ]
    rows = [
        [str(i), f"FlyHostel{FLYHOSTEL_ID}", f"{NUMBER_OF_ANIMALS}X", "2026-01-01",
         str(FLYHOSTEL_ID), str(NUMBER_OF_ANIMALS), "2026-01-01", "14-00-00", str(identity), "NONE",
         str(REFERENCE_HOUR), "CS", "FEM"]
        for i, identity in enumerate((1, 2))
    ]
    rows.append(["2", "FlyHostel9", "1X", "2025-01-01", "9", "1", "2025-01-01", "10-00-00",
                 "NONE", "NONE", "8", "CS", "FEM"])
    return header, rows


def write_side_files(basedir):
    with open(os.path.join(basedir, "camera.pfs"), "w") as handle:
        handle.write("# camera settings\n")

    header, rows = ethoscope_metadata_rows()
    with open(os.path.join(basedir, "metadata.csv"), "w") as handle:
        handle.write("\n".join(",".join(row) for row in [header] + rows[:2]) + "\n")

    with open(os.path.join(basedir, "landmarks.toml"), "w") as handle:
        handle.write('notch_count = 1\nroi_list = [\n')
        handle.write("""  "0 Ellipse {'center': [200, 200], 'axes': [20, 20], 'angle': 0}",\n""")
        handle.write('  "1 Polygon [[12.0, 190.0], [30.0, 190.0], [30.0, 210.0], [12.0, 210.0]]",\n')
        handle.write(']\n')

    for chunk in CHUNKS:
        cv2.imwrite(os.path.join(basedir, f"{str(chunk).zfill(6)}.png"), make_frame(chunk * CHUNKSIZE))
        environment = [
            {"frame_number": chunk * CHUNKSIZE + i, "camera_temperature": 40.0, "temperature": 25.0 + chunk,
             "humidity": 60.0, "light": 1000.0}
            for i in (0, CHUNKSIZE // 2)
        ]
        with open(os.path.join(basedir, f"{str(chunk).zfill(6)}.extra.json"), "w") as handle:
            json.dump(environment, handle)


def make_blob(identity, frame_number):
    chunk = frame_number // CHUNKSIZE
    modified = identity == 1 and frame_number == MODIFIED_FRAME
    blob = SimpleNamespace(
        frame_number=frame_number, in_frame_index=in_frame_index(identity),
        centroid=tuple(float(v) for v in fly_xy(identity, frame_number)),
        area=np.pi * FLY_RADIUS ** 2, modified=modified,
        fragment_identifier=10 * chunk + identity,
        final_identities=[local_identity(identity, chunk)], chunk=chunk,
    )
    if modified:
        blob._annotation = {"class": "fly"}
    return blob


class FakeListOfBlobs:
    """Stand-in for idtrackerai.list_of_blobs.ListOfBlobs, loaded from a pickle"""

    def __init__(self, blobs_in_video):
        self.blobs_in_video = blobs_in_video

    @classmethod
    def load(cls, path):
        with open(path, "rb") as handle:
            return cls(pickle.load(handle))


def write_idtrackerai_output(basedir):
    idtrackerai = os.path.join(basedir, "idtrackerai")
    rows = []
    for chunk in CHUNKS:
        session = os.path.join(idtrackerai, f"session_{str(chunk).zfill(6)}")
        os.makedirs(os.path.join(session, "preprocessing"))
        blobs_in_video = [
            [make_blob(identity, frame_number) for identity in (1, 2)]
            for frame_number in range(chunk * CHUNKSIZE, (chunk + 1) * CHUNKSIZE)
        ]
        with open(os.path.join(session, "preprocessing", "blobs_collection.npy"), "wb") as handle:
            pickle.dump(blobs_in_video, handle)

        video_object = SimpleNamespace(episodes_start_end=[(0, CHUNKSIZE // 2), (CHUNKSIZE // 2, CHUNKSIZE)])
        np.save(os.path.join(session, "video_object.npy"), np.array(video_object, dtype=object))

        with open(os.path.join(idtrackerai, f"session_{str(chunk).zfill(6)}_tracking_output.txt"), "w") as handle:
            handle.write(f"total number of global_fragments: {3 + chunk}\nEstimated accuracy: 0.9{chunk}12\n")

        for identity in (1, 2):
            rows.append((chunk, in_frame_index(identity), in_frame_index(identity),
                         local_identity(identity, chunk), local_identity(identity, chunk + 1), 0, 0, identity))

    with open(os.path.join(idtrackerai, "concatenation-overlap.csv"), "w") as handle:
        handle.write(",chunk,in_frame_index_before,in_frame_index_after,local_identity,local_identity_after,is_inferred,is_broken,identity\n")
        for i, row in enumerate(rows):
            handle.write(",".join(str(e) for e in (i,) + row) + "\n")


def build_raw_experiment(root):
    """Write the raw recording of EXPERIMENT under root (a FLYHOSTEL_VIDEOS folder). Returns its basedir"""
    basedir = os.path.join(root, RELPATH)
    os.makedirs(basedir)
    write_recording(basedir)
    write_idtrackerai_conf(basedir)
    write_side_files(basedir)
    write_idtrackerai_output(basedir)
    return basedir


def dbfile_path(basedir):
    return os.path.join(basedir, EXPERIMENT + ".db")


def export_dbfile(basedir, monkeypatch, tables=("default_and_trajectory",)):
    """Run the exporter (fh export) in basedir and return the path to the dbfile"""
    import flyhostel.data.sqlite3.idtrackerai as idtrackerai_exporter
    from flyhostel.data.sqlite3 import export_dataset

    monkeypatch.setattr(idtrackerai_exporter, "ListOfBlobs", FakeListOfBlobs)
    monkeypatch.chdir(basedir)
    export_dataset(os.path.join(basedir, "metadata.yaml"), CHUNKS, reset=True, tables=list(tables))
    return dbfile_path(basedir)


def validate_dbfile(basedir):
    """Emulate the output of human validation: _VAL copies of the tables and an annotations zip"""
    with sqlite3.connect(dbfile_path(basedir)) as conn:
        for table in ("ROI_0", "IDENTITY", "CONCATENATION"):
            conn.execute(f"CREATE TABLE {table}_VAL AS SELECT * FROM {table};")
    os.makedirs(os.path.join(basedir, "flyhostel", "validation"), exist_ok=True)
    open(os.path.join(basedir, "flyhostel", "validation", f"{EXPERIMENT}_annotations.zip"), "w").close()


def copy_experiment(src_root, dst_root):
    shutil.copytree(src_root, dst_root, symlinks=True)
    return os.path.join(dst_root, RELPATH)
