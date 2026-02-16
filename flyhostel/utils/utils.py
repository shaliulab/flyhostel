import time
import json
import logging
import os.path
import yaml
import re
import shutil
import glob
import sys
import sqlite3
import joblib
from confapp import conf
import numpy as np
import pandas as pd
from matplotlib import cm

logger=logging.getLogger(__name__)

if sys.version_info > (3, 8):
    import pickle
else:
    try:
        import pickle
    except:
        logger.warning("Python version < 3.8 detected. Loading pickle5 instead of pickle")
        import pickle5 as pickle


from flyhostel.constants import CONFIG_FILE, DEFAULT_CONFIG, ANALYSIS_FOLDER
from flyhostel.quantification.constants import TRAJECTORIES_SOURCE
logger = logging.getLogger(__name__)


def get_spaced_colors_util(n, norm=False, black=True, cmap="jet"):
    RGB_tuples = cm.get_cmap(cmap)
    if norm:
        colors = [RGB_tuples(i / n) for i in range(n)]
    else:
        RGB_array = np.asarray([RGB_tuples(i / n) for i in range(n)])
        BRG_array = np.zeros(RGB_array.shape)
        BRG_array[:, 0] = RGB_array[:, 2]
        BRG_array[:, 1] = RGB_array[:, 1]
        BRG_array[:, 2] = RGB_array[:, 0]
        colors = [tuple(BRG_array[i, :] * 256) for i in range(n)]
    if black:
        black = (0.0, 0.0, 0.0)
        colors.insert(0, black)
    return colors


def get_experiment_identifier(basedir):
    return "_".join(basedir.rstrip(os.path.sep).split(os.path.sep)[-3:])

    

def get_dbfile(basedir):
    dbfile=os.path.join(
        basedir,
        get_experiment_identifier(basedir) + ".db"
    )
    assert os.path.exists(dbfile), f"{dbfile} not found"
    return dbfile

def get_basedir(experiment):
    tokens = experiment.split("_")
    basedir=f"/flyhostel_data/videos/{tokens[0]}/{tokens[1]}/{'_'.join(tokens[2:4])}"
    return basedir

def get_pixels_per_mm(experiment):
    dbfile=get_dbfile(get_basedir(experiment))
    return int(float(load_metadata_prop("pixels_per_cm", dbfile=dbfile))/10)

def get_number_of_animals(experiment):
    tokens = experiment.split("_")
    number_of_animals=int(tokens[1].rstrip("X"))
    return number_of_animals

def get_wavelet_downsample(experiment):
    DOWNSAMPLES={
        150: 5,
        47: 1,
    }
    return DOWNSAMPLES[int(get_framerate(experiment))]

def add_suffix(filename, suffix=""):

    if suffix != "":
        basename, ext = os.path.splitext(filename)
        filename = basename + "_" + suffix + ext
    
    return filename

def get_square_width(experiment):
    pixels_per_mm=get_pixels_per_mm(experiment)
    if pixels_per_mm < 25:
        return 100
    else:
        return 200


def get_square_height(*args, **kwargs):
    return get_square_width(*args, **kwargs)


def load_config(path=CONFIG_FILE):

    if not os.path.exists(path):
        config = DEFAULT_CONFIG
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        os.makedirs(config["videos"]["folder"], exist_ok=True)

        try:
            with open(path, "w") as fh:
                json.dump(config, fh)
                logger.warning(
                    f"Saving configuration below to {path}"\
                    f" {config}"
                )
        except Exception as error:
            logger.error(
                "Cannot save configuration to {path}"\
                " Please make sure the file exists and it's writable"
            )
    
    else:
        with open(path, "r") as fh:
            config = json.load(fh)
        
    return config


def copy_files_to_store(imgstore_folder, files, overwrite=False, n_jobs=None):

    trajectories_source_path = os.path.join(imgstore_folder, f"{TRAJECTORIES_SOURCE}.pkl")

    trajectories_source={}

    if os.path.exists(trajectories_source_path):
        with open(trajectories_source_path, "rb") as filehandle:
            trajectories_source.update(pickle.load(filehandle))


    if n_jobs is None:
        n_jobs = conf.NUMBER_OF_JOBS_FOR_COPYING_TRAJECTORIES

    output = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(copy_file_to_store)(
            file, imgstore_folder, overwrite, trajectories_source_path
        ) for file in files
    )

    trajectories_source.update({k: v for k, v in output})

    with open(trajectories_source_path, "wb") as filehandle:
        pickle.dump(trajectories_source, filehandle)

    with open(trajectories_source_path.replace(".pkl", ".yml"), "w") as filehandle:
        yaml.dump(trajectories_source, filehandle)


def copy_file_to_store(file, imgstore_folder, overwrite, trajectories_source_path):

    # NOTE
    # some_folder/session_N/trajectories/trajectories.npy
    session = file.split("/")[::-1][2]
    chunk = int(session.replace("session_", ""))

    dest_filename = str(chunk).zfill(6) + ".npy"
    dest_path = os.path.join(imgstore_folder, dest_filename)

    file_exists = os.path.exists(dest_path)

    if not file_exists:
        clean_copy(file, dest_path)
    elif os.path.getmtime(dest_path) >= os.path.getmtime(file):
        logger.debug(f"{file} is updated")
    else:
        if overwrite:
            clean_copy(file, dest_path)
        else:
            logger.debug(f"{dest_path} exists. Not overwriting")
        
    return file, os.path.basename(dest_path)



def raw_copy(file, dest_path):
    shutil.copy(file, dest_path)

def find_chunk_from_filename(file):

    match=int(re.search("session_(\d{6})", file).group(1))
    return match

def find_start_and_end_of_chunk(session_folder, chunk):

    with sqlite3.connect(os.path.join(session_folder, "..", "..", "index.db")) as con:
            cur = con.cursor()
            cur.execute(f"SELECT frame_number FROM frames WHERE chunk={chunk};")
            start= cur.fetchone()[0]
            end = cur.fetchall()[-1][0]

    return start, end



def clean_copy(file, dest_path):

    data=np.load(file, allow_pickle=True).item()._trajectories

    if "chunk" in data:
        chunk = data["chunk"]
    else:
        logger.warning(f"Trajectories file {file} does not carry the source chunk")
        chunk = find_chunk_from_filename(file)
        data["chunk"] = chunk

    #logger.info(f"Copying {file} --> {dest_path}")
    print(f"Copying {file} --> {dest_path}")
    np.save(dest_path, data, allow_pickle=True)



def get_sqlite_file(animal):

    logger.warning("Deprecated. Please replace with get_dbfile")

    tokens = animal.split("_")[:4]
    sqlite_files = glob.glob(f"{os.environ['FLYHOSTEL_VIDEOS']}/{tokens[0]}/{tokens[1]}/{tokens[2]}_{tokens[3]}/{'_'.join(tokens)}.db")
    assert len(sqlite_files) == 1
    sqlite_file=sqlite_files[0]

    assert os.path.exists(sqlite_file)
    return sqlite_file

def load_metadata_prop(prop, animal=None, dbfile=None):

    if dbfile is None:
        dbfile = get_sqlite_file(animal)

    try:
        with sqlite3.connect(dbfile) as connection:
            cursor = connection.cursor()
            cursor.execute(f"SELECT value FROM METADATA WHERE field = '{prop}';")
            prop = cursor.fetchone()[0]
    except sqlite3.OperationalError as error:
        logger.error("%s METADATA table cannot be read". dbfile)
        raise error
    else:
        return prop
    
def get_wavelet_profile(experiment):
    framerate=get_framerate(experiment)
    PROFILES={150: "default", 47: "highres", 47.1: "highres2"}
    return PROFILES[framerate]


def get_framerate(experiment):
    dbfile=get_dbfile(get_basedir(experiment))
    return float(load_metadata_prop("framerate", dbfile=dbfile))

def get_partition_size(experiment):
    return int(
        100 * np.floor(get_framerate(experiment)*3600 / 100)
    )

def load_roi_width(dbfile):
    try:
        with sqlite3.connect(dbfile) as conn:
            cursor=conn.cursor()

            cursor.execute(
                """
            SELECT w FROM ROI_MAP;
            """
            )
            [(roi_width,)] = cursor.fetchall()
            cursor.execute(
                """
            SELECT h FROM ROI_MAP;
            """
            )
            [(roi_height,)] = cursor.fetchall()

        roi_width=int(roi_width)
        roi_height=int(roi_height)
        roi_width=max(roi_width, roi_height)
    except sqlite3.OperationalError as error:
        logger.error("%s ROI_MAP table cannot be read", dbfile)
        raise error
    
    else:
        return roi_width

def parse_identity(id):
    return int(id.split("|")[1])


def get_local_identities_from_experiment(experiment, frame_number):

    tokens = experiment.split("_")
    experiment_path=os.path.sep.join([tokens[0], tokens[1], "_".join(tokens[2:4])])
    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], experiment_path)
    if not os.path.exists(basedir):
        basedirs=glob.glob(basedir+"*")
        assert len(basedirs) == 1, f"{basedir} not found"
        basedir=basedirs[0]
        experiment = "_".join(basedir.split(os.path.sep))


    dbfile = os.path.join(basedir, experiment + ".db")
    table=get_local_identities(dbfile, [frame_number])
    return table


def get_local_identities_v1(dbfile, frame_numbers, identity_table="IDENTITY"):
    chunksize=get_chunksize(dbfile=dbfile)

    with sqlite3.connect(dbfile) as conn:
        cursor = conn.cursor()
        query = "SELECT frame_number, identity, local_identity FROM {} WHERE frame_number IN ({})".format(
            identity_table,
            ','.join(['?'] * len(frame_numbers))
        )
        cursor.execute(query, frame_numbers)
        
        table = cursor.fetchall()
    
    table=pd.DataFrame.from_records(table, columns=["frame_number", "identity", "local_identity"])
    table["chunk"]=table["frame_number"]//chunksize
    
    return table

def get_local_identities_v2(dbfile, frame_numbers=None, identity_table=None):
    chunksize=get_chunksize(dbfile=dbfile)

    if frame_numbers is not None:
        chunks=(np.array(frame_numbers)//chunksize).tolist()
    else:
        chunks=None
    with sqlite3.connect(dbfile) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM CONCATENATION_VAL;"
        cursor.execute(query)
        table = cursor.fetchall()
    
    table=pd.DataFrame.from_records(table, columns=["id", "chunk", "identity", "local_identity"])
    table["local_identity"]=table["local_identity"].astype(np.int32)
    table["identity"]=table["identity"].astype(np.int32)
    
    
    if chunks is not None:
        table=table.loc[table["chunk"].isin(chunks)]
    
    table["frame_number"]=table["chunk"]*chunksize

    return table


def get_local_identities(dbfile, *args, **kwargs):
    try:
        return get_local_identities_v2(dbfile, *args, **kwargs)
    except sqlite3.OperationalError as error:
        logger.debug("%s is not human validated", dbfile)
        logger.debug(error)
        return get_local_identities_v1(dbfile, *args, **kwargs)


def annotate_local_identity(df, experiment):
    """
    Add a new column local_identity based on the experiemnt+identity+frame_number link
    """
    assert "chunk" in df.columns
    assert "identity" in df.columns
    assert "frame_number" in df.columns   

    dbfile=get_dbfile(get_basedir(experiment))
    
    local_identity_index=get_local_identities(dbfile, df["frame_number"])[["identity", "chunk", "local_identity"]]
    df=df.merge(local_identity_index, on=["chunk", "identity"], how="left")
    return df

def get_chunksize(experiment=None, dbfile=None):

    if dbfile is None:
        assert experiment is not None
        dbfile = get_dbfile(get_basedir(experiment))

    with sqlite3.connect(dbfile) as conn:
        cursor=conn.cursor()
        cursor.execute(f"SELECT value FROM METADATA WHERE field = 'chunksize';",)
        chunksize = int(float(cursor.fetchone()[0]))
    return chunksize

def get_single_animal_video(basedir, frame_number, table, identity, chunksize):
    """
    
    table: data frame containing columns frame_number, identity, local_identity
    """
    assert os.path.isdir(basedir)
    chunk = frame_number // chunksize
    table_current_frame = table.loc[(table["frame_number"] == frame_number)]

    local_identity = table_current_frame.loc[table_current_frame["identity"] == identity, "local_identity"]
    if local_identity.shape[0] == 0:
        single_animal_video=None
    else:
        local_identity=local_identity.item()
        single_animal_video = os.path.join(basedir, "flyhostel", "single_animal", str(local_identity).zfill(3), str(chunk).zfill(6) + ".mp4")
    
    if single_animal_video is None:
        logger.warning("identity %s not found in chunk %s", identity, chunk)
    
    return single_animal_video


def restore_cache(path):
    if os.path.exists(path):
        logger.debug("Loading ---> %s", path)
        before=time.time()
        try:
            with open(path, "rb") as handle:
                out=pickle.load(handle)
        except:
            return False, None
        after=time.time()
        logger.debug("Loading %s took %s seconds", path, after-before)
        

        return True, out
    else:
        return False, None

def save_cache(path, data):
    logger.debug(f"Caching ---> {path}")
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=4)


def annotate_time_in_dataset(dataset, index, t_column="t", t_after_ref=None):
    before=time.time()
    assert index is not None
    assert "frame_number" in dataset.columns

    if t_column in dataset.columns:
        dataset_without_t=dataset.drop(t_column, axis=1)
    else:
        dataset_without_t=dataset
    dataset=dataset_without_t.merge(index[["frame_number", t_column]], on=["frame_number"])
    if t_after_ref is not None and t_column == "frame_time":
        dataset["t"]=dataset[t_column]+t_after_ref
    after=time.time()
    logger.debug("annotate_time_in_dataset took %s seconds", after-before)
    return dataset


def establish_dataframe_framework(dt):
    import cudf

    xf = pd if isinstance(dt, pd.DataFrame) else cudf if isinstance(dt, cudf.DataFrame) else None
    if xf is None:
        raise TypeError("dt must be either a pandas or cuDF DataFrame.")
    return xf


def get_first_frame(dbfile):

    chunksize=get_chunksize(dbfile=dbfile)
    with sqlite3.connect(dbfile) as conn:
        cursor=conn.cursor()
        cursor.execute("SELECT chunk FROM CONCATENATION LIMIT 1;")
        chunk=int(cursor.fetchone()[0])

    frame_number=chunk*chunksize
    return frame_number

def get_last_frame(dbfile):

    chunksize=get_chunksize(dbfile=dbfile)
    with sqlite3.connect(dbfile) as conn:
        cursor=conn.cursor()
        cursor.execute("SELECT chunk FROM CONCATENATION ORDER BY chunk DESC LIMIT 1;")
        chunk=int(cursor.fetchone()[0])+1

    frame_number=chunk*chunksize
    return frame_number

def get_pose_file(experiment, identity, pose_name):
    animal=experiment + "__" + str(identity).zfill(2)
    basedir=get_basedir(experiment)
    pose_file=os.path.join(
        basedir, "motionmapper",
        str(identity).zfill(2),
        f"pose_{pose_name}",
        animal,
        animal + ".h5"
    )
    return pose_file

def get_identities(experiment):
    number_of_animals=get_number_of_animals(experiment)
    if number_of_animals==1:
        identities=[0]
    else:
        identities=list(range(1, number_of_animals+1))
    
    return identities


def get_local_identity(dbfile, chunk, identity):
    table_name="CONCATENATION_VAL"

    with sqlite3.connect(dbfile) as conn:
        cursor=conn.cursor()
        cmd=f"SELECT local_identity FROM {table_name} WHERE identity = {identity} AND chunk = {chunk};"
        print(cmd)
        cursor.execute(cmd)
        local_identity=int(cursor.fetchone()[0])
        return local_identity


def build_interaction_video_key(experiment, row, interaction_name="rejections"):
    return f"{experiment}_{interaction_name}" + \
        "_" + str(row["first_frame"]) + \
        "_" + str(row["last_frame_number"]) + \
        "_" + str(row["local_identity"]).zfill(3) + \
        "_" + str(row["identity"]).zfill(2)


def animal_to_id(individual):
    return individual[:26] + "|" + individual.split("__")[1]

import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Union

import os
import shlex
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union, Optional

import shlex
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RsyncFailure(RuntimeError):
    returncode: int
    cmd: List[str]
    stdout: str = ""
    stderr: str = ""
    log_path: Optional[str] = None
    root: str = ""
    dest: str = ""
    n_files: int = 0
    sample_files: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        cmd_str = " ".join(shlex.quote(x) for x in self.cmd)
        parts = [
            "rsync failed with exit code {}".format(self.returncode),
            "Root: {}".format(self.root),
            "Dest: {}".format(self.dest),
            "Files listed: {}".format(self.n_files),
            "Command:\n  {}".format(cmd_str),
        ]

        if self.log_path:
            parts.append("Rsync log file: {}".format(self.log_path))

        if self.sample_files:
            parts.append(
                "Sample files (from --files-from):\n  {}".format("\n  ".join(self.sample_files))
            )

        if self.stderr.strip():
            parts.append("---- rsync stderr ----\n{}".format(self.stderr.rstrip()))
        if self.stdout.strip():
            parts.append("---- rsync stdout ----\n{}".format(self.stdout.rstrip()))

        return "\n".join(parts)

def rsync_files_from(
    files: Iterable[Union[str, os.PathLike]],
    path: Union[str, os.PathLike],
    *,
    dry_run: bool = False,
    progress: bool = False,
    very_verbose: bool = True,
    keep_log_on_success: bool = False,
) -> None:
    """
    Copy files to a local destination directory using rsync, preserving only the
    subpath after the `/./` marker.

    Example:
        "foo/bar/./baz/file1"  ->  "<path>/baz/file1"

    Supports mixed roots: different prefixes before `/./` are grouped and synced
    in separate rsync calls.

    On rsync failure, raises RsyncFailure containing the full command, exit code,
    stdout/stderr, and the path to an rsync log file.
    """
    dest = Path(path).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    marker = "/./"
    groups: dict[str, list[str]] = defaultdict(list)

    # Parse + group by root
    for f in files:
        s = os.fspath(f)
        if marker not in s:
            raise ValueError(f"Missing '{marker}' marker in path: {s!r}")

        root, rel = s.split(marker, 1)
        if not root:
            raise ValueError(f"Empty root (before '{marker}') in: {s!r}")
        if not rel:
            raise ValueError(f"Empty relative part (after '{marker}') in: {s!r}")

        rel = rel.lstrip("/")
        groups[root].append(f"./{rel}")

    # Run rsync per root
    for root, rel_paths in groups.items():
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            raise FileNotFoundError(f"Rsync root does not exist or is not a directory: {root_path}")

        payload = ("\0".join(rel_paths) + "\0").encode("utf-8")

        # Keep rsync log even if rsync fails, so you can inspect it.
        # Use delete=False and clean up manually on success if desired.
        with tempfile.NamedTemporaryFile(prefix="rsync_files_", suffix=".lst", delete=True) as tf, \
             tempfile.NamedTemporaryFile(prefix="rsync_", suffix=".log", delete=False) as logf:

            tf.write(payload)
            tf.flush()
            log_path = logf.name

            cmd = [
                "rsync",
                "-aR",                # archive + relative paths (honors ./ cut point)
                "--whole-file",       # faster for local copies (no delta algorithm)
                "--from0",
                f"--files-from={tf.name}",
                "--human-readable",
                "--stats",
                f"--log-file={log_path}",
                "--log-file-format=%t [%p] %o %f (%l bytes) %i %B",
            ]

            # Make rsync produce more diagnostics
            if very_verbose:
                cmd += ["-vv"]        # way more detail than -v
                # useful categories; add/remove as desired
                cmd += ["--info=flist2,name2,stats2,progress2"]
            elif progress:
                cmd += ["--info=progress2"]

            if dry_run:
                cmd += ["-n", "--itemize-changes"]

            cmd += [str(root_path) + "/", str(dest) + "/"]

            # Capture stdout/stderr so we can show them on failure.
            # Note: rsync uses stderr for progress/info; capturing is fine.
            proc = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if proc.returncode != 0:
                # show a small sample of file list (helps find a culprit path)
                sample = []
                if rel_paths:
                    # head/tail-ish sample without dumping thousands of entries
                    if len(rel_paths) <= 10:
                        sample = rel_paths
                    else:
                        sample = rel_paths[:5] + ["..."] + rel_paths[-5:]

                raise RsyncFailure(
                    returncode=proc.returncode,
                    cmd=cmd,
                    stdout=proc.stdout or "",
                    stderr=proc.stderr or "",
                    log_path=log_path,
                    root=str(root_path),
                    dest=str(dest),
                    n_files=len(rel_paths),
                    sample_files=sample,
                )

            # Clean up log if successful (optional)
            if not keep_log_on_success:
                try:
                    os.unlink(log_path)
                except OSError:
                    pass



def rsync_files_from_less_verbose(
    files: Iterable[Union[str, os.PathLike]],
    path: Union[str, os.PathLike],
    *,
    dry_run: bool = False,
    progress: bool = False,
) -> None:
    """
    Copy files to a local destination directory using rsync, preserving only the
    subpath after the `/./` marker.

    Example:
        "foo/bar/./baz/file1"  ->  "<path>/baz/file1"

    Supports mixed roots: different prefixes before `/./` are grouped and synced
    in separate rsync calls.
    """
    dest = Path(path).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    marker = "/./"
    groups: dict[str, list[str]] = defaultdict(list)

    # Parse + group by root
    for f in files:
        s = os.fspath(f)
        if marker not in s:
            raise ValueError(f"Missing '{marker}' marker in path: {s!r}")

        root, rel = s.split(marker, 1)
        if not root:
            raise ValueError(f"Empty root (before '{marker}') in: {s!r}")
        if not rel:
            raise ValueError(f"Empty relative part (after '{marker}') in: {s!r}")

        # Ensure rsync sees a relative path beginning with "./"
        rel = rel.lstrip("/")
        groups[root].append(f"./{rel}")

    # Run rsync per root
    for root, rel_paths in groups.items():
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            raise FileNotFoundError(f"Rsync root does not exist or is not a directory: {root_path}")

        # NUL-delimited file list for --from0
        payload = ("\0".join(rel_paths) + "\0").encode("utf-8")

        with tempfile.NamedTemporaryFile(prefix="rsync_files_", suffix=".lst", delete=True) as tf:
            tf.write(payload)
            tf.flush()

            cmd = [
                "rsync",
                "-aR",                  # archive + relative paths (honors ./ cut point)
                "--whole-file",          # faster for local copies (no delta algorithm)
                "--from0",
                f"--files-from={tf.name}",
            ]

            if progress:
                cmd += ["--info=progress2"]
            if dry_run:
                cmd += ["-n", "--itemize-changes"]

            # Trailing slashes matter:
            # source is "contents of root", destination is a directory
            cmd += [str(root_path) + "/", str(dest) + "/"]

            subprocess.run(cmd, check=True)


def dunder_to_slash(experiment):
    tokens = experiment.split("_")
    return tokens[0] + "/" + tokens[1] + "/" + "_".join(tokens[2:4])
