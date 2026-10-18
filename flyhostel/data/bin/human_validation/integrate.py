import argparse
import datetime
import logging
import math
import os.path
import shlex
import sys
import sqlite3
import traceback

from flyhostel.utils import get_basedir, get_dbfile
from flyhostel.utils.utils import get_chunksize, get_chunk_range_from_zt, get_number_of_animals, load_reference_hour
from flyhostel.data.human_validation.cvat.main import (
    integrate_human_annotations,
    save_human_annotations
)


def write_validation_call(dbfile, cmd):
    with sqlite3.connect(dbfile) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS VALIDATION_CALL (command TEXT, value TEXT)"
        )
        conn.execute(
            "INSERT INTO VALIDATION_CALL (command, value) VALUES (?, ?)",
            ("integrate", cmd),
        )

class _Tee:
    """Write to a stream and to a log file at the same time"""

    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile

    def write(self, data):
        self.stream.write(data)
        self.logfile.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class tee_output:
    """
    Mirror everything printed to the screen (stdout, stderr and logging handlers
    writing to either of them) to a log file
    """

    def __init__(self, path):
        self.path = path

    def _logging_handlers(self):
        loggers = [logging.getLogger()] + [
            logger for logger in logging.Logger.manager.loggerDict.values()
            if isinstance(logger, logging.Logger)
        ]
        for logger in loggers:
            for handler in logger.handlers:
                if type(handler) is logging.StreamHandler:
                    yield handler

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.logfile = open(self.path, "a", buffering=1)
        self.stdout, self.stderr = sys.stdout, sys.stderr
        tees = {
            id(self.stdout): _Tee(self.stdout, self.logfile),
            id(self.stderr): _Tee(self.stderr, self.logfile),
        }
        sys.stdout = tees[id(self.stdout)]
        sys.stderr = tees[id(self.stderr)]

        # logging handlers keep a reference to the stream they were created with
        self.handlers = []
        for handler in self._logging_handlers():
            if id(handler.stream) in tees:
                self.handlers.append((handler, handler.stream))
                handler.setStream(tees[id(handler.stream)])
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.logfile.write("".join(traceback.format_exception(exc_type, exc, tb)))
        for handler, stream in self.handlers:
            handler.setStream(stream)
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self.logfile.close()
        return False


def get_parser():

    ap=argparse.ArgumentParser(conflict_handler="resolve")
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--folder", type=str, default=None)
    ap.add_argument("--fn-interval", type=int, nargs=2, required=False, default=[None, None])
    ap.add_argument("--zt0", type=str, default=None, help="Time of day of ZT0 (HH:MM or HH:MM:SS, same timezone as the folder name) used to resolve --start-zt / --end-zt, like in the idtrackerai pipeline. Default: reference_hour in the experiment's metadata")
    ap.add_argument("--start-zt", type=float, default=None, help="Validate the chunks from the one that reaches this ZT (hours since the last ZT0 before the experiment start, can be above 24). Missing means first chunk. Not compatible with --fn-interval")
    ap.add_argument("--end-zt", type=float, default=None, help="Validate the chunks until the last one that starts before this ZT. Missing means last chunk. Not compatible with --fn-interval")
    ap.add_argument("--tasks", type=int, nargs="+", required=False)
    ap.add_argument("--frames-from-annotation", dest="frames_from_annotation", action="store_true", help="Infer frame window from annotations")
    ap.add_argument("--multisex", action="store_true", help="Are there flies from both sexes in the experiment?")
    return ap


def get_frame_interval(args):
    """
    First and last (excluded) frame number to validate,
    either from --fn-interval or from the chunks overlapping --start-zt / --end-zt
    The chunks are selected with the same function as the idtrackerai pipeline (zt_chunk_range.py)
    """
    if args.start_zt is None and args.end_zt is None:
        return args.fn_interval

    if args.fn_interval != [None, None]:
        raise ValueError("Pass either --fn-interval or --start-zt / --end-zt, not both")

    basedir=get_basedir(args.experiment)
    if args.zt0 is None:
        identity=0 if get_number_of_animals(args.experiment)==1 else 1
        zt0=load_reference_hour(get_dbfile(basedir), identity)
    else:
        zt0=args.zt0

    start_chunk, end_chunk=get_chunk_range_from_zt(basedir, zt0, start_zt=args.start_zt, end_zt=args.end_zt)
    chunksize=get_chunksize(args.experiment)
    print(f"ZT0 = {zt0}: ZT{args.start_zt}-ZT{args.end_zt} -> chunks {start_chunk}-{end_chunk}")
    return start_chunk*chunksize, (end_chunk+1)*chunksize


def main():

    cmd = shlex.join(sys.orig_argv)

    ap=get_parser()
    ap.add_argument("--number-of-rows", type=int, default=1, help="If images in cvat are a grid, how many rows the grid has")
    ap.add_argument("--number-of-cols", type=int, default=1, help="If images in cvat are a grid, how many rows the grid has")
    ap.add_argument("--redownload", action="store_true", default=False, required=False)
    ap.add_argument("--use-cache", action="store_true", default=False, required=False)
    ap.add_argument("--log-file", type=str, default=None, help="File where everything printed to the screen is also saved. Default: <folder>/integrate_<timestamp>.log")

    args=ap.parse_args()

    if args.folder is None:
        folder = f"{get_basedir(args.experiment)}/flyhostel/validation"
    else:
        folder=args.folder

    if args.log_file is None:
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file=os.path.join(folder, f"integrate_{timestamp}.log")
    else:
        log_file=args.log_file

    with tee_output(log_file):
        print(f"Saving log to {log_file}")
        print(cmd)
        integrate(args, folder, cmd)


def integrate(args, folder, cmd):

    if args.tasks is None:
        tasks=None
    else:
        tasks=tuple(args.tasks)

    first_frame_number, last_frame_number=get_frame_interval(args)

    integrate_human_annotations(
        args.experiment,
        folder=folder,
        tasks=tasks,
        first_frame_number=first_frame_number,
        last_frame_number=last_frame_number,
        redownload=args.redownload,
        number_of_rows=args.number_of_rows,
        number_of_cols=args.number_of_cols,
        frames_from_annotation=args.frames_from_annotation,
        multisex=args.multisex,
        use_cache=args.use_cache,
    )

    dbfile=get_dbfile(get_basedir(args.experiment))
    write_validation_call(dbfile, cmd)
    

def save():

    ap=get_parser()
    args=ap.parse_args()

    if args.folder is None:
        folder = f"{get_basedir(args.experiment)}/flyhostel/validation"
    else:
        folder=args.folder

    first_frame_number, last_frame_number=get_frame_interval(args)

    save_human_annotations(
        args.experiment, folder,
        first_frame_number=first_frame_number,
        last_frame_number=last_frame_number,
        frames_from_annotation=args.frames_from_annotation,
    )

