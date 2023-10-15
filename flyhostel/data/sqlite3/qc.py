from abc import ABC
import joblib
import re
import sqlite3
import os.path
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

def read_n_fragments(basedir, chunk):
    logger.debug("Reading number of fragments")
    fragment_file = os.path.join(
        basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}",
        "crossings_detection_and_fragmentation", "fragments.npy"
    )
    if os.path.exists(fragment_file):
        list_of_fragments = np.load(fragment_file, allow_pickle=True).item()
        n_fragments = len(list_of_fragments.fragments)
        return n_fragments
    else:
        return None


def parse_n_global_fragments(lines):
    logger.debug("Parsing number of global fragments")
    hit = None
    for line in lines[::-1]:
        hit=re.search("total number of global_fragments: (\d*)", line)
        if hit:
            break
    if hit is not None:
        n_global_fragments = int(hit.group(1))
    else:
        n_global_fragments = None

    return n_global_fragments


def parse_accuracy(lines):
    logger.debug("Reading idtrackerai tracking accuracy")
    hit = None
    for line in lines[::-1]:
        hit=re.search("Estimated accuracy: (.*)", line)
        if hit:
            break
    if hit is not None:
        accuracy = float(hit.group(1))
    else:
        accuracy = None

    return accuracy


cmd = "INSERT INTO QC (chunk, fragments, global_fragments, accuracy) VALUES (?, ?, ?, ?);"


def load_qc_params(basedir, chunk, file):
    n_fragments = read_n_fragments(basedir, chunk)
    
    with open(file, "r") as filehandle:
        lines = filehandle.readlines()
        accuracy = parse_accuracy(lines)
        n_global_fragments = parse_n_global_fragments(lines)
        # add other parsers here as needed

    params={
        "chunk": chunk,
        "accuracy": accuracy,
        "global_fragments": n_global_fragments,
        "fragments": n_fragments,
    }
    return params


class QCExporter(ABC):

    _basedir = None

    @staticmethod
    def run_qc(basedir, chunks=None):

        logger.debug("Selecting tracking_output.txt files in %s", basedir)
        idtrackerai_folder=os.path.join(basedir, "idtrackerai")
        paths = os.listdir(idtrackerai_folder)
        paths = [
            path
            for path in paths
            if path.startswith("session_") and path.endswith("_tracking_output.txt")
        ]
        tracking_outputs = []
        for path in paths:
            hit=re.search("session_(\d{6})", path)
            if hit:
                chunk = int(hit.group(1))
                if chunks is not None:
                    if chunk in chunks:                       
                        tracking_outputs.append((chunk, path))
                else:
                    tracking_outputs.append((chunk, path))


        table = joblib.Parallel(n_jobs=-2)(
            joblib.delayed(load_qc_params)(
                basedir, chunk, os.path.join(idtrackerai_folder, file)
            )
            for chunk, file in tracking_outputs
        )

        qc_table=pd.DataFrame.from_records(table)
        # check that the table is not empty
        if qc_table.shape[0] > 0:
            qc_table.sort_values("chunk", inplace=True)
        return qc_table


    def init_qc_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                print("Dropping QC")
                cur.execute("DROP TABLE IF EXISTS QC;")
            cur.execute("CREATE TABLE IF NOT EXISTS QC (id INTEGER PRIMARY KEY AUTOINCREMENT, chunk int(3), fragments int(3), global_fragments int(3), accuracy real(3));")

    def write_qc_table(self, dbfile, chunks):

        if "1X" in dbfile:
            with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                cur=conn.cursor()
                data = []
                chunks=sorted(chunks)
                for chunk in chunks:
                    data.append((chunk, 1, 1, 1.0))
                
                cur.executemany(
                        cmd,
                        data
                    )
                        
        else:

            qc_table = self.run_qc(basedir=self._basedir, chunks=chunks)

            logger.debug("Exporting QC table to %s", dbfile)
            data=[]
            with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                cur = conn.cursor()
                for _, row in qc_table.iterrows():
                    chunk = row["chunk"]
                    args = (
                        chunk, row["fragments"], row["global_fragments"], round(row["accuracy"], 3),
                    )
                    args2=[]
                    for e in args:
                        try:
                            e = e.item()
                        except AttributeError:
                            pass
                        args2.append(e)
                    args=tuple(args2)
                    chunk=args[0]
                    if chunk in chunks:
                        data.append(args)
                    else:
                        pass

                if data:
                    conn.executemany(
                        cmd,
                        data
                    )