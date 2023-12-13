import os.path
import glob
import sqlite3

import pandas as pd

def get_sqlite_file(animal):

    tokens = animal.split("_")[:4]
    sqlite_files = glob.glob(f"{os.environ['FLYHOSTEL_VIDEOS']}/{tokens[0]}/{tokens[1]}/{tokens[2]}_{tokens[3]}/{'_'.join(tokens)}.db")
    assert len(sqlite_files) == 1
    sqlite_file=sqlite_files[0]

    assert os.path.exists(sqlite_file)
    return sqlite_file

def load_metadata_prop(prop, animal=None, dbfile=None):

    if dbfile is None:
        dbfile = get_sqlite_file(animal)

    with sqlite3.connect(dbfile) as connection:
        cursor = connection.cursor()
        cursor.execute(f"SELECT value FROM METADATA WHERE field = '{prop}';")
        prop = cursor.fetchone()[0]
    return prop

def load_roi_width(dbfile):
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

def get_local_identities(dbfile, frame_numbers):

    with sqlite3.connect(dbfile) as conn:
        cursor = conn.cursor()
        query = "SELECT frame_number, identity, local_identity FROM identity WHERE frame_number IN ({})".format(
            ','.join(['?'] * len(frame_numbers))
        )
        cursor.execute(query, frame_numbers)
        
        table = cursor.fetchall()
    
    table=pd.DataFrame.from_records(table, columns=["frame_number", "identity", "local_identity"])
    return table

