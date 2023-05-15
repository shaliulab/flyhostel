import os.path
import re
import glob
import sqlite3

import yaml
import cv2

def table_is_not_empty(dbfile, table_name):
    """
    Returns True if the passed table has at least one row
    """

    with sqlite3.connect(dbfile, check_same_thread=False) as conn:
        cur=conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count=cur.fetchone()[0]

    return count > 0


def ensure_type(var, var_name, typ):

    if not isinstance(var, typ):
        try:
            var=typ(var)
        except ValueError as exc:
            raise ValueError(f"{var_name} {var} with type {type(var)} cannot be coerced to {typ}") from exc

    return var

def serialize_arr(arr, path):
    """
    Transform an image (np.array) to bytes for export in SQLite
    """

    cv2.imwrite(path, arr, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

    with open(path, "rb") as filehandle:
        bstring = filehandle.read()

    return bstring



def parse_experiment_properties(basedir=None):
    wd=os.getcwd()

    if basedir is not None:
        os.chdir(basedir)
    
    try:   
        idtrackerai_conf_path = glob.glob("20*.conf")[0]
        
        date_time = re.search("(20.*).conf", idtrackerai_conf_path).group(1)
        flyhostel_id = int(re.search(
            "FlyHostel([0-9])",
            os.path.realpath("metadata.yaml")
        ).group(1))
    
    finally:
        os.chdir(wd)


    with open(idtrackerai_conf_path, "r", encoding="utf8") as filehandle:
        idtrackerai_conf = yaml.load(filehandle, yaml.SafeLoader)

    number_of_animals = int(idtrackerai_conf["_number_of_animals"]["value"])
    return (idtrackerai_conf_path, idtrackerai_conf), (flyhostel_id, number_of_animals, date_time)