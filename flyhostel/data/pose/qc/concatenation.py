import sqlite3
import pandas as pd

from flyhostel.utils import get_dbfile, get_basedir
from flyhostel.data.pose.loaders.concatenation import load_concatenation_table


def check_concatenation(experiment, **kwargs):
    basedir=get_basedir(experiment)
    dbfile=get_dbfile(basedir)
    
    with sqlite3.connect(dbfile) as conn:
        cursor=conn.cursor()
        table=load_concatenation_table(cur=cursor, basedir=basedir, **kwargs)

    return table
