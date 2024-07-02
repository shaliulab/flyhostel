import logging
import sqlite3
import pandas as pd

logger=logging.getLogger()

def check_if_validated(dbfile):

    with sqlite3.connect(dbfile) as conn:
        cur=conn.cursor()
        cur.execute(
            """SELECT name FROM sqlite_master  
            WHERE type='table';"""
        )
        tables=[value[0] for value in cur.fetchall()]

        if "IDENTITY_VAL" in tables:
            return "_VAL"
        else:
            return ""

def get_identity(number_of_animals, dbfile, local_identity, chunk):
    if number_of_animals==1:
        identity="0"
        validated=True
    else:
        if "_VAL" == check_if_validated(dbfile):
            validated=True
            concatenation_table="CONCATENATION_VAL"
        else:
            logger.warning("%s not validated", dbfile)
            validated=False
            concatenation_table="CONCATENATION"
        with sqlite3.connect(dbfile) as conn:
            sql=f"SELECT identity FROM {concatenation_table} WHERE chunk = {chunk} AND local_identity = {local_identity};"
            identity=str(pd.read_sql(con=conn, sql=sql).iloc[0].item())
    return identity, validated

