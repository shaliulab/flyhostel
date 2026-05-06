from abc import ABC
import logging
import json
import os.path
import sqlite3
import numpy as np
import pandas as pd

from flyhostel.utils import get_dbfile
logger=logging.getLogger(__name__)

def infer_analysis_path(basedir, local_identity, chunk, number_of_animals):
    if number_of_animals==1:
        return os.path.join(basedir, "flyhostel", "single_animal", "000",                        str(chunk).zfill(6)+".mp4.predictions.h5")
    else:
        return os.path.join(basedir, "flyhostel", "single_animal", str(local_identity).zfill(3), str(chunk).zfill(6)+".mp4.predictions.h5")


def load_concatenation_table(cur, basedir, concatenation_table="CONCATENATION_VAL", errors="raise"):
    cur.execute("SELECT value FROM METADATA where field ='idtrackerai_conf';")
    conf=cur.fetchone()[0]
    number_of_animals=int(json.loads(conf)["_number_of_animals"]["value"])

    cur.execute(f"PRAGMA table_info('{concatenation_table}');")
    header=[row[1] for row in cur.fetchall()]

    cur.execute(f"SELECT * FROM {concatenation_table};")
    records=cur.fetchall()
    concatenation=pd.DataFrame.from_records(records, columns=header)
    concatenation["chunk"]=concatenation["chunk"].astype(int)
    
    concatenation.sort_values("chunk", inplace=True)
    diff=concatenation["chunk"].drop_duplicates().diff().iloc[1:]

    
    if errors == "raise":
        if not (diff==1).all():
            rows=np.where(diff!=1)[0].tolist()
            rows=sorted(rows + (np.array(rows)+1).tolist())
            print(concatenation.iloc[1:].loc[sorted(rows)])
            raise ValueError("Missing chunks in concatenation")

    concatenation["dfile"] = [
        infer_analysis_path(basedir, int(row["local_identity"]), str(int(row["chunk"])).zfill(6), number_of_animals=number_of_animals)
        for i, row in concatenation.iterrows()
    ]
    return concatenation

class ConcatenationLoader(ABC):
    
    basedir=None
    number_of_animals=None
    identity=None

    def load_concatenation_table(self):

        if self.number_of_animals==1:
            conc_tab="CONCATENATION"
        else:
            conc_tab="CONCATENATION_VAL"
        
        dbfile = get_dbfile(self.basedir)
        table=None
        with sqlite3.connect(dbfile) as conn:
            cur=conn.cursor()
            table=load_concatenation_table(cur, self.basedir, concatenation_table=conc_tab)
        return table

