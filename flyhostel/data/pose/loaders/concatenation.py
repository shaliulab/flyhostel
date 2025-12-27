from abc import ABC
import logging
import sqlite3
import pandas as pd

from flyhostel.utils import get_dbfile
logger=logging.getLogger(__name__)

class ConcatenationLoader(ABC):
    
    basedir=None
    number_of_animals=None
    identity=None

    def load_concatenation_table(self):

        if self.number_of_animals==1:
            conc_tab="CONCATENATION"
        else:
            conc_tab="CONCATENATION_VAL"
        
        file = get_dbfile(self.basedir)
        sql=f"SELECT * FROM {conc_tab} WHERE identity = {self.identity};"
        with sqlite3.connect(file) as conn:
            concatenation_table=pd.read_sql(con=conn, sql=sql)
        return concatenation_table
