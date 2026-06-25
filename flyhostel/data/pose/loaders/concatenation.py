from abc import ABC
import logging
import sqlite3

from flyhostel.utils import (
    experiment_is_validated,
)
from flyhostel.utils.pose_export import load_concatenation_table

from flyhostel.utils import get_dbfile
logger=logging.getLogger(__name__)


class ConcatenationLoader(ABC):
    
    basedir=None
    number_of_animals=None
    experiment=None
    identity=None

    def load_concatenation_table(self):

        if self.number_of_animals>1 and experiment_is_validated(self.experiment):
            conc_tab="CONCATENATION_VAL"
        else:
            conc_tab="CONCATENATION"
        
        dbfile = get_dbfile(self.basedir)
        table=None
        with sqlite3.connect(dbfile) as conn:
            cur=conn.cursor()
            table=load_concatenation_table(cur, self.basedir, concatenation_table=conc_tab)
        return table

