from abc import ABC
import sqlite3
import glob
import os.path
import logging

logger=logging.getLogger(__name__)

class FilesystemInterface(ABC):

    basedir=None
    experiment=None
    identity=None


    def load_dbfile(self):
        dbfiles=glob.glob(self.basedir + "/FlyHostel*.db")
        assert len(dbfiles) == 1, f"{len(dbfiles)} dbfiles found in {self.basedir}: {' '.join(dbfiles)}"
        dbfile=dbfiles[0]
        self.assert_file_integrity(dbfile)

        return dbfile
    

    @staticmethod
    def assert_file_integrity(dbfile):

        try: 
            with sqlite3.connect(dbfile) as connection:
                cursor=connection.cursor()
                cursor.execute("SELECT * FROM METADATA;")
                cursor.fetchall()
            
        except sqlite3.DatabaseError as error:
            logger.error("Cant read %s", dbfile)
            raise error



    def load_datasetnames(self):
        raise NotImplementedError

        datasetnames = []
        animals=os.listdir(None)
        datasetnames=sorted(list(filter(lambda animals: animals.startswith(self.experiment), animals)))

        if not datasetnames:
            logger.warning(f"No datasets starting with {self.experiment} found in {POSE_DATA}")
    
        else:
            if self.identity is not None:
                datasetnames=[datasetnames[self.identity-1]]
        return datasetnames

    
    @staticmethod
    def make_ids(datasetnames):
        identities = [
            d[:26] +  "|" + d[-2:]
            for d in datasetnames
        ]
        return identities