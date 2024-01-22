from abc import ABC
import glob
import os.path
import logging

logger=logging.getLogger(__name__)

POSE_DATA=os.environ["POSE_DATA"]

class FilesystemInterface(ABC):

    basedir=None
    experiment=None
    identity=None


    def load_dbfile(self):
        dbfiles=glob.glob(self.basedir + "/FlyHostel*.db")
        assert len(dbfiles) == 1, f"{len(dbfiles)} dbfiles found in {self.basedir}: {' '.join(dbfiles)}"
        return dbfiles[0]


    def load_datasetnames(self):

        datasetnames = []
        animals=os.listdir(POSE_DATA)
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