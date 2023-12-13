from abc import ABC, abstractmethod
import logging
import os.path

import numpy as np
import pandas as pd

from .utils import get_local_identities_from_experiment, get_sqlite_file
logger = logging.getLogger(__name__)

DEG_DATA="/Users/FlySleepLab Dropbox/Data/flyhostel_data/fiftyone/FlyBehaviors/DEG/FlyHostel_deepethogram/DATA"

def parse_entry(data_entry, verbose=True):
    tokens = data_entry.split("_")
    if len(tokens) != 6:
        if verbose:
            print(f"Invalid entry: {data_entry}")
        return False, None
    
    return True, tokens


class DEGLoader(ABC):

    def __init__(self, *args, **kwargs):
        self.experiment=None
        self.deg=None
        self.datasetnames=None
        super(DEGLoader, self).__init__(*args, **kwargs)

    @abstractmethod
    def get_chunksize(self, dbfile):
        raise NotImplementedError
    
    @staticmethod
    def parse_chunk(data_entry):
        ret, tokens = parse_entry(data_entry)
        if ret:
            chunk = int(tokens[-2])
            return chunk
        else:
            return None

    @staticmethod
    def parse_local_identity(data_entry):
        ret, tokens = parse_entry(data_entry)
        if ret:
            local_identity = int(tokens[-1])
            return local_identity
        else:
            return None

    @staticmethod
    def parse_experiment(data_entry):
        ret, tokens = parse_entry(data_entry)
        if ret:
            experiment = "_".join(tokens[:-2])
            return experiment
        else:
            return None
    
    @staticmethod
    def parse_number_of_animals(data_entry):
        ret, tokens = parse_entry(data_entry)
        if ret:
            number_of_animals=int(tokens[1].replace("X",""))
            return number_of_animals
        else:
            return None

    def filter_by_id(self, data_entry, identity, chunksize=45000, verbose=True):

        local_identity=self.parse_local_identity(data_entry)
        chunk=self.parse_chunk(data_entry)
        experiment_=self.parse_experiment(data_entry)

        if local_identity is None:
            logger.warning("%s is corrupt. No local_identity can be parsed", data_entry)
            return False, None
        
        if chunk is None:
            logger.warning("%s is corrupt. No chunk can be parsed", data_entry)
            return False, None

        if experiment_ is None:
            logger.warning("%s is corrupt. No experiment can be parsed", data_entry)
            return False, None
        

        frame_number = chunk*chunksize
        table=get_local_identities_from_experiment(experiment_, int(frame_number))

        identity_=int(table["identity"].loc[table["local_identity"]==local_identity])

        if not (experiment_ == self.experiment and identity_ == identity):
            return False, None
        
        labels_file=f"{DEG_DATA}/{data_entry}/{str(chunk).zfill(6)}_labels.csv"

        return True, labels_file

    def load_deg_data_prediction(self, identity, verbose=True):
        raise NotImplementedError()
    

    def load_deg_data_gt(self, identity=None, verbose=True):
        if identity is not None:
            identities=[identity]
        else:
            if len(self.datasetnames) == 1:
                identities=[0]
            else:
                identities=[animal.split("__")[1] for animal in self.datasetnames]

        for identity in identities:
            self.load_deg_data_gt_single_animal(identity=identity, verbose=verbose)


    def load_deg_data_gt_single_animal(self, identity, verbose=True):
        all_labels=[]
        id = self.experiment[:26] + "|" + str(identity).zfill(2)
        animal = self.experiment + "__" + str(identity).zfill(2)
        dbfile=get_sqlite_file(animal)
        chunksize=self.get_chunksize(dbfile)
        counter=0

        for data_entry in os.listdir(DEG_DATA):

            ret, labels_file = self.filter_by_id(data_entry, identity, chunksize=chunksize, verbose=verbose)
            local_identity=self.parse_local_identity(data_entry)
            chunk=self.parse_chunk(data_entry)
            
            if not ret or labels_file is None:
                continue

            if not os.path.exists(labels_file):
                if verbose:
                    print(f"{labels_file} not found")
                continue
            
            labels=read_label_file(data_entry, labels_file, verbose=verbose)
            labels["chunk"]=chunk
            labels["local_identity"]=local_identity
            # labels["frame_idx"]=np.arange(0, labels.shape[0])
            labels["id"]=id
            labels["frame_number"]=labels["chunk"]*chunksize+labels["frame_idx"]

            all_labels.append(labels)
            del labels
            counter+=1


        if len(all_labels) == 0:
            print(f"No labels found for {self.experiment}__{str(identity).zfill(2)}")
            return None

        else:
            print(f"id {id}: Number of label.csv found {counter}")
            labels=pd.concat(all_labels, axis=0)
            if self.deg is None:
                self.deg = labels
            else:
                self.deg = pd.concat([self.deg, labels], axis=0)




def read_label_file(data_entry, labels_file, verbose=False):

    labels=pd.read_csv(labels_file, index_col=0)

    # if pe and inactive ae true, then it's a separate behavior
    labels["pe_inactive"]=((labels["pe"]==1) & (labels["inactive"]==1))*1
    labels["pe"].loc[labels["pe_inactive"]==1]=0
    labels["inactive"].loc[labels["pe_inactive"]==1]=0
    behaviors=labels.columns

    behavr_count_per_frame = labels.values.sum(axis=1)
    criteria = behavr_count_per_frame >= 1

    if not (criteria).all() and verbose:
        print(f"{data_entry} has {np.sum(behavr_count_per_frame < 1)} unlabeled frames")

    frames, behav_ids = np.where(labels.values==1)
    behav_sequence=[]
    for frame, behav_id in zip(frames, behav_ids):
        behav_sequence.append((frame, behaviors[behav_id]))
    labels=pd.DataFrame.from_records(behav_sequence)
    labels.columns=["frame_idx", "behavior"]
    return labels


