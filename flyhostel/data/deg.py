import time
from abc import abstractmethod
import logging
import os.path

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from flyhostel.utils import restore_cache, save_cache
from flyhostel.utils import get_local_identities_from_experiment, get_sqlite_file, get_chunksize
logger = logging.getLogger(__name__)

DEG_DATA="/Users/FlySleepLab Dropbox/Data/flyhostel_data/fiftyone/FlyBehaviors/DEG/FlyHostel_deepethogram/DATA"
RESTORE_FROM_CACHE_ENABLED=False

def parse_entry(data_entry, verbose=True):
    tokens = data_entry.split("_")
    if len(tokens) != 6:
        if verbose:
            logger.error(f"Invalid entry: {data_entry}")
        return False, None

    return True, tokens


class DEGLoader:

    def __init__(self, *args, **kwargs):
        self.experiment=None
        self.deg=None
        self.datasetnames=None
        self.store_index=None
        self.meta_info={}
        super(DEGLoader, self).__init__(*args, **kwargs)


    @abstractmethod
    def load_store_index(self, cache=None):
        raise NotImplementedError()

    def load_deg_data(self, *args, min_time=-np.inf, max_time=+np.inf, stride=1, time_system="zt", ground_truth=True,  cache=None, **kwargs):

        if cache is not None and RESTORE_FROM_CACHE_ENABLED:

            path = os.path.join(cache, f"{self.experiment}_{min_time}_{max_time}_{stride}_deg_data.pkl")
            before=time.time()
            ret, self.deg=restore_cache(path)
            after=time.time()
            logger.debug("Loading %s took %s seconds", path, after-before)
            if ret:
                logger.debug("Loaded %s rows from cache", self.deg.shape[0])
                return

        if ground_truth:
            self.load_deg_data_gt(*args, **kwargs)
        else:
            self.load_deg_data_prediction(*args, **kwargs)

        if self.deg is not None:
            self.load_store_index(cache=cache)
            t=self.store_index["frame_time"]+self.meta_info["t_after_ref"]

            if min_time==-np.inf:
                min_fn=self.store_index["frame_number"].iloc[0]
            else:
                min_fn=self.store_index["frame_number"].iloc[
                    np.argmax(t>=min_time)
                ]

            if max_time==np.inf:
                max_fn=self.store_index["frame_number"].iloc[-1]+1
            else:
                max_fn=self.store_index["frame_number"].iloc[
                    -(np.argmax(t[::-1]<max_time)-1)
                ]

            self.deg=self.deg.loc[
                (self.deg["frame_number"] >= min_fn) & (self.deg["frame_number"] < max_fn)
            ]
            self.deg["behavior"].loc[pd.isna(self.deg["behavior"])]="unknown"
            before=time.time()
            self.deg=self.annotate_two_or_more_behavs_at_same_time_(self.deg)
            after=time.time()
            logger.debug("Took %s seconds to annotate two or more behaviors", after-before)

        if cache and self.deg is not None:
            path = os.path.join(cache, f"{self.experiment}_{min_time}_{max_time}_{stride}_deg_data.pkl")
            save_cache(path, self.deg)




    @staticmethod
    def annotate_two_or_more_behavs_at_same_time_(*args, **kwargs):
        return annotate_two_or_more_behavs_at_same_time(*args, **kwargs)



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
        chunksize=get_chunksize(dbfile)
        counter=0
        ignored_suffixes=[".dvc"]
        ignored_prefixes=["."]
        entries=os.listdir(DEG_DATA)

        pb=tqdm(total=len(entries))


        for data_entry in entries:
            if data_entry=="split.yaml":
                pb.update(1)
                continue

            if any((data_entry.startswith(pattern) for pattern in ignored_prefixes)):
                pb.update(1)
                continue

            if any((data_entry.endswith(pattern) for pattern in ignored_suffixes)):
                pb.update(1)
                continue

            ret, labels_file = self.filter_by_id(data_entry, identity, chunksize=chunksize, verbose=verbose)
            local_identity=self.parse_local_identity(data_entry)
            chunk=self.parse_chunk(data_entry)

            if not ret or labels_file is None:
                pb.update(1)
                continue

            if not os.path.exists(labels_file):
                if verbose:
                    print(f"{labels_file} not found")
                pb.update(1)
                continue

            labels=read_label_file(data_entry, labels_file, verbose=verbose)
            if labels is None:
                print(f"{labels_file} cannot be read")
                pb.update(1)
                continue

            labels["chunk"]=chunk
            labels["local_identity"]=local_identity
            # labels["frame_idx"]=np.arange(0, labels.shape[0])
            labels["id"]=id
            labels["frame_number"]=labels["chunk"]*chunksize+labels["frame_idx"]

            all_labels.append(labels)
            del labels
            counter+=1
            pb.update(1)


        if len(all_labels) == 0:
            logger.info(f"No labels found for {self.experiment}__{str(identity).zfill(2)}")
            return None

        else:
            logger.info(f"id {id}: Number of label.csv found {counter}")
            labels=pd.concat(all_labels, axis=0)
            if self.deg is None:
                self.deg = labels
            else:
                self.deg = pd.concat([self.deg, labels], axis=0)


def read_label_file(data_entry, labels_file, verbose=False):

    labels=pd.read_csv(labels_file, index_col=0)

    # if pe and inactive ae true, then it's a separate behavior
    labels["pe_inactive"]=((labels["pe"]==1) & (labels["inactive"]==1))*1
    labels.loc[labels["pe_inactive"]==1, "pe"]=0
    labels.loc[labels["pe_inactive"]==1, "inactive"]=0
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

    try:
        labels.columns=["frame_idx", "behavior"]
        # labels.loc[labels["behavior"]=="turn", "behavior"]="twitch"
    except ValueError:
        logger.warning("%s cannot be loaded. Is it empty? The number of rows is %s", labels_file, labels.shape[0])
        return None
    return labels


def annotate_two_or_more_behavs_at_same_time(deg):
    """
    If more than 1 behavior is present in a given frame,
    create a new behavioral label by chaining said behaviors with +
    So for example, if the fly is walking and feeding at the same time,
    make it the behavior feed+walk
    """

    # Group by frame_number and id, join behaviors with '+', and reset index
    counts=deg.groupby("frame_number").size().reset_index()
    counts.columns=["frame_number", "counts"]

    deg.set_index(["frame_number"], inplace=True)

    deg_single=deg.loc[
        counts.loc[counts["counts"]==1, "frame_number"].values
    ].reset_index()
    deg_multi=deg.loc[
        counts.loc[counts["counts"]>1, "frame_number"].values
    ].reset_index()


    deg_group = deg_multi.groupby(["frame_number"])["behavior"].agg(lambda x: "+".join(sorted(list(set(x))))).reset_index()
    deg_multi=deg_group.merge(
        deg_multi.drop(["behavior"], axis=1).drop_duplicates(),
        on=["frame_number"]
    )
    deg=pd.concat([deg_single, deg_multi], axis=0)
    return deg