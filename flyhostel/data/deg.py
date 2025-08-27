import time
from abc import abstractmethod
import logging
import os.path

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from flyhostel.data.pose.constants import DEG_DATA
from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.utils import restore_cache, save_cache
from flyhostel.utils import get_local_identities_from_experiment, get_sqlite_file, get_chunksize
from flyhostel.data.pose.ethogram.utils import annotate_bout_duration, annotate_bouts

logger = logging.getLogger(__name__)

BEHAVIORS=["walk", "groom", "feed", "inactive", "micromovement", "pe", "background"]
LABELS=["walk", "background", "groom", "feed", "inactive+rejection", "inactive+pe", "inactive+micromovement", "inactive"]
SOCIAL_BEHAVIORS=["rejection", "touch", "interactor", "interactee"]
RESTORE_FROM_CACHE_ENABLED=False

def parse_rejections_entry(data_entry, verbose=True):
    tokens = data_entry.split("_")
    tokens=tokens[:4] + [int(tokens[5])//CHUNKSIZE] + [int(tokens[7])]
    if len(tokens) != 6:
        if verbose:
            logger.error(f"Invalid entry: {data_entry}")
        return False, None

    return True, tokens
def parse_entry(data_entry, verbose=True):
    if "rejections" in data_entry:
        return parse_rejections_entry(data_entry, verbose=verbose)

    tokens = data_entry.split("_")
    if len(tokens) != 6:
        if verbose:
            logger.error(f"Invalid entry: {data_entry}")
        return False, None

    return True, tokens

def load_deg_data_gt_single_animal(experiment=None, identity=None, verbose=True):
    all_labels=[]

    counter=0
    ignored_suffixes=[".dvc", ".pkl"]
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

        chunk=parse_chunk(data_entry)
        labels_file=f"{DEG_DATA}/{data_entry}/{str(chunk).zfill(6)}_labels.csv"

        if not os.path.exists(labels_file):
            if verbose:
                logger.debug(f"{labels_file} not found")
            pb.update(1)
            continue


        if identity is not None:
            ret = filter_by_id(data_entry, experiment=experiment, identity=identity, chunksize=CHUNKSIZE, verbose=verbose)
        else:
            ret=True
        
        local_identity=parse_local_identity(data_entry)
        chunk=parse_chunk(data_entry)

        if not ret:
            pb.update(1)
            continue

        if not os.path.exists(labels_file):
            if verbose:
                logger.debug(f"{labels_file} not found")
            pb.update(1)
            continue

        labels=read_label_file(data_entry, labels_file, verbose=verbose, chunk=chunk, local_identity=local_identity)
        if labels is None:
            logger.debug(f"{labels_file} cannot be read")
            pb.update(1)
            continue

        labels["frame_number"]=labels["chunk"]*CHUNKSIZE+labels["frame_idx"]
        labels["entry"]=data_entry
        all_labels.append(labels)
        del labels
        counter+=1
        pb.update(1)

    if len(all_labels)>0:
        all_labels=pd.concat(all_labels, axis=0)
        return all_labels
    else:
        return None


def parse_experiment(data_entry):
    ret, tokens = parse_entry(data_entry)
    if ret:
        experiment = "_".join(tokens[:-2])
        return experiment
    else:
        return None


def parse_chunk(data_entry):
    ret, tokens = parse_entry(data_entry)
    if ret:
        chunk = int(tokens[-2])
        return chunk
    else:
        return None

def parse_local_identity(data_entry):
    ret, tokens = parse_entry(data_entry)
    if ret:
        local_identity = int(tokens[-1])
        return local_identity
    else:
        return None


def parse_number_of_animals(data_entry):
    ret, tokens = parse_entry(data_entry)
    if ret:
        number_of_animals=int(tokens[1].replace("X",""))
        return number_of_animals
    else:
        return None

def filter_by_id(data_entry, experiment, identity, chunksize=45000, verbose=True):

    local_identity=parse_local_identity(data_entry)
    chunk=parse_chunk(data_entry)
    experiment_=parse_experiment(data_entry)

    if local_identity is None:
        logger.warning("%s is corrupt. No local_identity can be parsed", data_entry)
        return False

    if chunk is None:
        logger.warning("%s is corrupt. No chunk can be parsed", data_entry)
        return False

    if experiment_ is None:
        logger.warning("%s is corrupt. No experiment can be parsed", data_entry)
        return False


    frame_number = chunk*chunksize
    try:
        table=get_local_identities_from_experiment(experiment_, int(frame_number))
    except Exception as error:
        logger.debug(error)
        return False

    identity_=int(table["identity"].loc[table["local_identity"]==local_identity])

    if not (experiment_ == experiment and identity_ == identity):
        return False

    return True


def annotate_rejections(deg):
    deg.loc[
        (deg["twitch"]==1) & (deg["rejection"]==1) & (deg["behavior"]=="inactive+micromovement"),
        "behavior"  
    ]="inactive+rejection"
    return deg

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


    def filter_by_time(self, min_time, max_time, cache):

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

    def load_deg_data_long(self, *args, min_time=-np.inf, max_time=+np.inf, stride=1, ground_truth=True,  cache=None, **kwargs):
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
            self.load_deg_data_gt(*args, experiment=self.experiment, **kwargs)
        else:
            self.load_deg_data_prediction(*args, **kwargs)

        if self.deg is not None:
            logger.info("Loaded DEG dataset of size %s", self.deg.shape)
            self.filter_by_time(min_time, max_time, cache=cache)
            self.deg["behavior"].loc[pd.isna(self.deg["behavior"])]="unknown"


    def load_deg_data(self, *args, min_time=-np.inf, max_time=+np.inf, stride=1, time_system="zt", ground_truth=True,  cache=None, **kwargs):

        self.load_deg_data_long(*args, min_time=min_time, max_time=max_time, stride=stride, ground_truth=ground_truth,  cache=cache, **kwargs)

        if self.deg is not None:
            self.deg_long=self.deg.copy() 
            before=time.time()            
            self.deg=self.annotate_two_or_more_behavs_at_same_time_(self.deg)
            after=time.time()
            logger.debug("Took %s seconds to annotate two or more behaviors", after-before)
            self.annotate_rejections_(self.deg)
            self.deg.sort_values("frame_number", inplace=True)
            self.deg=annotate_bouts(self.deg, variable="behavior")
            self.deg=annotate_bout_duration(self.deg, fps=150)
            self.deg["score"]=None
            
        if cache and self.deg is not None:
            path = os.path.join(cache, f"{self.experiment}_{min_time}_{max_time}_{stride}_deg_data.pkl")
            save_cache(path, self.deg)


    def annotate_rejections_(self, x):
        return annotate_rejections(x)


    @staticmethod
    def annotate_two_or_more_behavs_at_same_time_(*args, **kwargs):
        return annotate_two_or_more_behavs_at_same_time(*args, **kwargs)


    def load_deg_data_prediction(self, identity, verbose=True):
        raise NotImplementedError()


    def load_deg_data_gt(self, experiment, identity=None, verbose=True):
        if identity is None:
            identity=int(self.datasetnames[0].split("__")[1])
        
        labels=load_deg_data_gt_single_animal(experiment=experiment, identity=identity, verbose=verbose)
    
        if labels is None:
            logger.info(f"No labels found for {self.experiment}__{str(identity).zfill(2)}")
            return None

        else:
            id = self.experiment[:26] + "|" + str(identity).zfill(2)
            logger.info(f"id {id}: Number of label.csv found {labels.shape[0]}")
            animal = self.experiment + "__" + str(identity).zfill(2)
            labels["id"]=id
            labels["animal"]=animal

            if self.deg is None:
                self.deg = labels
            else:
                self.deg = pd.concat([self.deg, labels], axis=0)


def apply_semantic_rules(data_entry, labels):
    

    # if data_entry=="FlyHostel1_1X_2023-11-13_11-00-00_000127_000":
    #     import ipdb; ipdb.set_trace()

    tracks=pd.DataFrame({x: labels[x] for x in ["twitch", "rejection", "touch"]})
    tracks["frame_idx"]=np.arange(labels.shape[0])

    rows=labels.index[labels[["walk", "feed"]].values.sum(axis=1)==2]
    if len(rows)>0:
        logger.warning("Please set walk+feed bout to just walk or feed. Setting to feed now")
        labels.loc[rows, "walk"]=0
        labels.loc[rows, "feed"]=1


    incompatible_frames=(labels[["walk", "feed"]].values==1).sum(axis=1)>1
    assert incompatible_frames.sum()==0, f"Incompatible labels in {data_entry}, {labels.loc[incompatible_frames]}"

    incompatible_frames=(labels[["walk", "groom"]].values==1).sum(axis=1)>1
    assert incompatible_frames.sum()==0, f"Incompatible labels in {data_entry}, {labels.loc[incompatible_frames]}"

    # if one of the behavior tracks is active, set all others to 0
    for behavior in ["walk", "groom", "feed"]:
        rows=labels.index[np.where(labels[behavior]==1)]
        labels.loc[rows, labels.columns]=0
        labels.loc[rows, behavior]=1
    
    # if any of these behaviors is active, just call it micromovement and don't differentiate betwen them
    rows=labels.index[
        np.where(np.bitwise_and(
            labels["inactive"]==1,
            (labels[["micromovement", "twitch", "turn"]].values==1).any(axis=1)
        ))
    ]
    labels.loc[rows, ["micromovement", "twitch", "turn"]]=0
    labels.loc[rows, "inactive"]=1
    labels.loc[rows, "micromovement"]=1

    # if these two behaviors are present, set everything else to not present
    rows=labels.index[np.where(np.bitwise_and(
            labels["inactive"]==1,
            labels["pe"]==1,
        ))
    ]
    labels.loc[rows, labels.columns]=0
    labels.loc[rows, "inactive"]=1
    labels.loc[rows, "pe"]=1

    # if any of these behaviors is present and the animal is not inactive, set it to background
    rows=labels.index[
        np.where(
            np.bitwise_and(
            ~(labels["inactive"]==1),
            (labels[["micromovement", "twitch", "turn", "pe"]].values==1).any(axis=1)
        ))
    ]
    labels.loc[rows, labels.columns]=0
    labels.loc[rows, "background"]=1

    return labels, tracks


def melt_labels(labels, behaviors, tracks=None):
    frames, behav_ids = np.where(labels[behaviors].values==1)
    behav_sequence=[]

    for frame, behav_id in zip(frames, behav_ids):
        behav_sequence.append((frame, behaviors[behav_id]))
    labels=pd.DataFrame.from_records(behav_sequence)

    try:
        labels.columns=["frame_idx", "behavior"]
        if tracks is not None:
            labels=labels.merge(tracks, on="frame_idx")
        
        return labels
    except ValueError:
        return None
    

def read_label_file_raw(labels_file, **kwargs):
    labels_raw=pd.read_csv(labels_file, index_col=0)
    for key, value in kwargs.items():
        labels_raw[key]=value
    labels_raw["frame_idx"]=np.arange(labels_raw.shape[0])
    return labels_raw

def read_label_file(data_entry, labels_file, verbose=False, **kwargs):
   
    labels_raw=read_label_file_raw(labels_file, **kwargs)

    labels, tracks=apply_semantic_rules(data_entry, labels_raw)
    behavr_count_per_frame = (labels[BEHAVIORS].values==1).sum(axis=1)
    criteria = behavr_count_per_frame >= 1
    if not (criteria).all() and verbose:
        print(f"{data_entry} has {np.sum(behavr_count_per_frame < 1)} unlabeled frames")

    
    labels=melt_labels(labels, behaviors=BEHAVIORS, tracks=tracks)
    if labels is None:
        logger.warning("%s cannot be loaded", labels_file)
        return None
    
    for key, value in kwargs.items():
        labels[key]=value

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
