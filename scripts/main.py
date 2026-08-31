import argparse
import os.path
import logging

import pandas as pd
import numpy as np
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.groups.group import FlyHostelGroup
from tqdm.auto import tqdm
from flyhostel.utils.utils import RsyncFailure
from flyhostel.utils import get_basedir, get_dbfile


logger=logging.getLogger(__name__)

chunks=list(range(0, 400))

METADATA_FILE=os.path.join(
    os.environ["HOME"],
    "opt/vsc-scripts/nextflow/pipelines/behavior_prediction/animals.csv"
)

def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flyhostel", type=str, required=True, help="If passed, only experiments from this flyhostel will be backed up. Example: FlyHostel1")
    ap.add_argument("--number-of-animals", type=int, nargs="+", default=None, help="If passed, only experiments with this group size will be backed up")
    ap.add_argument("--dest", type=str, required=True, help="Path to the data repository where the flyhostel data will be mirrored")
    ap.add_argument("--chronologically", action="store_true", default=False, help="Sort experiments by date only, and not by number of animals + date (default)")
    ap.add_argument("-n", "--dry-run", action="store_true", default=False)
    ap.add_argument("-D", "--debug", action="store_true", default=False)
    return ap


def report_failed_experiment(experiment, status="FAIL"):
    with open("status.txt", "a") as handle:
        handle.write(f"{experiment},{status}\n")


def backup_experiment(experiment, single_meta, **kwargs):
    status=None
    basedir=get_basedir(experiment)
    if not os.path.exists(basedir):
        report_failed_experiment(experiment, "NOT FOUND")
        return 1

    identities=single_meta["identity"].tolist()

    number_of_animals=int(experiment.split("_")[1].replace("X", ""))
    assert len(identities)==number_of_animals

    loaders=[FlyHostelLoader(experiment, identity) for identity in identities]
    group=FlyHostelGroup.from_list(loaders, protocol=None)
    try:
        status=group.backup(chunks=chunks, **kwargs)
    except RsyncFailure as error:
        logger.error(error)
        report_failed_experiment(experiment)

    if status is not None:
        report_failed_experiment(experiment, status=status)

def main():

    ap = get_parser()
    args=ap.parse_args()

    assert os.path.exists(args.dest)

    metadata=pd.read_csv(METADATA_FILE, header=None)
    metadata.columns = [
        "experiment",
        "basedir",
        "identity",
        "done",
        "status",
        "comment"
    ]
    metadata=metadata.query("comment=='SELECT'")

    metadata["flyhostel"]=metadata["experiment"].str.slice(start=0, stop=10)
    metadata["number_of_animals"]=[int(x[1].replace("X", "")) for x in metadata["experiment"].str.split("_")]
    if args.flyhostel is not None:
        metadata=metadata.query("flyhostel == @args.flyhostel")
    if args.number_of_animals is not None:
        metadata=metadata.query("number_of_animals.isin(@args.number_of_animals)")

    dbfiles=[]
    for basedir in metadata["basedir"]:
        try:
            dbfile=get_dbfile(basedir)
        except AssertionError:
            dbfile=np.nan
        dbfiles.append(dbfile)

    metadata["dbfile"]=dbfiles
    experiments_dropped=metadata.loc[metadata["dbfile"].isna()]["experiment"].drop_duplicates().tolist()
    if len(experiments_dropped)>0:
        logger.warning("Dropping %s experiments: %s", len(experiments_dropped), experiments_dropped)
        metadata=metadata.loc[~(metadata["experiment"].isin(experiments_dropped))]

    experiments=metadata["experiment"].drop_duplicates().tolist()

    if args.chronologically:
        experiments=sorted(experiments, key=lambda x: x.split("_")[2])
    print(f"Will backup {len(experiments)} experiments")

    with open("status.txt", "w") as handle:
        handle.write("experiment,status\n")

    pb=tqdm(total=len(experiments))
    for experiment, single_meta in metadata.groupby("experiment"):
        pb.set_description(experiment)
        try:
            backup_experiment(experiment, single_meta, path=args.dest, dry_run=args.dry_run, debug=args.debug)
        except AssertionError as error:
            logger.error("Cannot backup %s", experiment)
            logger.error(error)
        pb.update(1)

if __name__ == "__main__":
    main()
