import os
import argparse
import pickle
from .utils import load_animals
from .bsoid_interpolation import bsoid_interpolation
from .median_filter import median_filter
DATASETS=os.environ["MOTIONMAPPER_DATA"]

def preprocess(experiment, chunks, n_jobs):
    """
    Entrypoint to preprocess the pose estimates
    1) Impute NaNs
    
    TODO explain
    """
    h5s_pandas, indices, params=bsoid_interpolation(experiment, chunks, n_jobs=n_jobs)
    h5s_pandas=median_filter(h5s_pandas)

    animals = load_animals(experiment)
    for animal_id, animal in enumerate(animals):
        df=h5s_pandas[animal_id]
        df_index=indices[animal_id]
        out_file=f"{DATASETS}/{animal}_positions.h5"
        print(f"--> {out_file}")
        df.to_hdf(out_file, key="pose")
        df_index.to_hdf(f"{DATASETS}/{animal}_positions.h5", key="index")

    with open(f"{DATASETS}/{experiment}.pkl", "wb") as handle:
        pickle.dump(params, handle)


def get_parser():

    ap = argparse.ArgumentParser()
    group=ap.add_mutually_exclusive_group()
    group.add_argument("--experiment", type=str, help="Experiment key (FlyHostelX_XX_XXXXX)")
    group.add_argument("--dbfile", type=str, help="Path to .db file")
    ap.add_argument("--n-jobs", type=int, default=None, required=False)
    ap.add_argument("--chunks", nargs="+", type=int, required=True)
    return ap

def main():
    ap = get_parser()
    args=ap.parse_args()
    if args.experiment is not None:
        experiment = args.experiment
    else:
        dbfile=args.dbfile
        experiment = "_".join(os.path.dirname(dbfile).split(os.path.sep)[-3:])


    chunks=args.chunks
    n_jobs=args.n_jobs

    return preprocess(experiment, chunks, n_jobs)
