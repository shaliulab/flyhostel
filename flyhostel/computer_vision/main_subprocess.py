import argparse
import logging
import logging.config
import os.path
import yaml
import joblib
import numpy as np
import subprocess
import shlex
# import imgstore.stores.multi as imgstore

LOGGING_FILE=os.path.join(os.environ["HOME"], ".config", "qab2022.yaml")

from library import generate_dataset
import utils

from confapp import conf
try:
    import local_settings
    conf += local_settings
except ImportError:
    pass

logger = logging.getLogger(__name__)

with open(LOGGING_FILE, "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

logging.config.dictConfig(config)

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment")
    return ap


def main():
    ap = get_parser()
    args = ap.parse_args()
    generate_dataset(args.experiment, compute_thresholds=conf.COMPUTE_THRESHOLDS, tolerance=conf.TOLERANCE, crop=conf.CROP, rotate=conf.ROTATE)

if __name__ == "__main__":
    main()
