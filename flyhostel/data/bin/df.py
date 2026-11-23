import argparse
import logging

from flyhostel.data.df import get_free_fraction, notify_free_fraction
from flyhostel.configuration import load_config

logger = logging.getLogger(__name__)
config = load_config()

# read with defaults: a new installation (empty config) must still be able to import flyhostel.data.bin
# (`flyhostel` CLI); main() fails with a clear message when they are needed and missing
VIDEOS_FOLDER = config.get("videos", {}).get("folder")
EMAIL_ADDRESS = config.get("email", {}).get("destination")
THRESHOLD = config.get("email", {}).get("threshold")

def get_parser(ap=None):
    if ap is None:
        ap=argparse.ArgumentParser()
    ap.add_argument("--mail", default=None)
    return ap


def main(args=None, ap=None):

    if args is None:
        if ap is None:
            ap = get_parser()

        args = ap.parse_args()

    try:
        free_fraction = round(get_free_fraction(VIDEOS_FOLDER) * 100, 2)
        logger.debug(f"Free fraction: {free_fraction}")
    
        if free_fraction < THRESHOLD:
            logger.warning(f"Less than {THRESHOLD} % of flyhostel partition is free")
            notify_free_fraction(free_fraction, EMAIL_ADDRESS)
            logger.debug("Sent email")
    except Exception as error:
        logger.error(error)
        raise error

