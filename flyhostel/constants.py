import os
import os.path

from flyhostel.quantification.modelling.constants import *
RAW="data"
ANNOTATED="dt_sleep"
BINNED="dt_binned"
DEFAULT_TIME_WINDOW_LENGTH=10
DEFAULT_VELOCITY_CORRECTION_COEF=0.2
DEFAULT_MIN_TIME_IMMOBILE=300
DEFAULT_SUMMARY_TIME_WINDOW=30*60
DEFAULT_REFERENCE_HOUR=6
DEFAULT_MAX_BRIEF_AWAKENING=30 # seconds

ANNOTATION_PARAMETERS = {
    "min_time_immobile": DEFAULT_MIN_TIME_IMMOBILE,
    "velocity_correction_coef": DEFAULT_VELOCITY_CORRECTION_COEF,
    "time_window_length": DEFAULT_TIME_WINDOW_LENGTH,
    "max_brief_awakening": DEFAULT_MAX_BRIEF_AWAKENING,
}


CONFIG_FILE = os.path.join(os.environ.get("HOME", "."), ".config", "flyhostel", "config.conf")
LOGGING_CONFIG = os.path.join(os.environ.get("HOME", "."), ".config", "flyhostel", "logging.yaml")

DEFAULT_CONFIG = {"videos": {"folder": "/flyhostel_data/videos"}, "logging": {"sensors": "WARNING", "arduino": "WARNING"}}
N_JOBS = 1
ETHOGRAM_FREQUENCY = 300
INDEX_FORMAT=".npz"
COLORS = {"T": [(249, 168, 37), (255, 245, 157)], "F": [(69, 39, 160), (206, 147, 216)]}
NUMBER_OF_JOBS_FOR_COPYING_TRAJECTORIES=1
ANALYSIS_FOLDER="idtrackerai"
OUTPUT_FOLDER="flyhostel"
NUMBER_OF_JOBS_TO_GENERATE_INDIVIDUAL_VIDEOS=1