import os
import os.path

RAW_DATA="data.csv"
ANNOTATED_DATA="dt_sleep.csv"
BINNED_DATA="dt_binned.csv"
DEFAULT_TIME_WINDOW_LENGTH=10
DEFAULT_VELOCITY_CORRECTION_COEF=0.02
DEFAULT_MIN_TIME_IMMOBILE=300
DEFAULT_SUMMARY_TIME_WINDOW=30*60
DEFAULT_REFERENCE_HOUR=6

CONFIG_FILE = os.path.join(os.environ["HOME"], ".config", "flyhostel.conf")
DEFAULT_CONFIG = {"videos": {"folder": "/flyhostel_data/videos"}, "logging": {"sensors": "WARNING", "arduino": "WARNING"}}
N_JOBS = 1
ETHOGRAM_FREQUENCY = 300
INDEX_FORMAT=".npz"
