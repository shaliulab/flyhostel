"""Every lab-specific path flyhostel uses, in one place.

    from flyhostel import lab_paths
    lab_paths.FLYHOSTEL_CACHE

Each entry is read once, when this module is first imported: the environment variable named next
to it, else the default shown (the lab server's layout). To run flyhostel elsewhere, set the
variables; do not edit the defaults. `~` is the user's home folder (os.path.expanduser: %USERPROFILE%
on Windows, never needs $HOME). Nothing here is created or required at import time: a default that
only exists on the lab server fails only when the code that needs it runs. $FLYHOSTEL_METADATA,
$DEG_PROJECTS and $NEXTFLOW_PIPELINES mean the same as in the viewer's
idtrackerai_validator_server/lab_paths.py.

Not here: $FLYHOSTEL_VIDEOS, read when it is used (flyhostel_videos() below, or
`os.environ["FLYHOSTEL_VIDEOS"]` where it is required), and the other variables some scripts require
without a default ($FLYHOSTEL_RESULTS, $LTA_DATA, $SLEAP_DATA, $SLEAP_PROJECT_PATH, $DEEPETHOGRAM_DATA,
$DEEPETHOGRAM_PROJECT_PATH); paths inside an experiment folder (`<basedir>/flyhostel/...`: the
pipeline's layout, not the lab's); conda environments found next to this one (roi_proposer/propose.py);
facts of the operating system (/etc/machine-id, /etc/hostname) and the system's temporary folder
(tempfile: $TMPDIR / %TEMP%). The CVAT login ($CVAT_HOST, $CVAT_USERNAME, $CVAT_PASSWORD) and the
incubator ($INCUBATOR_URL) are read in human_validation/cvat/constants.py and data/environment.py.
The flyhostel/modelling submodule (flyhostel-modelling) is a separate repository, and the shell
scripts and systemd units in scripts/ keep their own paths.
"""
import os


def _env(name, default):
    """$name, else default (as given: a `~` in $name is not expanded)."""
    return os.environ.get(name, default)


HOME = os.path.expanduser("~")


# --- Data roots on the lab server ---------------------------------------------------------------
# where experiments are recorded: <FLYHOSTEL_VIDEOS>/FlyHostelN/NX/DATE_TIME. Read $FLYHOSTEL_VIDEOS
# with flyhostel_videos() when the code fell back to this default; the places that never read
# $FLYHOSTEL_VIDEOS (centroids loader, video server, roi_proposer --videos, the config file's
# default) use FLYHOSTEL_VIDEOS_DEFAULT as is
FLYHOSTEL_VIDEOS_DEFAULT = "/flyhostel_data/videos"


def flyhostel_videos():
    """$FLYHOSTEL_VIDEOS (read now, not at import), else FLYHOSTEL_VIDEOS_DEFAULT"""
    return os.environ.get("FLYHOSTEL_VIDEOS", FLYHOSTEL_VIDEOS_DEFAULT)


# cache of the loaders (centroids, pose, behavior, ...: the cache= argument)
FLYHOSTEL_CACHE = _env("FLYHOSTEL_CACHE", "/flyhostel_data/cache")
# CSV exports of the metadata sheets (centroids loader)
FLYHOSTEL_METADATA = _env("FLYHOSTEL_METADATA", "/flyhostel_data/metadata")
# temperature / humidity / light of the incubators (fh-environment sync)
ENVIRONMENT_DB = _env("FLYHOSTEL_ENVIRONMENT", "/flyhostel_data/environment/environment.db")
# proboscis extension annotations: $PE_ANNOTATIONS_DB, else $PE_DB, else the default;
# the resegmentation (proboscis_extension/segmentation.py) uses $RESEG_DB, else $PE_DB, else the default
PE_DB = _env("PE_DB", "/flyhostel_data/videos/pe_annotations.db")
PE_ANNOTATIONS_DB = os.environ.get("PE_ANNOTATIONS_DB") or PE_DB
RESEG_DB = _env("RESEG_DB", PE_DB)
# deepethogram projects: DEG/FlyHostel_deepethogram (150 fps), DEG/FlyHostel_deepethogram_47fps,
# DEG-REJECTIONS/rejections_deepethogram (touch rejections)
DEG_PROJECTS = _env("DEG_PROJECTS", "/flyhostel_data/fiftyone/FlyBehaviors")
DEG_ROOT = DEG_PROJECTS + "/DEG"
DEG_PROJECT_47FPS = DEG_ROOT + "/FlyHostel_deepethogram_47fps/"
DEG_REJECTIONS_PROJECT = DEG_PROJECTS + "/DEG-REJECTIONS/rejections_deepethogram/"
# SLEAP labels whose skeleton flyhostel.data.pose.sleap draws with (a Mac Dropbox path: only loaded
# when sleap is installed, skipped when missing)
SLEAP_REF_LABELS = _env("SLEAP_REF_LABELS",
                        "/Users/FlySleepLab_Dropbox/Data/flyhostel_data/fiftyone/FlyBehaviors/labels_train.slp")
# logs and plots of the incubator sensors (sensors/node.py)
ETHOSCOPE_SENSORS = _env("ETHOSCOPE_SENSORS", "/ethoscope_data/sensors")
# clips served by the behavior viewer (human_validation/behavior_interruptions)
BEHAVIOR_VIEWER_DATASETS = _env("BEHAVIOR_VIEWER_DATASETS", "/opt/behavior-viewer/static/datasets")


# --- vsc-scripts' nextflow pipelines (experiment lists) ----------------------------------------------
NEXTFLOW_PIPELINES = _env("NEXTFLOW_PIPELINES", "/home/vibflysleep/opt/vsc-scripts/nextflow/pipelines")
# experiments and animals of the behavior prediction (utils.load_experiments, classifierv2)
ANIMALS_CSV = NEXTFLOW_PIPELINES + "/behavior_prediction/animals.csv"
# groups of the interaction detection (utils.load_experiments)
INTERACTION_INDEX_CSV = NEXTFLOW_PIPELINES + "/interaction_detection/index.csv"
# groups of classifierv2
FLY_GROUPS_CSV = NEXTFLOW_PIPELINES + "/interaction_detection/fly_groups.csv"
# videos of the interaction detection (data/interactions/constants.py)
INTERACTION_VIDEOS_CSV = NEXTFLOW_PIPELINES + "/interaction_detection/videos.csv"
# the pose estimation's config, its sleap_model is the pose model version of the ethogram
POSE_NEXTFLOW_CONFIG = NEXTFLOW_PIPELINES + "/pose_estimation/nextflow.config"
# animals.csv of the user running scripts/main.py (backup: ~/opt/vsc-scripts, not $NEXTFLOW_PIPELINES)
BACKUP_ANIMALS_CSV = _env("BACKUP_ANIMALS_CSV",
                          os.path.join(HOME, "opt/vsc-scripts/nextflow/pipelines/behavior_prediction/animals.csv"))


# --- Other lab code checked out on the server ----------------------------------------------------
# drosophila-social-network-analysis checkout (data/interactions/export.py)
DSNA_FOLDER = _env("DSNA_FOLDER", "/home/vibflysleep/opt/drosophila-social-network-analysis")
# motionmapperpy project of the long timescale analysis (data/interactions/mmpy_umap.py)
MMPY_PROJECT = _env("MMPY_PROJECT", "/home/vibflysleep/opt/long_timescale_analysis/slurm/FlyHostel_long_timescale_analysis/")
# B-SOID checkout, added to sys.path by data/pose/bsoid_interpolation.py
BSOID_DIR = _env("BSOID_DIR", "/home/vibflysleep/opt/B-SOID/")
# tripwire of data/deepethogram/video.build_key (fails on purpose when the function is called)
WATCH_BUILD_KEY = _env("WATCH_BUILD_KEY", "/home/vibflysleep/watch_build_key.txt")


# --- This package's configuration (~/.config, ~/.local/share) --------------------------------------
CONFIG_DIR = os.path.join(HOME, ".config", "flyhostel")
# settings of flyhostel.utils.load_config (made with DEFAULT_CONFIG on first use)
CONFIG_FILE = _env("FLYHOSTEL_CONFIG_FILE", os.path.join(CONFIG_DIR, "config.conf"))
# logging.config.dictConfig of flyhostel, applied on import when the file exists
# (else flyhostel/default_logging.yaml)
LOGGING_CONFIG = _env("FLYHOSTEL_LOGGING_CONFIG", os.path.join(CONFIG_DIR, "logging.yaml"))
# training data and weights of the roi-proposer
ROI_PROPOSER_DIR = _env("FLYHOSTEL_ROI_PROPOSER_DIR", os.path.join(HOME, ".local", "share", "flyhostel", "roi_proposer"))
# logging of the computer_vision scripts (qab2022 dataset)
QAB2022_LOGGING_FILE = _env("QAB2022_LOGGING_FILE", os.path.join(HOME, ".config", "qab2022.yaml"))


# --- Temporary and cache files ------------------------------------------------------------------------
# joblib cache of the CVAT integration (human_validation/cvat/main.py)
JOBLIB_CACHE = _env("FLYHOSTEL_JOBLIB_CACHE", "/tmp/joblib_cache")
# image SLEAP needs to build a Video for get_color_manager
SLEAP_PNG = _env("FLYHOSTEL_SLEAP_PNG", os.path.expanduser("~/sleap.png"))


# --- System files (Linux) ------------------------------------------------------------------------------
# AuthPass of the mail sender (data/df/mail.py)
SSMTP_CONF = _env("SSMTP_CONF", "/etc/ssmtp/ssmtp.conf")
# serial port of the incubator sensor (sensors/server.py, scripts/debug_sensors.py)
SENSOR_SERIAL_PORT = _env("FLYHOSTEL_SENSOR_PORT", "/dev/ttyACM0")
# fonts of the time label (data/video/utils.py: first one found) and the behavior annotations (ffmpeg)
DEJAVU_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEJAVU_SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LIBERATION_SANS = "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
LIBERATION_SANS_BOLD = "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
