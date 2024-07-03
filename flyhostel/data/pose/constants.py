import itertools
import os
import logging

logger=logging.getLogger(__name__)

try:
    from motionmapperpy import setRunParameters
    WAVELET_DOWNSAMPLE=setRunParameters().wavelet_downsample
    MOTIONMAPPER_PARAMS=setRunParameters()
except ModuleNotFoundError:
    logger.warning("motionmapper not available")
    WAVELET_DOWNSAMPLE=None
    MOTIONMAPPER_PARAMS=None


chunksize=45000
framerate=150        # framerate of input videos
centroid_framerate=2
# "/Users/FlySleepLab Dropbox/Antonio/FSLLab/Projects/FlyHostel4/notebooks/datasets/"

palette="rainbow"
prefix="2023-07-19"
ZT0_HOUR=11

# Bodyparts
bodyparts_wo_joints = [
    "thorax", "abdomen", "head", "proboscis",
    "rW", "lW",
    "fRL","mRL","rRL",
    "fLL","mLL","rLL",      
]

WITH_JOINTS=True
def get_bodyparts():
    if WITH_JOINTS:
        bodyparts=bodyparts_wo_joints + [
            "fRLJ","mRLJ","rRLJ",
            "fLLJ","mLLJ","rLLJ",       
        ]
        return bodyparts
    else:
        return bodyparts_wo_joints

bodyparts=get_bodyparts()


legs = [bp for bp in bodyparts if "L" in bp.lower()]
wings = [bp for bp in bodyparts if "W" in bp.lower()]
core = ["thorax", "abdomen", "head", "proboscis"]


# for documentation purpose
filters={"nanmedian": {"window_size": 0.2, "min_window_size": 40, "order": 0}, "nanmean": {"window_size": 0.2, "min_window_size":10, "order": 1}}


body_parts_chosen=bodyparts
thorax_pos=bodyparts.index("thorax")

skeleton=[(0, i) for i in range(1, len(body_parts_chosen)-1)] + [
        (body_parts_chosen.index("head"), body_parts_chosen.index("proboscis"))
]


criteria=["interpolate" for i in range(0, len(body_parts_chosen)-1)] + ["head"]

score_filter=[None for i in range(0, len(body_parts_chosen)-1)] + [None]
# score_filter=[None for i in range(0, len(body_parts_chosen)-1)] + ["elbow"]
labels=[bp[:2] for bp in body_parts_chosen[:3]] + [".".join([bp[0]] + list(filter(lambda x: x not in [".", "_"] and x == x.upper(), bp))) for bp in body_parts_chosen[3:-1]] + ["pr"]

assert len(body_parts_chosen) == len(criteria)
assert len(body_parts_chosen) == len(score_filter)
assert len(body_parts_chosen) == len(labels)
assert len(body_parts_chosen) == len(criteria) == len(score_filter) == len(labels)

# Pose filtering
interpolate_seconds={bp: 3 for bp in bodyparts}
interpolate_seconds["proboscis"]=0.5


min_score={
    "head":0,"thorax":0,
    "abdomen":0,"proboscis":0.75,
    "lW":0.1,"rW":0.1,
    "fLL":0.6,"fRL":0.6,
    "mLL":0.75,"mRL":0.75,
    "rLL":0.75,"rRL":0.75,
    "fLLJ":0.5,"fRLJ":0.5,
    "mLLJ":0.75,"mRLJ":0.75,
    "rLLJ":0.75,"rRLJ":0.75
}

bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))
bodyparts_speed=list(itertools.chain(*[[bp + "_speed"] for bp in bodyparts]))
MIN_TIME=float("-inf")
MAX_TIME=float("+inf")

MAX_JUMP_MM=1
JUMP_WINDOW_SIZE_SECONDS=0.5
PARTITION_SIZE=framerate*3600
PX_PER_CM=175
APPLY_MEDIAN_FILTER=False
ROI_WIDTH_MM=60
SQUARE_HEIGHT=100
SQUARE_WIDTH=100
DIST_MAX_MM=2.5
MIN_INTERACTION_DURATION=.3 # seconds
MIN_TIME_BETWEEN_INTERACTIONS=0.5 # seconds. Interactions closer than this in time become one


inactive_states=["inactive", "inactive+pe", "inactive+micromovement", "inactive+twitch", "background"]
DEFAULT_FILTERS=["rle", "jump"]

DEG_DATA=os.path.join(os.environ["DEEPETHOGRAM_PROJECT_PATH"], "DATA")

BEHAVIOR_IDX_MAP={
    "walk": (1,),
    "groom": (2,),
    "feed": (3,),
    "background": (0,),
    "inactive+micromovement": (5,7),
    "inactive+rejection": (5,7,9),
    "inactive+pe": (4,5),
    "inactive": (5,),
}
