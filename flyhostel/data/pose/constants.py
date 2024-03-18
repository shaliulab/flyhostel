import itertools
import os

import numpy as np


chunksize=45000
framerate=150        # framerate of input videos
centroid_framerate=2
DATASETS=os.environ["MOTIONMAPPER_DATA"]
# "/Users/FlySleepLab Dropbox/Antonio/FSLLab/Projects/FlyHostel4/notebooks/datasets/"

palette="rainbow"
prefix="2023-07-19"
root_path=os.environ["POSE_DATA"]
ZT0_HOUR=11

# Bodyparts
bodyparts_wo_joints = [
    'thorax', 'abdomen', 'foreLeft_Leg', 'foreRightLeg', 'head', 'leftWing',
    'midLeftLeg', 'midRightLeg', 'proboscis', 'rearLeftLeg',
    'rearRightLeg', 'rightWing'
]

WITH_JOINTS=True
def get_bodyparts():
    if WITH_JOINTS:
        bodyparts=[
            "thorax", "abdomen", "head", "proboscis",
            "rW", "lW",
            "fRL","mRL","rRL",
            "fLL","mLL","rLL",
            "fRLJ","mRLJ","rRLJ",
            "fLLJ","mLLJ","rLLJ",       
        ]
        return bodyparts
    else:
        return bodyparts_wo_joints

bodyparts=get_bodyparts()


legs = [bp for bp in bodyparts if "leg" in bp.lower()]
wings = [bp for bp in bodyparts if "wing" in bp.lower()]
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

min_score={bp: 0.5 for bp in bodyparts}
min_score["proboscis"]=0.8

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
DIST_MAX_MM=4
SQUARE_HEIGHT=100
SQUARE_WIDTH=100
MIN_INTERACTION_DURATION=1 # seconds
inactive_states=["inactive", "pe_inactive", "inactive+micromovement", "inactive+twitch"]
