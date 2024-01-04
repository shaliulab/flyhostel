import itertools
import os

import numpy as np


chunksize=45000
framerate=150        # framerate of input videos
bsoid_framerate=10
stride=framerate//bsoid_framerate
centroid_framerate=2
bsoid_chunksize=chunksize // stride
centroid_chunksize=centroid_framerate * chunksize // framerate
DATASETS=os.environ["MOTIONMAPPER_DATA"]
# "/Users/FlySleepLab Dropbox/Antonio/FSLLab/Projects/FlyHostel4/notebooks/datasets/"

palette="rainbow"
prefix="2023-07-19"
root_path=os.environ["POSE_DATA"]
ZT0_HOUR=11

# Bodyparts
bodyparts = [
    'thorax', 'abdomen', 'foreLeft_Leg', 'foreRightLeg', 'head', 'leftWing',
    'midLeftLeg', 'midRightLeg', 'proboscis', 'rearLeftLeg',
    'rearRightLeg', 'rightWing'
]
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
interpolate_seconds={bp: 30 for bp in bodyparts}
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