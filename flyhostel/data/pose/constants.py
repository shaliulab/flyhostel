import numpy as np
import os

thorax_pos=0
body_parts_chosen=[
    "thorax",
    "abdomen",
    "head",
    "foreLeft_Leg",
    "midLeftLeg",
    "rearLeftLeg",
    "foreRightLeg",
    "midRightLeg",
    "rearRightLeg",
    "leftWing",
    "rightWing",
    "proboscis"
]
skeleton=[(0, i) for i in range(1, len(body_parts_chosen)-1)] + [
        (body_parts_chosen.index("head"), body_parts_chosen.index("proboscis"))
]


criteria=["interpolate" for i in range(0, len(body_parts_chosen)-1)] + ["head"]

score_filter=[None for i in range(0, len(body_parts_chosen)-1)] + ["elbow"]
labels=[bp[:2] for bp in body_parts_chosen[:3]] + [".".join([bp[0]] + list(filter(lambda x: x not in [".", "_"] and x == x.upper(), bp))) for bp in body_parts_chosen[3:-1]] + ["pr"]

assert len(body_parts_chosen) == len(criteria)
assert len(body_parts_chosen) == len(score_filter)
assert len(body_parts_chosen) == len(labels)
assert len(body_parts_chosen) == len(criteria) == len(score_filter) == len(labels)



chunksize=45000
framerate=150        # framerate of input videos
bsoid_framerate=10
stride=framerate//bsoid_framerate
first_chunk=20
centroid_framerate=2
bsoid_chunksize=chunksize // stride
centroid_chunksize=centroid_framerate * chunksize // framerate
DATASETS=os.environ["MOTIONMAPPER_DATA"]
# "/Users/FlySleepLab Dropbox/Antonio/FSLLab/Projects/FlyHostel4/notebooks/datasets/"

palette="rainbow"
prefix="2023-07-19"
root_path=os.environ["BSOID_DATA"]
