"""
Constants variables for the sqlite3 module
"""

import os
from flyhostel.data.pose.constants import get_bodyparts

TABLES = [
    "METADATA", "IMG_SNAPSHOTS", "ROI_MAP", "VAR_MAP", "ROI_0",
    "IDENTITY", "CONCATENATION", "BEHAVIORS", "STORE_INDEX",
    "ENVIRONMENT", "AI", "ORIENTATION", "QC", "POSE", "LANDMARKS"
]

BEHAVIORS=["background", "inactive", "walk", "pe", "rejection", "groom", "fall"]

PRESETS = {
    "all": TABLES,
    "default": [
        "METADATA", "IMG_SNAPSHOTS", "ROI_MAP", "VAR_MAP",
        "CONCATENATION","STORE_INDEX",
        "ENVIRONMENT", "AI", "QC", "LANDMARKS",
    ],
    "roi0": [
        "ROI_0"
    ],
    "trajectory": [
        "ROI_0", "IDENTITY"
    ],
    "index": [
        "STORE_INDEX"
    ],
    "identity": [
        "CONCATENATION", "IDENTITY", "METADATA"
    ],
    "qc": ["QC"],
}

PRESETS["default_and_trajectory"] = PRESETS["default"] + PRESETS["trajectory"]

NODES=get_bodyparts()


RAISE_EXCEPTION_IF_METADATA_NOT_FOUND=True
METADATA_FILE = "metadata.csv"
