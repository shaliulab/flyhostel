"""
Constants variables for the sqlite3 module
"""

import os
import warnings

TABLES = [
    "METADATA", "IMG_SNAPSHOTS", "ROI_MAP", "VAR_MAP", "ROI_0",
    "IDENTITY", "CONCATENATION", "BEHAVIORS", "STORE_INDEX",
    "ENVIRONMENT", "AI", "ORIENTATION"
]

RAISE_EXCEPTION_IF_METADATA_NOT_FOUND=True
METADATA_FILE = "metadata.csv"

try:
    DOWNLOAD_BEHAVIORAL_DATA=os.environ.get("DOWNLOAD_BEHAVIORAL_DATA", None)
    assert DOWNLOAD_BEHAVIORAL_DATA is not None and os.path.exists(DOWNLOAD_BEHAVIORAL_DATA)

except AssertionError:
    warnings.warn(
        """
        download-behavioral-data not found.
        Automatic download of metadata not available.
        Please ensure the DOWNLOAD_BEHAVIORAL_DATA environment variable is set
        and pointing to a download-behavioral-data executable
        """)
    DOWNLOAD_BEHAVIORAL_DATA = None
