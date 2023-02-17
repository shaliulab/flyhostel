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
    DOWNLOAD_FLYHOSTEL_METADATA=os.environ.get("DOWNLOAD_FLYHOSTEL_METADATA", None)
    assert DOWNLOAD_FLYHOSTEL_METADATA is not None and os.path.exists(DOWNLOAD_FLYHOSTEL_METADATA)

except AssertionError:
    warnings.warn(
        """
        download-behavioral-data not found.
        Automatic download of metadata not available.
        Please ensure the DOWNLOAD_FLYHOSTEL_METADATA environment variable is set
        and pointing to a download-behavioral-data executable
        """)
    DOWNLOAD_FLYHOSTEL_METADATA = None
