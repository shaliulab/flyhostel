import pathlib
import os.path
import os
from setuptools import setup, find_packages
import json

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

PKG_NAME = "flyhostel"
version = "1.1.0"

with open(f"{PKG_NAME}/__init__.py", "w") as fh:
    fh.write(f"__version__ = '{version}'\n")

setup(
    name=PKG_NAME,
    version=version,
    packages = find_packages(),
    extras_require={
        "sensor": ["pyserial"],
        "plotting": ["zeitgeber"],
        "quant": ["recordtype", "zeitgeber", "trajectorytools"],
        "dropbox": ["dropy"]
    },
    entry_points={
        "console_scripts": [
            "fh=flyhostel.__main__:main",
            "fh-sensor=flyhostel.sensors.__main__:main",
            "fh-upload=flyhostel.data.upload:main",
            "fh-copy=flyhostel.data.idtrackerai:copy",
            "fh-simulate=flyhostel.quantification.modelling.main:main",
            ]
    },
)



