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
version = "1.1.3"

with open(f"{PKG_NAME}/__init__.py", "w") as fh:
    fh.write(f"__version__ = '{version}'\n")

setup(
    name=PKG_NAME,
    version=version,
    packages = find_packages(),
    extras_require={
        "sensor": ["pyserial"],
        "quant": [
            "recordtype",
            "trajectorytools-shaliulab==0.3.5"
        ],
        "dropbox": ["dropy"]
    },
    install_requires=[
        "zeitgeber",
        "matplotlib",
        "pyaml",
        "imgstore-shaliulab>=0.4.0"
    ],
    entry_points={
        "console_scripts": [
            "fh=flyhostel.__main__:main",
            ]
    },
)



