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

# attention. you need to update the numbers ALSO in the imgstore/__init__.py file
version = "1.0.1"

CONFIG_FILE = "/etc/flyhostel.conf"
if not os.access(CONFIG_FILE, os.W_OK):
    CONFIG_FILE = os.path.join(os.environ["HOME"], ".config", "flyhostel.conf")

with open(f"{PKG_NAME}/_version.py", "w") as fh:
    fh.write(f"__version__ = '{version}'\n")
with open(f"{PKG_NAME}/__init__.py", "w") as fh:
    fh.write(f"CONFIG_FILE = '{CONFIG_FILE}'\n")

# This call to setup() does all the work
setup(
    name=PKG_NAME,
    version=version,
    #description="High resolution monitoring of Drosophila",
    #long_description=README,
    #long_description_content_type="text/markdown",
    ##url="https://github.com/realpython/reader",
    #author="Antonio Ortega",
    #author_email="antonio.ortega@kuleuven.be",
    #license="MIT",
    #classifiers=[
    #    "License :: OSI Approved :: MIT License",
    #    "Programming Language :: Python :: 3",
    #    "Programming Language :: Python :: 3.7",
    #],
    packages = find_packages(),
    #include_package_data=True,
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
            ]
    },
)


if not os.path.exists(CONFIG_FILE):
    videos_folder = "/flyhostel_data/videos"
    os.makedirs(videos_folder, exist_ok=True)
    config = {"videos": {"folder": videos_folder}, "logging": {"sensors": "WARNING", "arduino": "WARNING"}}
    with open(CONFIG_FILE, "w") as fh:
        json.dump(config, fh)

