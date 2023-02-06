import pathlib
from setuptools import setup, find_packages

import warnings

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

PKG_NAME = "flyhostel"
version = "1.1.5"

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
        "zeitgeber>=0.0.2",
        "matplotlib",
        "pyaml",
        "imgstore-shaliulab>=0.4.0",
        "confapp-shaliulab",
        "sklearn",
        "recordtype",
        "tqdm",
        #"feed_integration",
    ],
    entry_points={
        "console_scripts": [
            "fh=flyhostel.__main__:main",
            ]
    },
)



warnings.warn("Make sure that idtrackerai, torch, torchvision, confapp, zeitgeber, trajectorytools, feed_integration, dropy are installed")
