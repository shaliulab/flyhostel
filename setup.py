import pathlib
from setuptools import setup, find_packages

import warnings

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

PKG_NAME = "flyhostel"
version = "1.1.6"

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
    },
    install_requires=[
        "zeitgeber>=0.0.2",
        "matplotlib",
        "pyaml",
        "imgstore-shaliulab>=0.4.0",
        "confapp-shaliulab",
        "scikit-learn",
        "recordtype",
        "tqdm",
        "yolov7tools==1.1",
        #"feed_integration",
    ],
    entry_points={
        "console_scripts": [
            "fh=flyhostel.__main__:main",
            "fh-server=flyhostel.server.server:main",
            "fh-validate=flyhostel.data.bin.dashboard:main",
            "missing-chunk-detector=flyhostel.utils.missing_chunk_detector:main",
            ]
    },
)



warnings.warn("Make sure that idtrackerai, torch, torchvision, confapp, zeitgeber, trajectorytools, feed_integration, dropy are installed")
