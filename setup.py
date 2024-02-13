import pathlib
from setuptools import setup, find_packages

import warnings

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

PKG_NAME = "flyhostel"
version = "1.1.7"

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
        "pandas<=1.3.5",
        "confapp-shaliulab",
        "scikit-learn",
        "recordtype",
        "tqdm",
        "h5py",
        "hdf5storage",
        "yolov7tools==1.1",
        "vidio",
    ],
    entry_points={
        "console_scripts": [
            "fh=flyhostel.__main__:main",
            "fh-server=flyhostel.server.server:main",
            "fh-validate=flyhostel.data.bin.dashboard:main",
            "missing-chunk-detector=flyhostel.utils.missing_chunk_detector:main",
            "compile-pose=flyhostel.data.bin.pose:main",
            "train-umap=flyhostel.data.bin.umap:main",
            "project-pose=flyhostel.data.bin.behavior:main",
            "predict-behavior=flyhostel.data.bin.ethogram:main",
            "draw-ethogram=flyhostel.data.bin.ethogram:draw_ethogram",
            "compute-interactions=flyhostel.data.bin.interactions:main",
            "filter-pose=flyhostel.data.bin.filter_pose:main",
            "preprocess-pose=flyhostel.data.bin.pose:preprocess",
            "annotate-video=flyhostel.data.bin.movie:main",
            "list-frames-with-no-animals=flyhostel.data.sqlite3.missing_animals:main",
            "make-identogram=flyhostel.data.bin.human_validation:main",
            "auto-annotate-qc=flyhostel.data.bin.human_validation:annotate_scene_quality",
            "make-space-time-images=flyhostel.data.bin.human_validation.annotate:main",
            "integrate-human-annotations=flyhostel.data.bin.human_validation.integrate:main",
            "cvat-label-constructor=flyhostel.data.bin.human_validation.label_constructor:main",
            ]
    },
)


warnings.warn("Make sure that idtrackerai, torch, torchvision, confapp, zeitgeber, trajectorytools, feed_integration, dropy are installed")
