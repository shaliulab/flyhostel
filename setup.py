import pathlib
import warnings

from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

PKG_NAME = "flyhostel"
version = "1.1.7"

install_requires = [
    "zeitgeber>=0.0.2",
    "matplotlib",
    "pyaml",
    "imgstore-shaliulab>=0.4.0",
    "pandas",
    "confapp-shaliulab",
    "scikit-learn",
    "recordtype",
    "tqdm",
    "h5py",
    "hdf5storage",
    "vidio",
    # vidio requires opencv-python-headless unconstrained; idtrackerai pins
    # opencv-python<4, which forces numpy 1.x. Keep both on the same ABI.
    "opencv-python-headless<4",
    "webcolors",
    "GitPython",
    "colour",
    "pyarrow",
    # "sleap-io",
    #"cupy>=12.2.0",
    #"cudf>=23.10.02",
]


def vendored(dist_name, subdir, extras=None):
    """Build a PEP 508 direct reference to a submodule under libraries/.

    The path must be absolute: pip resolves relative paths against the working
    directory, not against this file. `dist_name` must be the name declared in
    that package's own setup.py, or pip rejects the install as a name mismatch.
    """
    path = HERE / "libraries" / subdir
    if not ((path / "setup.py").exists() or (path / "pyproject.toml").exists()):
        warnings.warn(
            "libraries/{} is empty, so {} will not be installed. "
            "Run: git submodule update --init --recursive".format(subdir, dist_name)
        )
        return None

    name = "{}[{}]".format(dist_name, ",".join(extras)) if extras else dist_name
    return "{} @ {}".format(name, path.resolve().as_uri())


for requirement in (
    vendored("ethoscopy", "ethoscopy"),
    # The distribution is named idtrackerai-shaliulab, not idtrackerai.
    # [gpu] pulls in torch and torchvision.
    vendored("idtrackerai-shaliulab", "idtrackerai", extras=["gpu"]),
):
    if requirement is not None:
        install_requires.append(requirement)


setup(
    name=PKG_NAME,
    version=version,
    packages=find_packages(exclude=["libraries", "libraries.*"]),
    extras_require={
        "sensor": ["pyserial"],
        "quant": [
            "recordtype",
            "trajectorytools-shaliulab==0.3.5"
        ],
        "detection": ["yolov7tools==1.1"],
    },
    include_package_data=True,
    package_data={"flyhostel": ["default_logging.yaml"]},
    install_requires=install_requires,
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
            "export-filter-pose=flyhostel.data.bin.export_filter_pose:main",
            "preprocess-pose=flyhostel.data.bin.pose:preprocess",
            "annotate-video=flyhostel.data.bin.movie:main",
            "list-frames-with-no-animals=flyhostel.data.sqlite3.missing_animals:main",
            "make-identogram=flyhostel.data.bin.human_validation.annotate:main",
            "export-images=flyhostel.data.bin.human_validation.export:main",
            "auto-annotate-qc=flyhostel.data.bin.human_validation.qc:main",
            "integrate-human-annotations=flyhostel.data.bin.human_validation.integrate:main",
            "save-human-annotations=flyhostel.data.bin.human_validation.integrate:save",
            "cvat-label-constructor=flyhostel.data.bin.human_validation.label_constructor:main",
            "fh-make-video=flyhostel.data.bin.video:main",
            "fh-make-csv=flyhostel.data.bin.video:save_csv",
            "find-chunk-interval=flyhostel.data.bin.find_chunk_interval:main",
            "get-wavelet-profile=flyhostel.data.bin.utils:main_get_wavelet_profile",
            "get-framerate=flyhostel.data.bin.utils:main_get_framerate",
            "get-chunksize=flyhostel.data.bin.utils:main_get_chunksize",
            "get-number-of-animals=flyhostel.data.bin.utils:main_get_number_of_animals",

            ]
    },
)

warnings.warn(
    "Make sure that torch, torchvision, confapp, zeitgeber, "
    "trajectorytools, feed_integration, dropy are installed"
)
