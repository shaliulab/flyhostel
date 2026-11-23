import os
import pathlib
import subprocess
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
    "imgstore-shaliulab>=0.4.23",
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
    "tomli",
    # "sleap-io",
    #"cupy>=12.2.0",
    #"cudf>=23.10.02",
]


GITHUB = "https://github.com/shaliulab"

# Commits of the libraries/ submodules that are installed as dependencies. Used only when the submodule
# is not checked out (a GitHub archive, an sdist, a clone without --recursive): the dependency is then
# fetched as a GitHub archive of that commit (public repositories, no ssh key needed). In a git checkout
# the gitlink (`git ls-tree HEAD libraries/<name>`) takes precedence and a warning is printed when it
# differs from the pin below: bump the pin together with the submodule (flyhostel/tests/test_setup.py
# checks that they agree).
PINS = {
    "libraries/ethoscopy": "e04fe1b02ea9459e3d7329bda99f0eb02c3f8045",
    "libraries/idtrackerai": "759b518c3bd40ad3ec8b05779b0f84a5861f7b18",
}


def _gitlink(subdir):
    """Commit the submodule `subdir` points at in this checkout (None outside a git checkout)."""
    parent, name = os.path.split(subdir)
    try:
        out = subprocess.run(
            ["git", "-C", str(HERE / parent), "ls-tree", "HEAD", name],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=30,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return out[2] if len(out) >= 3 and out[1] == "commit" else None


def pinned(subdir):
    commit = _gitlink(subdir)
    if commit is None:
        return PINS[subdir]
    if commit != PINS[subdir]:
        warnings.warn(
            "{} points at {} but setup.py pins {}: using the submodule's commit "
            "(update PINS)".format(subdir, commit, PINS[subdir])
        )
    return commit


def vendored(dist_name, repo, extras=None):
    """Build a PEP 508 direct reference to the submodule libraries/<repo>.

    When the submodule is checked out (a recursive git clone, which is also what pip does for a git+
    URL) the local checkout is installed, so local changes are picked up. Otherwise (a GitHub archive or
    an sdist: no .git, no submodules) the pinned commit is fetched from GitHub as an archive.

    The path must be absolute: pip resolves relative paths against the working directory, not against
    this file. `dist_name` must be the name declared in that package's own setup.py, or pip rejects the
    install as a name mismatch.
    """
    subdir = "libraries/" + repo
    path = HERE / subdir
    name = "{}[{}]".format(dist_name, ",".join(extras)) if extras else dist_name
    if (path / "setup.py").exists() or (path / "pyproject.toml").exists():
        return "{} @ {}".format(name, path.resolve().as_uri())
    return "{} @ {}/{}/archive/{}.tar.gz".format(name, GITHUB, repo, pinned(subdir))


install_requires.append(vendored("ethoscopy", "ethoscopy"))


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
        # The distribution is named idtrackerai-shaliulab, not idtrackerai. [gpu] pulls in torch and torchvision.
        "idtrackerai": [vendored("idtrackerai-shaliulab", "idtrackerai", extras=["gpu"])],
        # needs opencv>=4: install in its own environment, see flyhostel/roi_proposer/README.md
        "roi_proposer": ["ultralytics==8.3.40", "numpy<2", "tomli"],
    },
    include_package_data=True,
    package_data={"flyhostel": ["default_logging.yaml"]},
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "fh=flyhostel.__main__:main",
            "fh-server=flyhostel.server.server:main",
            "fh-environment=flyhostel.data.environment:main",
            "fh-validate=flyhostel.data.bin.dashboard:main",
            "missing-chunk-detector=flyhostel.utils.missing_chunk_detector:main",
            "compile-pose=flyhostel.data.bin.pose:main",
            "train-umap=flyhostel.data.bin.umap:main",
            "project-pose=flyhostel.data.bin.behavior:main",
            "predict-behavior=flyhostel.data.bin.ethogram:main",
            "compute-sleep=flyhostel.data.sleep:main",
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
            "roi-proposer=flyhostel.roi_proposer.cli:main",

            ]
    },
)

warnings.warn(
    "Make sure that torch, torchvision, confapp, zeitgeber, "
    "trajectorytools, feed_integration, dropy are installed"
)
