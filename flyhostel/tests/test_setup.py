"""setup.py's PINS (the libraries/ submodules fetched as GitHub archives when they are not checked out,
e.g. an install from an archive or sdist) must match the commits the submodules point at."""
import ast
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _pins():
    tree = ast.parse((ROOT / "setup.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PINS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("PINS not found in setup.py")


def test_pins_match_gitlinks():
    try:
        subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    for subdir, commit in _pins().items():
        out = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "HEAD", subdir],
                             capture_output=True, text=True, check=True).stdout.split()
        assert out[1:3] == ["commit", commit], "{}: setup.py pins {}, the submodule points at {}".format(
            subdir, commit, out[2] if len(out) > 2 else None)
