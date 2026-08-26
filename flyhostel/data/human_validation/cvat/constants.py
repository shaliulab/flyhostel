import os

# Read at import time but never required: the package must import cleanly on
# machines that have nothing to do with CVAT. Anything that actually talks to
# CVAT calls _require_cvat() first.
cvat_host = os.environ.get("CVAT_HOST")
cvat_username = os.environ.get("CVAT_USERNAME")
cvat_password = os.environ.get("CVAT_PASSWORD")

CVAT_BASE = "http://{}:8080".format(cvat_host) if cvat_host else None

_REQUIRED = ("CVAT_HOST", "CVAT_USERNAME", "CVAT_PASSWORD")


def cvat_is_configured():
    """True if every CVAT environment variable is set."""
    return all(os.environ.get(name) for name in _REQUIRED)


def _require_cvat():
    """Return (base_url, username, password), or raise if anything is missing.

    Call this at the top of any function that contacts CVAT, so the failure
    names the missing variable instead of surfacing as a KeyError on import.
    """
    missing = [name for name in _REQUIRED if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "CVAT is not configured: {} not set. Export {} to use the CVAT "
            "integration.".format(
                ", ".join(missing),
                "them" if len(missing) > 1 else "it",
            )
        )

    # Read from the environment rather than the module-level values, so a
    # variable set after import is still picked up.
    host = os.environ["CVAT_HOST"]
    return (
        "http://{}:8080".format(host),
        os.environ["CVAT_USERNAME"],
        os.environ["CVAT_PASSWORD"],
    )