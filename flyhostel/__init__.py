import os
import logging
import logging.config

import yaml
from confapp import conf

from flyhostel.constants import LOGGING_CONFIG

# Shipped with the package, used when the user has no config of their own.
DEFAULT_LOGGING_CONFIG = os.path.join(os.path.dirname(__file__), "default_logging.yaml")


def _drop_unavailable_handlers(config):
    """Remove handlers that cannot be constructed on this machine.

    SysLogHandler raises at dictConfig time if its address does not exist,
    which is the case in containers, on CI runners, on macOS and on Windows.
    """
    handlers = config.get("handlers", {})
    unavailable = [
        name
        for name, spec in handlers.items()
        if spec.get("class") == "logging.handlers.SysLogHandler"
        and isinstance(spec.get("address"), str)
        and not os.path.exists(spec["address"])
    ]

    for name in unavailable:
        del handlers[name]

    if unavailable:
        for logger in config.get("loggers", {}).values():
            logger["handlers"] = [
                h for h in logger.get("handlers", []) if h not in unavailable
            ]
        root = config.get("root")
        if root:
            root["handlers"] = [
                h for h in root.get("handlers", []) if h not in unavailable
            ]

    return config


def _configure_logging():
    path = LOGGING_CONFIG if os.path.isfile(LOGGING_CONFIG) else DEFAULT_LOGGING_CONFIG

    try:
        with open(path, "r") as filehandle:
            config = yaml.load(filehandle, yaml.SafeLoader)
        logging.config.dictConfig(_drop_unavailable_handlers(config))
    except Exception as error:
        # Logging setup must never stop the package from importing.
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).warning(
            "Could not apply logging config from %s (%s); using defaults.", path, error
        )


# Configure logging before importing submodules, so that loggers created at
# their import time are the ones this config applies to.
_configure_logging()

conf += "flyhostel.constants"
