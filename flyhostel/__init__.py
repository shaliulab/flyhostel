__version__ = '1.1.3'

import sys
import os
import logging.config

import yaml
from confapp import conf

from .constants import LOGGING_CONFIG

sys.path.insert(0, os.getcwd())

try:
    import local_settings
    conf += local_settings
except:
    pass

conf += "flyhostel.constants"

with open(LOGGING_CONFIG, "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

logging.config.dictConfig(config)
