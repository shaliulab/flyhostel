from confapp import conf
import yaml
import logging
import logging.config
from flyhostel.constants import LOGGING_CONFIG

conf += "flyhostel.constants"

with open(LOGGING_CONFIG, "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

logging.config.dictConfig(config)