conf += "flyhostel.constants"

with open(LOGGING_CONFIG, "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

logging.config.dictConfig(config)

