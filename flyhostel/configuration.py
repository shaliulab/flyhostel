import json
from flyhostel.constants import CONFIG_FILE

def load_config():
    with open(CONFIG_FILE, "r") as filehandle:
        config = json.load(filehandle)
    return config

