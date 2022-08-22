import os.path
import json
from flyhostel.constants import CONFIG_FILE

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as filehandle:
            config = json.load(filehandle)
    else:
        config = {}
    return config

