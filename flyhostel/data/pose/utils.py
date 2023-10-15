import os
from .constants import root_path


def load_animals(experiment):
    data_directories=os.listdir(root_path)
    data_directories=[directory for directory in data_directories if directory.startswith(experiment)]
    data_directories=sorted(data_directories)
    return data_directories