import os.path
import yaml
from .constants import PARAMETERS_FILE

DEFAULT_PARAMETERS = {
    "neighbor_threshold": 50,
    "movement_bout_length": 100,
    "probability_interaction_movement": 0.5,
    "probability_spontaneous_movement": 0.1,
}

def load_parameters(parameters_file=PARAMETERS_FILE):

    parameters = DEFAULT_PARAMETERS.copy()

    if os.path.exists(parameters_file):
        with open(parameters_file, "r") as filehandle:
            parameters.update(
                yaml.load(filehandle, yaml.SafeLoader)
            )

    return parameters
