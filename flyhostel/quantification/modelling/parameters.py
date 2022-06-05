import os.path
import yaml
from .constants import PARAMETERS_FILE

# 50 mm = 460 pixels
# 2 mm = 460 * 2 / 50 = 18

DEFAULT_PARAMETERS = {
    "neighbor_threshold": 18,
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
