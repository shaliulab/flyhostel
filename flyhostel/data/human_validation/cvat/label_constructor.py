import webcolors
import json
from idtrackerai.utils.py_utils import get_spaced_colors_util

def label_constructor(number_of_animals, output_json="./labels.json"):
    
    colors=get_spaced_colors_util(number_of_animals, black=False)
    colors=[webcolors.rgb_to_hex([int(e) for e in color[:3][::-1]]) for color in colors]
    BLACK="#000000"

    data=[]
    for identity in range(1, number_of_animals+1):
        data.append({
            "name": str(identity),
            "attributes": [],
            "color": colors[identity-1]
        })
    
    data.append({
        "name": "FMB",
        "attributes": [],
        "color": BLACK
    })

    with open(output_json, "w") as handle:
        json.dump(data, handle)

