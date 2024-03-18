import webcolors
import json
from idtrackerai.utils.py_utils import get_spaced_colors_util

def label_constructor(number_of_animals=None, tags=[], output_json="./labels.json"):
    
    colors=get_spaced_colors_util(number_of_animals, black=False)
    colors=[webcolors.rgb_to_hex([int(e) for e in color[:3][::-1]]) for color in colors]

    data=[]
    if number_of_animals is not None:
        for identity in range(1, number_of_animals+1):
            data.append({
                "name": str(identity),
                "attributes": [],
                "color": colors[identity-1]
            })
    
    for tag, color in tags:
        data.append({
            "name": tag,
            "attributes": [],
            "color": color
        })

    with open(output_json, "w") as handle:
        json.dump(data, handle)

