import argparse
from flyhostel.data.human_validation.cvat.label_constructor import label_constructor
from flyhostel.data.human_validation.cvat.cvat_integration import UNASSIGNED_LABEL
from flyhostel.data.human_validation.cvat.courtship import COURTSHIP_LABEL

def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--number-of-animals", type=int, required=True)
    ap.add_argument("--output", type=str, default="./labels.json", required=False)
    return ap

def main():
    ap=get_parser()
    args=ap.parse_args()
    label_constructor(number_of_animals=args.number_of_animals, tags=[
        ("FMB",  "#000000"),
        ("NEW-FRAGMENT", "#ffffff"),
        (UNASSIGNED_LABEL, "#ffffff"),
        (COURTSHIP_LABEL, "#ff0000"),        
    ], output_json=args.output)