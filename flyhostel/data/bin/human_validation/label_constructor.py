import argparse
from flyhostel.data.human_validation.cvat.label_constructor import label_constructor

def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--number-of-animals", type=int, required=True)
    ap.add_argument("--output", type=str, default="./labels.json", required=False)
    return ap

def main():
    ap=get_parser()
    args=ap.parse_args()
    label_constructor(number_of_animals=args.number_of_animals, output_json=args.output)