"""
roi-proposer: propose DATE_TIME.toml and landmarks.toml for new experiments

    roi-proposer propose /flyhostel_data/videos/FlyHostel1/3X/2025-06-21_15-00-00
    roi-proposer build-dataset && roi-proposer train && roi-proposer evaluate
"""
import argparse

from . import dataset, evaluate, propose, train

COMMANDS = {
    "propose": (propose, "Write DATE_TIME.toml and landmarks.toml for experiment(s)"),
    "build-dataset": (dataset, "Build the training dataset from the experiments that already have both tomls"),
    "train": (train, "Train the model"),
    "evaluate": (evaluate, "Compare proposals with the hand-made ROIs of the validation experiments"),
}


def get_parser():
    ap = argparse.ArgumentParser(prog="roi-proposer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = ap.add_subparsers(dest="command", required=True)
    for name, (module, help) in COMMANDS.items():
        sub = subparsers.add_parser(name, help=help, description=module.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        module.get_parser(sub)
        sub.set_defaults(func=module.main)
    return ap


def main():
    args = get_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
