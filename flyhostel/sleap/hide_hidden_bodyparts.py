import logging
import argparse
from tqdm.auto import tqdm
from sleap.io.dataset import Labels
logger=logging.getLogger(__name__)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels")
    ap.add_argument("--output")
    args=ap.parse_args()

    labels=Labels.load_file(args.labels)

    for instance in tqdm(labels.user_instances + labels.predicted_instances):
        instance.fill_missing(max_x=0, max_y=0)
        for i, point in enumerate(instance.points):
            if not point.visible:
                point.x=i*10
                point.y=30

    
    labels.save_file(labels=labels, filename=args.output)

if __name__ == "__main__":
    main()
