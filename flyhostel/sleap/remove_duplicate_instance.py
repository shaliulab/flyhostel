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

    for lf in tqdm(labels.labeled_frames):
        if len(lf.instances)>1:
            lf.instances=lf.instances[:1]
    
    labels.save_file(labels=labels, filename=args.output)

if __name__ == "__main__":
    main()
