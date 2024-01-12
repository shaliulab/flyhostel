import argparse
from flyhostel.data.pose.umap import train_umap

def get_parser():


    ap=argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help=
                    """Path to a text file with experiment identifiers line by line
                    This identifiers will be used to build the training data of the UMAP    
                    """
    )
    return ap

def main():

    ap=get_parser()
    args=ap.parse_args()
    train_umap(args.input)
