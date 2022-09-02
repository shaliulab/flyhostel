import argparse

def get_parser():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", type=str, required=True)
    ap.add_argument("--source", type=str, default="selected")
    ap.add_argument("--dest", type=str, default="master")
    ap.add_argument("--interval", nargs="+", type=int)
    return ap
