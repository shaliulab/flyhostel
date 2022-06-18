import argparse

def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()
    ap.add_argument("state", type=int, choices=[0, 1])
    ap.add_argument("--debug", action="store_true", default=False)
    ap.add_argument("--arduino-port", dest="arduino_port", default=None)
    return ap
