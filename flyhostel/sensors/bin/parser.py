import argparse

def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument(
        "--arduino-port", dest="arduino_port", required=False, default=None, type=str,
    )
    ap.add_argument(
        "--html-port", dest="html_port", required=False, default=8000, type=int,
    )
    ap.add_argument(
        "--json-port", dest="json_port", required=False, default=9000, type=int,
    )
    ap.add_argument("--logfile", required=False, default="/temp_log.txt")
    ap.add_argument("--thread", default=False, action="store_true")
    return ap
