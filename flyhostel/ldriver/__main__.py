import serial
import logging
import argparse
import time

from flyhostel.arduino import Identifier


def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()
    ap.add_argument("state", type=int, choices=[0, 1])
    ap.add_argument("--debug", action="store_true", default=False)
    ap.add_argument("--arduino-port", dest="arduino_port", default=None)
    return ap


def main(args=None):

    if args is None:
        args = get_parser().parse_args()

    level = {True: logging.DEBUG, False: logging.WARNING}[args.debug]
    logging.basicConfig(level=level)

    identifier = Identifier
    status = False

    if args.arduino_port is None:
        while not status:
            try:
                port = identifier.report()["LED driver"]
                logging.info(f"Detected LED driver on port {port}")
                status = True
            except serial.serialutil.SerialException as error:
                logging.warning(error)
                status = False
                time.sleep(1)
    else:
        port = args.arduino_port
        status = True

    with serial.Serial(port) as ser:
        logging.debug(f"Setting state to {args.state}")
        cmd = f"S L {args.state * 255}\n".encode()
        logging.debug(cmd)
        ser.write(cmd)


if __name__ == "__main__":
    main()
