import argparse
import json
import logging

from flyhostel.utils import load_config
from flyhostel.sensors.sensor import Sensor
from flyhostel.sensors.server import Server


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


def main(args=None):
    """
    Start a sensor instance:

    * Reads data from the serial monitor and outputs to the logfile
    * Sets up html and json servers for easy query of the data
    """

    config = load_config()

    logging.basicConfig(level=getattr(logging, config["logging"]["sensors"]))

    if args is None:
        args = get_parser().parse_args()

    sensor = Sensor(logfile=args.logfile, port=args.arduino_port)

    if args.thread:
        sensor.start()
        html_server = Server(sensor, server_type="html", port=args.html_port)
        json_server = Server(sensor, server_type="json", port=args.json_port)

        html_server.start()
        json_server.start()

    else:
        sensor.get_readings()
        print(sensor._data)

    return 0


if __name__ == "__main__":
    main()
