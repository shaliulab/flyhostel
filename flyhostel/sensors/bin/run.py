import logging

from flyhostel.utils import load_config
from flyhostel.sensors.sensor import Sensor
from flyhostel.sensors.server import Server

from .parser import get_parser

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
