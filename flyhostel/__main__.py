#! /usr/bin env python

import argparse
import flyhostel.sensors
import flyhostel.sensors.io
import flyhostel.ldriver


def get_parser():
    ap = argparse.ArgumentParser(
        description="FlyHostel manages and monitors a behavioral experiment with Drosophila melanogaster"
    )
    subparsers = ap.add_subparsers()  # help=argparse.SUPPRESS)

    sensor_parser = subparsers.add_parser(
        "sensor",
        parents=[flyhostel.sensors.get_parser()],
        add_help=False,
        help="Command the environmental sensor in the setup",
    )
    sensor_parser.set_defaults(func=flyhostel.sensors.main)
    ldriver_parser = subparsers.add_parser(
        "ldriver",
        parents=[flyhostel.ldriver.get_parser()],
        add_help=False,
        help="Command the LED driver in the setup",
    )
    ldriver_parser.set_defaults(func=flyhostel.ldriver.main)
    sensor_io_parser = subparsers.add_parser(
        "sensor-io",
        parents=[flyhostel.sensors.io.get_parser()],
        add_help=False,
        help="Sensor IO",
    )
    sensor_io_parser.set_defaults(func=flyhostel.sensors.io.main)

    quantification_parser = subparsers.add_parser(
        "quant",
        parents=[flyhostel.quantification.get_parser()],
        add_help=False,
        help="Quantification",
    )
    quantification_parser.set_defaults(func=flyhostel.quantification.main)   

    return ap


def main():
    ap = get_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
