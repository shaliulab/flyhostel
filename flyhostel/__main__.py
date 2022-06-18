#! /usr/bin env python

import argparse
import flyhostel.sensors
import flyhostel.sensors.io
import flyhostel.ldriver
import flyhostel.quantification.main
import flyhostel.quantification.bin.parser
import flyhostel.data

import logging
logging.getLogger("flyhostel.data").setLevel(logging.DEBUG)
logging.getLogger("flyhostel.quantification").setLevel(logging.DEBUG)


def get_parser():
    ap = argparse.ArgumentParser(
        description="FlyHostel manages and monitors a behavioral experiment with Drosophila melanogaster"
    )
    subparsers = ap.add_subparsers()  # help=argparse.SUPPRESS)

    # Sensor module (to drive a custom made temp, humidity and light sensor)
    sensor_parser = subparsers.add_parser(
        "sensor",
        parents=[flyhostel.sensors.bin.parser.get_parser()],
        add_help=True,
        help="Command the environmental sensor in the setup",
    )
    sensor_parser.set_defaults(func=flyhostel.sensors.bin.run.main)
    
    # LED-driver module
    ldriver_parser = subparsers.add_parser(
        "ldriver",
        parents=[flyhostel.ldriver.bin.get_parser()],
        add_help=True,
        help="Command the LED driver in the setup",
    )
    ldriver_parser.set_defaults(func=flyhostel.ldriver.bin.run.main)
    
    # Sensor IO module (for reading and plotting the sensor data)
    sensor_io_parser = subparsers.add_parser(
        "sensor-io",
        parents=[flyhostel.sensors.io.bin.parser.get_parser()],
        add_help=True,
        help="Sensor IO",
    )
    sensor_io_parser.set_defaults(func=flyhostel.sensors.io.bin.run.main)

    # Quantification module (to quantify behaviors in flyhostel datasets)
    quantification_parser = subparsers.add_parser(
        "quant",
        parents=[flyhostel.quantification.bin.parser.get_parser()],
        add_help=True,
        help="""
        Quantification of behaviors recorded in a flyhostel dataset
        The following behaviors can be quantified:

        * sleep

        fh-quant --imgstore-folder `pwd` --output `pwd`/idtrackerai/  --interval X Y
        where X and Y are the first and last chunks considered in the quantification
        (actually the last one is Y-1)
        """
    )
    quantification_parser.set_defaults(func=flyhostel.quantification.bin.run.main)

    # Data module (to manage and administer flyhostel datasets)
    # NOTE: For now it just copies the trajectory files to the imgstore folder,
    # following the convention of chunk number with 6 zeros + .npy
    copy_parser = subparsers.add_parser(
        "copy",
        parents=[flyhostel.data.bin.copy.get_parser()],
        add_help=True,
        help="""
        Copy trajectory files in the session_ folders of an idtrackerai project
        into the parent imgstore folder, so they become part of the imgstore dataset
        A typical call looks as follows
        fh-copy --imgstore-folder `pwd` --analysis-folder `pwd`/idtrackerai/ --overwrite
        """,
    )
    copy_parser.set_defaults(func=flyhostel.data.bin.copy.main)

    modelling_parser = subparsers.add_parser(
        "modelling",
        parents=[flyhostel.quantification.modelling.bin.parser.get_parser()],
        add_help=True,
        help="""
        Modelling and simulation functinality.

        Simulate virtual flies that can be either awake or asleep and transition between both states
        spontaneously or by means of the interaction with another animal, each with some prob.
        This simulation teaches us about the effects of groups on sleep duration
        """,
    )
    modelling_parser.set_defaults(func=flyhostel.data.bin.copy.main)


    return ap


def main():
    ap = get_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
