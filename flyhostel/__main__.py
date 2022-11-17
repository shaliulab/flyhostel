#! /usr/bin env python

import argparse

try:
    import flyhostel.sensors.bin
    SENSOR_AVAILABLE=True
except Exception as error:
    SENSOR_AVAILABLE=False

import flyhostel.sensors.io.bin

try:
    import flyhostel.ldriver.bin
    LED_DRIVER_AVAILABLE=True
except Exception as error:
    LED_DRIVER_AVAILABLE=False

import flyhostel.quantification.bin
import flyhostel.computer_vision.bin
import flyhostel.quantification.modelling.bin
import flyhostel.data.bin.df
import flyhostel.data.bin.copy
import flyhostel.data.bin.download

def get_parser():
    ap = argparse.ArgumentParser(
        description="FlyHostel manages and monitors a behavioral experiment with Drosophila melanogaster"
    )
    subparsers = ap.add_subparsers()  # help=argparse.SUPPRESS)

    if SENSOR_AVAILABLE:
        # Sensor module (to drive a custom made temp, humidity and light sensor)
        sensor_parser = subparsers.add_parser(
            "sensor",
            parents=[flyhostel.sensors.bin.parser.get_parser()],
            add_help=False,
            help="Command the environmental sensor in the setup",
        )
        sensor_parser.set_defaults(func=flyhostel.sensors.bin.run.main)

    
    # Sensor IO module (for reading and plotting the sensor data)
    sensor_io_parser = subparsers.add_parser(
        "sensor-io",
        parents=[flyhostel.sensors.io.bin.parser.get_parser()],
        add_help=False,
        help="Sensor IO",
    )
    sensor_io_parser.set_defaults(func=flyhostel.sensors.io.bin.run.main)

    if LED_DRIVER_AVAILABLE:
        # LED-driver module
        ldriver_parser = subparsers.add_parser(
            "ldriver",
            parents=[flyhostel.ldriver.bin.parser.get_parser()],
            add_help=False,
            help="Command the LED driver in the setup",
        )
        ldriver_parser.set_defaults(func=flyhostel.ldriver.bin.run.main)

    # Quantification module (to quantify behaviors in flyhostel datasets)
    quantification_parser = subparsers.add_parser(
        "quant",
        parents=[flyhostel.quantification.bin.parser.get_parser()],
        add_help=False,
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
    
    # Quantification module (to quantify behaviors in flyhostel datasets)
    cv_parser = subparsers.add_parser(
        "cv",
        parents=[flyhostel.computer_vision.bin.parser.get_parser()],
        add_help=False,
        help="""
        
        """
    )
    cv_parser.set_defaults(func=flyhostel.computer_vision.bin.run.main)
    
    modelling_parser = subparsers.add_parser(
        "modelling",
        parents=[flyhostel.quantification.modelling.bin.parser.get_parser()],
        add_help=False,
        help="""
        Modelling and simulation functinality.

        Simulate virtual flies that can be either awake or asleep and transition between both states
        spontaneously or by means of the interaction with another animal, each with some prob.
        This simulation teaches us about the effects of groups on sleep duration
        """,
    )
    modelling_parser.set_defaults(func=flyhostel.data.bin.copy.main)

    # Data module (to manage and administer flyhostel datasets)
    # NOTE: For now it just copies the trajectory files to the imgstore folder,
    # following the convention of chunk number with 6 zeros + .npy
    copy_parser = subparsers.add_parser(
        "copy",
        parents=[flyhostel.data.bin.copy.get_parser()],
        add_help=False,
        help="""
        Copy trajectory files in the session_ folders of an idtrackerai project
        into the parent imgstore folder, so they become part of the imgstore dataset
        A typical call looks as follows
        fh-copy --imgstore-folder `pwd` --analysis-folder `pwd`/idtrackerai/ --overwrite
        """,
    )
    copy_parser.set_defaults(func=flyhostel.data.bin.copy.main)


    df_parser = subparsers.add_parser(
        "df",
        parents=[flyhostel.data.bin.df.get_parser()],
        add_help=False,
    )
    df_parser.set_defaults(func=flyhostel.data.bin.df.main)


    download_parser = subparsers.add_parser(
            "download",
            parents=[flyhostel.data.bin.download.get_parser()],
            add_help=False,
            help="Transfer files from Dropbox to the local computer",
        )
    download_parser.set_defaults(func=flyhostel.data.bin.download.main)

    return ap


def main():
    ap = get_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
