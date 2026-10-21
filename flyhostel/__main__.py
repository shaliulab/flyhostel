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

import flyhostel.quantification.bin.angle
import flyhostel.computer_vision.bin
import flyhostel.modelling.bin.parser
import flyhostel.modelling.bin.run
import flyhostel.data.bin.download
import flyhostel.data.bin.df
import flyhostel.data.bin.copy
import flyhostel.data.bin.export
import flyhostel.data.bin.video

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
        parents=[flyhostel.modelling.bin.parser.get_parser()],
        add_help=False,
        help="""
        Simulate virtual flies that can be either awake or asleep and transition between both states
        spontaneously or by means of the interaction with another animal (flyhostel.modelling).
        Their sleep is quantified with the velocity based sleep of flyhostel.data.sleep
        """,
    )
    modelling_parser.set_defaults(func=lambda args: flyhostel.modelling.bin.run.main(args=args))
    

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


    export_parser = subparsers.add_parser(
            "export",
            parents=[flyhostel.data.bin.export.get_parser()],
            add_help=False,
            help="Export data to SQLite database",
        )
    export_parser.set_defaults(func=flyhostel.data.bin.export.main)


    write_angle = subparsers.add_parser(
            "write_angle",
            parents=[flyhostel.quantification.bin.angle.get_parser()],
            add_help=False,
            help="Write detection angle",
        )
    write_angle.set_defaults(func=flyhostel.quantification.bin.angle.main)

    make_single_animal_video = subparsers.add_parser(
            "make_video",
            parents=[flyhostel.data.bin.video.get_parser()],
            add_help=False,
            help="Make single video from FlyHostel.db",
        )
    make_single_animal_video.set_defaults(func=flyhostel.data.bin.video.main)

    return ap


def main():
    ap = get_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
