import logging
import serial
import time
from flyhostel.arduino import utils

log = logging.getLogger("flyhostel.arduino.utils")
log.setLevel(logging.DEBUG)

port = "/dev/ttyACM0"
TIMEOUT=2

def main_v1():

    with serial.Serial(port, timeout=TIMEOUT) as ser:

        utils.write(ser, "T\n")
        data = utils.read(ser)
        utils.write(ser, "T\n")
        data = utils.read(ser)
        print(data)

def main_v2():

    with serial.Serial(port, timeout=TIMEOUT) as ser:

        data = utils.talk(ser, "T\n", wait_for_response=True, max_attempts=3)
        print(data)
    
def read_data():

    with serial.Serial(port, timeout=TIMEOUT) as ser:

        data = utils.talk(ser, "D\n", wait_for_response=True, max_attempts=3)
        print(data)


def main():
    return read_data()


if __name__ == "__main__":
    main()