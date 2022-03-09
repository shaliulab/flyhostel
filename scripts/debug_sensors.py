import logging
import serial
import time
import flyhostel.arduino.utils

log = logging.getLogger("flyhostel.arduino.utils")
log.setLevel(logging.DEBUG)


#ports = flyhostel.arduino.utils.list_ports()
#flyhostel.arduino.utils.identify_ports(ports)


port = "/dev/ttyACM0"
TIMEOUT=1

with serial.Serial(port, timeout=TIMEOUT) as ser:

    status = False
    #ser.write(b"T\n")
    ser.write(b"T")
    data = b""
    while True:
        new_data = ser.read(100)
    #ret, data = flyhostel.arduino.utils.read_from_serial(ser)
        print(new_data)
        data += new_data
        print(data)
