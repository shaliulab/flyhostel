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

def send_command(ser, command="T"):
    cmd = f"{command}\n"
    ser.write(cmd.encode("utf-8"))

with serial.Serial(port, timeout=TIMEOUT) as ser:

    status = False
    i = 0
    send_command(ser)
    while True:
        ret, data = flyhostel.arduino.utils.read_from_serial(ser)
        send_command(ser, command = "D")
        #i += 1
        #if i == 1:
        #    send_command(ser, command = "T")
        #    i = 0

