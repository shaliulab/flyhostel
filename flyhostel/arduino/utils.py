import json
import subprocess
import sys
import logging
import time

import serial
import serial.tools.list_ports

import flyhostel

with open(flyhostel.CONFIG_FILE, "r") as fh:
    config = json.load(fh)

logging.basicConfig(level=getattr(logging, config["logging"]["arduino"]))

TIMEOUT = 5


def list_ports():
    ports = [p.device for p in serial.tools.list_ports.comports()]
    ports = [p for p in ports if "ACM" in p]
    return ports


def read_from_serial(ser):
    data = ""
    ret = True
    start_time = time.time()
    while True:
        read = ser.read(100).decode("utf-8")
        logging.debug(read)
        if (
            (read == "")
            and (data != "")
            and ((time.time() - start_time) > ser.timeout)
        ):
            break
        else:
            data += read

        if len(read) < 100 and (time.time() - start_time) > ser.timeout:
            ret = False
            break

    return ret, data


def identify_ports(ports):
    identifiers = {}
    max_attempts = 3
    attempts = 0
    for port in ports:
        try:
            with serial.Serial(port, timeout=TIMEOUT) as ser:
                status = False
                while (not status) and (attempts < max_attempts):
                    attempts += 1
                    ser.write(b"T\n")
                    ret, raw_data = read_from_serial(ser)
                    status = raw_data != ""
                if attempts == max_attempts:
                    logging.error(f"Device on port {port} not responding")
                    continue
                try:
                    data = json.loads(raw_data)
                except json.decoder.JSONDecodeError as error:
                    message = sys.exc_info()
                    logging.warning(f"Parsing error on port {port}")
                    logging.debug(raw_data)
                    continue
                name = data["name"]
                identifiers[port] = name
        except serial.serialutil.SerialException as error:
            message = sys.exc_info()
            if "Permission denied" in str(message[1]):
                logging.debug(f"Permission denied error on port {port}")
            else:
                raise error

    return identifiers
