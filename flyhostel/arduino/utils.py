import json
import sys
import logging

import serial
import serial.tools.list_ports

import flyhostel
from  flyhostel.arduino.constants import (
    PORT_PREFIX,
    NEWLINE,
    EMPTY,
    TIMEOUT
)

with open(flyhostel.CONFIG_FILE, "r") as fh:
    config = json.load(fh)

logging.basicConfig(level=getattr(logging, config["logging"]["arduino"]))


    
def safe_json_load(ser, data):
    try:
        data = json.loads(data)
        status = 0
    except json.decoder.JSONDecodeError as error:
        message = sys.exc_info()
        logging.warning(f"Parsing error on port {ser.port}")
        logging.debug(data)
        data = None
        status = 1

    return status, data
    
def list_ports():
    ports = serial.tools.list_ports.comports()
    ports = [p.device for p in ports]
    ports = [p for p in ports if PORT_PREFIX in p]
    return ports

def contains_data(data):
    """
    Checks data actually contains something

    Arguments:
        * data (str)
    Returns:
        * valid (bool)
    """

    valid = data != "" and len(data) != 0
    return valid

def contains_data(data):
    """
    Checks data actually contains something

    Arguments:
        * data (str)
    Returns:
        * valid (bool)
    """

    valid = data != "" and len(data) != 0
    return valid


def read(ser):
    """
    Keep reading from serial until a newline character
    or the empty character are received

    Arguments:
    
        * ser (serial.Serial)

    Returns:
        * data (str)    
    """

    logging.debug(f"Reading from {ser.port}")

    data = "" 
    # NOTE
    # This line may not be needed
    ser.reset_output_buffer()

    while True:
        read_str = ser.read(1).decode("utf-8")
        data += read_str
        if read_str == NEWLINE or read_str == EMPTY:
            break
    
    data = data.strip(NEWLINE).strip("\r").strip(NEWLINE).strip("\r")
    return data

def write(ser, command):
    """
    Send the passed command to the device
    Makes sure it ends in newline and it is encoded to bytes    
    """
    
    command_log = command.strip('\n')
    logging.debug(f"Writing {command_log} to {ser.port}")

    if command[-1] != NEWLINE:
        command += NEWLINE

    command = command.encode() 
    
    return ser.write(command)


def talk(ser, command, wait_for_response=True, max_attempts=2):
    """
    Send a command to a Serial device

    If wait_for_response, also read what the device says.
    Try reading is up to max_attempts

    Arguments:
        * command (str): A command that the device knows how to interpret
        * wait_for_response (bool): Whether to expect a response back or not
        * max_attempts (int): How many times to try to hear the response from the device


    Returns
        * data (str) NEWLINE ended string
    """

    attempts=0
    while attempts < max_attempts:
        logging.debug(f"Talking to {ser.port}")

        write(ser, command)
        if not wait_for_response:
            break

        data = read(ser)

        if contains_data(data):
            logging.debug(f"Received {data} from {ser.port}")
            return data
        else:
            attempts+=1


    return None


def identify_ports(ports):
    """
    Store a descriptive name under each detected port
    
    The name is fetched

    Arguments
    
    """
    identifiers = {}
    for port in ports:
        try:
            with serial.Serial(port, timeout=TIMEOUT) as ser:
                name = read_device_name(ser)
                identifiers[port] = name
        except serial.serialutil.SerialException as error:
            message = sys.exc_info()
            if "Permission denied" in str(message[1]):
                logging.debug(f"Permission denied error on port {port}")
            else:
                raise error

    return identifiers


def read_device_name(ser):
    data = talk(ser, command = "T\n", wait_for_response=True, max_attempts=3)
    return safe_json_load(ser, data)[1]["name"]
        