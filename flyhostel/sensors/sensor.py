import threading
import time
import datetime
import logging
import os
import glob
import json

import serial

from flyhostel.arduino.utils import read_from_serial
from flyhostel.arduino import Identifier
import flyhostel

TIMEOUT = 5
MAX_COUNT=3

with open(flyhostel.CONFIG_FILE, "r") as fh:
    conf = json.load(fh)

try:
    FREQUENCY = conf["sensors"]["frequency"]
except Exception:
    FREQUENCY = 60


class Sensor(threading.Thread):

    _freq = FREQUENCY  # seconds

    def __init__(
        self, logfile=None, verbose=False, port=None, *args, **kwargs
    ):

        if port is None:
            port = self.detect()
        self._ser = serial.Serial(port, timeout=TIMEOUT)
        self.flush()
        self._logfile = logfile
        self._verbose = verbose
        self.reset()
        super().__init__(*args, **kwargs)

    def reset(self):
        self._data = {
            "temperature": 0,
            "pressure": 0,
            "altitude": 0,
            "light": 0,
            "humidity": 0,
            "time": 0,
        }
        for k, v in self._data.items():
            setattr(self, k, v)

    def detect(self):

        identifier = Identifier
        port = identifier.report()["Environmental sensor"]
        print("Detected Environmental sensor on port %s" % port)
        return port

    def flush(self):
        self._ser.flushInput()
        for i in range(5):
            self._ser.readline()

    def parse_data(self, data):

        assert len(data) >= 3

        magnitude = data[1].lower()  # temperature, humidity, etc
        measurement = float(data[2])  # value in some unit
        
        if magnitude == "temperature" and measurement < -100:
            logging.warning("Temperature is aberrant and the thermometer is probably malfunctioning. Shutting down")
            os.system("reboot")

        unit = ""
        if len(data) == 4:
            unit = data[3].lower()  # unit if any

        if unit == "Pa":
            measurement /= 100
            unit = "hPa"

        self._data[magnitude] = str(round(measurement, ndigits=2))
        setattr(self, magnitude, measurement)

        if self._verbose:
            print(self._data)

    def read(self):

        try:
            ret, lines = read_from_serial(self._ser)
            lines = lines.split("\r\n")
            data = [line.split(":") for line in lines]
            # remove the empty line
            data = [line for line in data if len(line) > 1]

        except Exception as e:
            print("ERROR:Could not decode serial data")
            print(e)
            return (1, (None,))

        try:
            for line in data:
                self.parse_data(line)
            return (0, (ret,))
        except Exception as error:
            logging.warning(error)
            time.sleep(1000)
            return (1, (None,))

    def get_readings(self, n=5):

        self._data["time"] = time.time()
        self._ser.write(b"D\r\n")

        for i in range(n):
            code, info = self.read()
            if (
                not info[0] and not code
            ):  # this signals the function is done with success
                return 0

        return 1

    def run(self):

        self.last_t = 0
        count = 0
        while True:

            try:
                status = self.get_readings()
                if status == 0:
                    count = 0
                else:
                    count +=1
                    if count == MAX_COUNT:
                        os.system("reboot")
                    else:
                        continue

                if (
                    self._logfile is not None
                    and time.time() > self.last_t + self._freq
                ):
                    self.write()

            except KeyboardInterrupt:
                break


            time.sleep(1)

    def write(self):
        with open(self._logfile, "a") as fh:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fh.write(
                "%s\t%s\t%s\t%s\t%s\t%s\n"
                % (
                    now,
                    self._data["temperature"],
                    self._data["humidity"],
                    self._data["light"],
                    self._data["pressure"],
                    self._data["altitude"],
                )
            )

        self.last_t = time.time()


if __name__ == "__main__":
    sensor = Sensor(verbose=True)
    sensor.start()
