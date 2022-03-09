import threading
import time
import datetime
import logging
import os
import traceback
import json

import serial

from flyhostel.arduino import utils
from flyhostel.arduino import Identifier
import flyhostel

TIMEOUT = 5
MAX_COUNT=3
DATA_SEPARATOR=","
METADATA_SEPARATOR=";"

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

        self.reset()

        if port is None:
            port = self.detect()

        self._ser = serial.Serial(port, timeout=TIMEOUT)
        self._logfile = logfile
        self._verbose = verbose
        self._data = {}
        super().__init__(*args, **kwargs)


    @property
    def last_time(self):
        return self._data["time"]

    @property
    def must_update(self):
        return time.time() > self.last_time + self._freq

    @property
    def has_logfile(self):
        return self._logfile is not None


    def __getattr__(self, value):
        if value in self._data.keys():
            return self._data[value]
        else:
            return super().__getattr__(value)

    def reset(self):
        self._data = {
            "temperature": 0,
            "pressure": 0,
            "altitude": 0,
            "light": 0,
            "humidity": 0,
            "timestamp": 0,
            "datetime": "",
        }

    def detect(self):

        identifier = Identifier
        port = identifier.report().get("Environmental sensor", None)
        if port is None:
            raise Exception("Environmental sensor not detected")
        else:
            print("Detected Environmental sensor on port %s" % port)
        return port


    def communicate(self):

        data = utils.talk(self._ser, "D\n")
        status, data = utils.safe_json_load(self._ser, data)
        
        data["timestamp"] = time.time()
        data["datetime"] = datetime.datetime.fromtimestamp(
            data["timestamp"]
        ).strftime("%Y-%m-%d %H:%M:%S")

        if status == 0:
            self._data = data

        return status

    def get_readings(self):
        status = self.communicate()
        return status

    def loop(self):

        status = self.get_readings()

        if self.has_logfile and self.must_update:
            self.write()
        
        return status


    def run(self):
        count = 0
        try:
            while True:

                status = self.loop()
                if status == 0:
                    count = 0
                else:
                    count +=1
                    if count == MAX_COUNT:
                        os.system("reboot")

        except KeyboardInterrupt:
            pass
        
    def write(self):
        with open(self._logfile, "a") as fh:
            fh.write(
                "%s\t%s\t%s\t%s\t%s\t%s\n"
                % (
                    self._data["datetime"],
                    self._data["temperature"],
                    self._data["humidity"],
                    self._data["light"],
                    self._data["pressure"],
                    self._data["altitude"],
                )
            )


if __name__ == "__main__":
    sensor = Sensor(verbose=True)
    sensor.start()
