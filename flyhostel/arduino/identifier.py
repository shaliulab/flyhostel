import serial.tools.list_ports
from .utils import identify_ports, list_ports

class Identifier:
    @classmethod
    def identify(cls):
        return identify_ports(list_ports())

    @classmethod
    def report(cls):
        ids = cls.identify()
        return {v: k for k, v in ids.items()}


if __name__ == "__main__":

    ports = list_ports()
    ids = identify_ports(ports)
    print(ids)
