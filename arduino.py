import typing

from PyQt5 import QtCore
import serial
from serial.tools import list_ports
import logging


def get_ports():
    """
    List all possible COM ports
    """
    com_ports = list_ports.comports()
    ports = [p for p in com_ports if p.product is not None and 'arduino' in p.product.lower()]
    return ports


class Trigger(QtCore.QObject):

    def __init__(self, port: serial.Serial) -> None:
        super().__init__()
        self.port = port

    def trig(self):
        self.port.write(b'1')
