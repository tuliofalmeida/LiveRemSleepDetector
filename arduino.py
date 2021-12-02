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
    for p in com_ports:
        print(p.product)
    ports = [p for p in com_ports if p.pid is not None]

    return ports


class Trigger(QtCore.QObject):

    def __init__(self, port: serial.Serial) -> None:
        super().__init__()
        self.port = port

    def trig(self,status):
        if status == True:
            self.port.write(b'1')
        else:
            self.port.write(b'0')

