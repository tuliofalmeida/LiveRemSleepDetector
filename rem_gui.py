import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from gui import Ui_MainWindow
import socket
import utilities as u


class Streamer(QtCore.QObject):
    data_ready = QtCore.pyqtSignal(dict)
    disconnected = QtCore.pyqtSignal()

    def __init__(self, intan_socket=None) -> None:
        super().__init__()
        self.socket = intan_socket
        self.intan_timer = QtCore.QTimer()
        self.intan_timer.timeout.connect(self.get_data)
        self.refresh_time = 2

    def start_stream(self):
        print('Starting stream')
        self.intan_timer.start(self.refresh_time)

    def get_data(self):
        print('Getting data')
        try:
            raw_data = self.socket.recv(200000)
            print(f"Data received: {len(raw_data)}")
        except socket.timeout:
            self.disconnected.emit()
            self.intan_timer.stop()
            return

        data = u.parse_block(raw_data, 4)
        print('Data parsed', data.shape)
        if data is None:
            self.disconnected.emit()        # FIXME: May not be the smartest
            self.intan_timer.stop()
            return
        data = data.reshape((-1, 4))
        self.data_ready.emit({'data': data[:, 0], 'acc': data[:, 1:]})


class REM(Ui_MainWindow):
    connected = QtCore.pyqtSignal(bool)
    lfp_ready = QtCore.pyqtSignal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.attempt_connect = True
        self.connect_delay = 1500
        self.connect_timer = QtCore.QTimer()
        self.connect_timer.timeout.connect(self.intan_connect)
        self.connect_timer.start(self.connect_delay)
        self.connected.connect(self.start_acq)
        self.data_th = QtCore.QThread()
        self.data_th.start()
        self.data_getter = Streamer(self.socket)
        self.data_getter.moveToThread(self.data_th)
        self.data_getter.data_ready.connect(self.get_data)
        self.data_getter.disconnected.connect(self.disconnected)
        self.init_socket()

    def init_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(1)
        self.data_getter.socket = self.socket

    def disconnected(self):
        self.init_socket()
        self.attempt_connect = True
        self.connect_timer.start(self.connect_delay)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        """
        closeEvent of the GUI.

        Clean up then close the GUI
        """
        self.data_th.quit()
        self.data_th.wait()
        self.data_widget.comp_th.quit()
        self.data_widget.comp_th.wait()
        a0.accept()

    def get_data(self, data):
        # FIXME: Fake accelerometer because complicated
        self.lfp_ready.emit({'lfp': data['data'], 'acc': data['acc']})

    def start_acq(self, is_connected):
        if is_connected:
            self.attempt_connect = False
            self.socket.settimeout(10)
            self.lfp_ready.connect(self.data_widget.comp_done)
            # self.lfp_ready.connect(self.data_widget.rem_comp.analyze_dict)
            self.data_getter.start_stream()

    def intan_connect(self):
        if not self.attempt_connect:
            return
        try:
            print('Connection attempt')
            self.socket.connect(('127.0.0.1', 5001))
            self.connected.emit(True)
            print('Connected to Intan')
        except (ConnectionRefusedError, TimeoutError, ConnectionAbortedError):
            self.connect_timer.start(self.connect_delay)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = REM()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
