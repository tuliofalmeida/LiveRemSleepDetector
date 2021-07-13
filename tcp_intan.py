import logging
import socket
import utilities as u
from PyQt5 import QtCore


class TcpHandler(QtCore.QObject):
    connect_ev = QtCore.pyqtSignal(bool)
    disconnect_ev = QtCore.pyqtSignal()

    def __init__(self, ip='127.0.0.1', port=5000, auto_retry=False, logname='LRDlog'):
        super().__init__()
        self.ip = ip
        self.port = port
        self.reconnect_delay = 1500
        self.logger = logging.getLogger(logname)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(.05)
        # Connection
        self.attemp_connect = True
        self.connected = False
        self.auto_retry = auto_retry
        self.connect_timer = QtCore.QTimer()
        self.connect_timer.timeout.connect(self.connect)
        # Reading returned values
        self.read_timer = QtCore.QTimer()
        self.read_timer.timeout.connect(self.read_data)
        self.read_delay = 25

    def connect(self):
        if not self.attemp_connect:
            return
        try:
            self.logger.info(f'Connection attempt {self.ip}:{self.port}')
            self.socket.connect((self.ip, self.port))
            self.connected = True
            self.attemp_connect = False
            self.read_timer.start(self.read_delay)
            self.logger.info(f'Connected to {self.ip}:{self.port}')
            self.connect_ev.emit(True)
        except (ConnectionRefusedError, TimeoutError, ConnectionAbortedError, BrokenPipeError):
            if self.auto_retry:
                self.attemp_connect = True
                self.logger.info(f'Retrying in {self.reconnect_delay} ms')
                self.connect_timer.start(self.reconnect_delay)

    def read_data(self):
        # TBD in each tcp socket
        pass


class IntanMaster(TcpHandler):
    COMMAND_BUFFER_sIZE = 1024
    channel_set = QtCore.pyqtSignal(int)

    def __init__(self, ip='127.0.0.1', port=5000, auto_retry=False, logname='LRDlog',
                 headstage='a'):
        super().__init__(ip, port, auto_retry, logname)
        self.headstage = headstage
        self.auto_acc_ch = True
        # Pinging for disconnections
        self.ping_timer = QtCore.QTimer()
        self.ping_delay = 10000
        self.ping_timer.timeout.connect(self.ping)
        self.sampling_rate = None
        self.c_ix = 1
        self.former_color = '#FF0000'

    def add_acc_ch(self):
        for acc_ix in range(1, 4):
            cmd = f'set {self.headstage}-aux{acc_ix}.tcpdataoutputenabled true'
            self.send_cmd(cmd)

    def clear_all_data_outputs(self):
        self.send_cmd('execute clearalldataoutputs')

    def ping(self):
        self.logger.debug(f'Pinging')
        cmd = 'get SampleRateHertz'
        ret = self.send_cmd(cmd)
        if ret == '':
            self.stop_pinging()
            self.attemp_connect = True
            self.connected = False
            self.disconnect_ev.emit()
        else:
            self.sampling_rate = ret
            self.ping_timer.start(self.ping_delay)

    def start_pinging(self):
        self.ping_timer.start(self.ping_delay)

    def stop_pinging(self):
        self.ping_timer.stop()

    def read_data(self):
        try:
            ret = self.socket.recv(self.COMMAND_BUFFER_sIZE)
            s_ret = ret.decode('utf-8')
            return s_ret
        except (socket.timeout, BrokenPipeError, OSError):
            return ''

    def send_cmd(self, cmd: str):
        if not self.connected:
            return False
        cmd = cmd.encode('utf-8')
        self.socket.sendall(cmd)
        s_ret = self.read_data()
        return s_ret

    def set_ch_ix(self, ch_ix: int):
        self.logger.debug(f'Setting channel exported to {ch_ix}')
        self.clear_all_data_outputs()
        self.restore_color()
        cmd = f'get {self.headstage}-{ch_ix:03d}.color'
        ret = self.send_cmd(cmd)
        if len(ret) > 7:
            self.former_color = ret[-7:]
        self.c_ix = ch_ix
        cmd = f'set {self.headstage}-{ch_ix:03d}.tcpdataoutputenabled true'
        self.send_cmd(cmd)
        if self.auto_acc_ch:
            self.add_acc_ch()
        cmd = f'set {self.headstage}-{ch_ix:03d}.color #FFFFFF'
        self.send_cmd(cmd)
        self.channel_set.emit(ch_ix)

    def restore_color(self):
        cmd = f'set {self.headstage}-{self.c_ix:03d}.color {self.former_color}'
        self.send_cmd(cmd)


class Streamer(TcpHandler):
    data_ready = QtCore.pyqtSignal(dict)
    data_error = QtCore.pyqtSignal()

    def __init__(self, ip='127.0.0.1', port=5001, auto_retry=False, logname='LRDlog', n_channels=4):
        super().__init__(ip, port, auto_retry, logname)
        self.n_channels = n_channels
        self.read_delay = 2
        self.socket.settimeout(1)

    def start_stream(self):
        print('Starting stream')
        self.read_timer.start(self.read_delay)

    def stop_stream(self):
        print('Stopping stream')
        self.read_timer.stop()

    def read_data(self):
        try:
            self.logger.debug('Reading data')
            raw_data = self.socket.recv(200000)
        except socket.timeout:
            self.disconnect_ev.emit()
            self.read_timer.stop()
            return

        data = u.parse_block(raw_data, self.n_channels)
        if data is None:
            self.logger.error('Data error')
            self.data_error.emit()
            return
        data = data.reshape((-1, self.n_channels))
        self.data_ready.emit({'lfp': data[:, 0], 'acc': data[:, 1:]})
