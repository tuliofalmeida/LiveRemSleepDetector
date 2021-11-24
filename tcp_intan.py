import logging
import socket
import utilities as u
from PyQt5 import QtCore
import re
from io import BytesIO
from multiprocessing import Queue


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
    COMMAND_BUFFER_SIZE = 1024
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
        self.sr_re = re.compile('([0-9]+)')
        self.c_ix = 1
        self.former_color = '#FF0000'

    def add_acc_ch(self):
        for acc_ix in range(1, 4):
            cmd = f'set {self.headstage}-aux{acc_ix}.tcpdataoutputenabled true'
            self.send_cmd(cmd)

    def clear_all_data_outputs(self):
        self.send_cmd('set runmode stop')
        self.send_cmd('execute clearalldataoutputs')

    def connect(self):
        super().connect()
        self.ping()

    def ping(self):
        self.logger.debug(f'Pinging')
        cmd = 'get SampleRateHertz'
        ret = self.send_cmd(cmd)
        if ret == '':
            self.stop_pinging()
            self.attemp_connect = True
            self.connected = False
            self.disconnect_ev.emit()
        elif ret:
            sr = self.sr_re.search(ret).group(1)
            self.sampling_rate = sr
            self.ping_timer.start(self.ping_delay)

    def run(self):
        cmd = 'set runmode record'
        r = self.send_cmd(cmd)
        if r:
            self.send_cmd('set runmode run')

    def start_pinging(self):
        self.ping_timer.start(self.ping_delay)

    def stop_pinging(self):
        self.ping_timer.stop()

    def read_data(self):
        try:
            ret = self.socket.recv(self.COMMAND_BUFFER_SIZE)
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


class DataFifo:

    def __init__(self) -> None:
        super().__init__()
        self.buffer = BytesIO()
        self.avail = 0
        self.size = 0
        self.write_fp = 0

    def read(self, size=None):
        """
        Read data from circular buffer

        Parameters
        ----------
        size: Optional, int

        Returns
        -------
        result: bytes
        """
        if size is None or size > self.avail:
            size = self.avail
        size = max(size, 0)

        result = self.buffer.read(size)
        self.avail -= size

        # If we did not read enough from the end, complete it with the beginning (circular buffer)
        if len(result) < size:
            self.buffer.seek(0)
            result += self.buffer.read(size-len(result))

        return result

    def write(self, data):
        """
        Append data to buffer

        Parameters
        ----------
        data: bytes

        Returns
        -------

        """
        if self.size < self.avail + len(data):
            # We need to expand the buffer
            new_buffer = BytesIO()
            new_buffer.write(self.read())  # Write whatever we currently have
            self.avail = new_buffer.tell()
            self.write_fp = self.avail
            read_fp = 0
            while self.size <= self.avail + len(data):
                self.size = max(self.size, 1024) * 2
            new_buffer.write(b'0' * (self.size - self.write_fp))  # Initialize
            self.buffer = new_buffer
        else:
            read_fp = self.buffer.tell()
        self.buffer.seek(self.write_fp)
        written = self.size - self.write_fp
        self.buffer.write(data[:written])
        self.write_fp += len(data)
        self.avail += len(data)
        if written < len(data):
            self.write_fp -= self.size
            self.buffer.seek(0)
            self.buffer.write(data[written:])
        self.buffer.seek(read_fp)


class Streamer(TcpHandler):
    data_ready = QtCore.pyqtSignal(dict)
    data_error = QtCore.pyqtSignal()
    magic_size = 1540

    def __init__(self, queue: Queue, ip='127.0.0.1', port=5001, auto_retry=False,
                 logname='LRDlog', n_channels=4):
        super().__init__(ip, port, auto_retry, logname)
        self.queue = queue
        self.n_channels = n_channels
        self.read_delay = 1
        self.parse_delay = 3
        self.socket.settimeout(1)  # Apparently necesary on Windows
        self.stopping = False
        self.buffer = DataFifo()
        self.parser_timer = QtCore.QTimer()
        self.parser_timer.timeout.connect(self.parse)

    def parse(self):
        raw_data = self.buffer.read(self.magic_size*15)
        if len(raw_data) == 0:
            return
        # self.logger.info('Parsing')
        data = u.parse_block(raw_data, self.n_channels)
        if data is None:
            self.logger.error(f'Data error {len(raw_data)}')
            self.data_error.emit()
            if self.stopping:
                self.stopping = False
                self.read_timer.stop()
            return
        # self.buffer = self.buffer[magic_size:]
        # data = data.reshape((-1, self.n_channels+1))
        # self.data_ready.emit({'lfp': data[:, 0], 'acc': data[:, 1:]})
        # self.data_ready.emit({'data': data})
        self.queue.put(data)

    def start_stream(self):
        self.read_timer.start(self.read_delay)
        self.parser_timer.start(self.parse_delay)

    def stop_stream(self):
        self.stopping = True

    def read_data(self):
        # FIXME: Get rid of this magic number

        try:
            # raw_data = self.socket.recv(144*320*5)
            raw_data = self.socket.recv(self.magic_size*30)
            # raw_data = self.socket.recv(50 * self.magic_size * 3)
            self.buffer.write(raw_data)
            # self.logger.info(f'Buffer size: {self.buffer.size}, Data Size {len(raw_data)}')
        except socket.timeout:
            self.disconnect_ev.emit()
            self.read_timer.stop()
            return

