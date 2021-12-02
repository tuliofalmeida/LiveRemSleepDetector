from os import times
import numpy as np
from rem_obj import REMDetector
from tcp_intan import IntanMaster, Streamer
import pyqtgraph as pg
import sys
from functools import partial
from PyQt5 import QtWidgets, QtCore, QtGui
import logging
from logging.handlers import RotatingFileHandler
from arduino import get_ports, Trigger
import serial
from multiprocessing import Queue
from queue import Empty
import time


class UI(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Live REM sleep detector - Girardeau lab')
        # Logging
        self.logname = 'LRDlog'
        self.logger = logging.getLogger(self.logname)
        self.logger.setLevel(logging.DEBUG)
        self._log_handler = RotatingFileHandler(f'{self.logname[:-3]}.log', maxBytes=int(1e6),
                                                backupCount=1)
        self._formatter = logging.Formatter(
            '%(asctime)s :: %(filename)s :: %(funcName)s :: line %(lineno)d :: %(levelname)s :: %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S')
        self._log_handler.setFormatter(self._formatter)
        self.logger.addHandler(self._log_handler)
        sys.excepthook = partial(handle_exception, self.logname)

        # Layouts
        self._main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self._main_widget)
        self._main_lyt = QtWidgets.QHBoxLayout(self._main_widget)
        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        # self._main_widget.setMinimumSize(800, 300)
        self._main_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Expanding)
        self._splitter.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                     QtWidgets.QSizePolicy.Expanding)

        # Left columns
        self.lfp_plot = pg.PlotWidget(name='LFP signal')
        self.ratio_plot = pg.PlotWidget(name='Delta / Theta ratio')
        self.acc_plot = pg.PlotWidget(name='Motion plot')
        left_wdg = QtWidgets.QWidget()
        v_lyt = QtWidgets.QVBoxLayout(left_wdg)
        v_lyt.addWidget(self.lfp_plot)
        v_lyt.addWidget(self.ratio_plot)
        v_lyt.addWidget(self.acc_plot)

        # Layout of the right column
        scroller = QtWidgets.QScrollArea(self)
        scroller.setWidgetResizable(True)
        right_w = QtWidgets.QWidget()
        scroller.setWidget(right_w)
        v_lyt_right = QtWidgets.QVBoxLayout(right_w)
        # Logo
        logo_lab = QtWidgets.QLabel(self)
        logo_lab.setPixmap(QtGui.QPixmap(
            'RatHippoBLA.png').scaledToHeight(200))
        v_lyt_right.addWidget(logo_lab)
        # Parameters
        options_lyt = QtWidgets.QFormLayout()
        self.headstage = QtWidgets.QComboBox()
        self.headstage.addItems(('A', 'B', 'C', 'D'))
        self.headstage.currentTextChanged.connect(self.update_headstage)
        options_lyt.addRow('Headstage port', self.headstage)
        self.ch_num = QtWidgets.QSpinBox()
        self.ch_num.setRange(0, 127)
        self.ch_num.valueChanged.connect(self.update_ch_num)
        options_lyt.addRow('Channel to use', self.ch_num)
        self.theta_low = QtWidgets.QDoubleSpinBox()
        self.theta_low.setValue(6)
        self.theta_low.setRange(1, 20)
        self.theta_low.valueChanged.connect(
            partial(self.update_filter, 'theta_low'))
        options_lyt.addRow('Theta low frequency (Hz)', self.theta_low)
        self.theta_high = QtWidgets.QDoubleSpinBox()
        self.theta_high.setValue(10)
        self.theta_high.setRange(1, 20)
        self.theta_high.valueChanged.connect(
            partial(self.update_filter, 'theta_high'))
        options_lyt.addRow('Theta high frequency (Hz)', self.theta_high)
        self.delta_low = QtWidgets.QDoubleSpinBox()
        self.delta_low.setRange(.1, 5)
        self.delta_low.setValue(1)
        options_lyt.addRow('Delta low frequency (Hz)', self.delta_low)
        self.delta_low.valueChanged.connect(
            partial(self.update_filter, 'delta_low'))
        self.delta_high = QtWidgets.QDoubleSpinBox()
        self.delta_high.setRange(.1, 5)
        self.delta_high.setValue(4)
        options_lyt.addRow('Delta high frequency (Hz)', self.delta_high)
        self.delta_high.valueChanged.connect(
            partial(self.update_filter, 'delta_high'))
        self.ratio_th = QtWidgets.QDoubleSpinBox()
        self.ratio_th.setSingleStep(0.1)
        self.ratio_th.setValue(0.45)
        self.ratio_th.setRange(.1, 50)
        self.ratio_th.valueChanged.connect(self.update_ratio)
        options_lyt.addRow('Delta / Theta ratio threshold', self.ratio_th)
        self.acc_th = QtWidgets.QDoubleSpinBox()
        self.acc_th.setValue(3)
        self.acc_th.valueChanged.connect(self.update_acc)
        options_lyt.addRow('Motion threshold', self.acc_th)
        self.buffer_dur = QtWidgets.QSpinBox()
        self.buffer_dur.setRange(4, 15)
        self.buffer_dur.setValue(4)
        self.buffer_dur.valueChanged.connect(self.buffer_dur_update)
        options_lyt.addRow('Buffer duration (s)', self.buffer_dur)
        self.win_dur = QtWidgets.QSpinBox()
        self.win_dur.setRange(2, 10)
        self.win_dur.setValue(2)
        self.win_dur.valueChanged.connect(self.win_dur_update)
        options_lyt.addRow('Window duration (s)', self.win_dur)
        self.sr_lbl = QtWidgets.QLabel('20000')
        options_lyt.addRow('Sampling rate (Hz)', self.sr_lbl)
        self.arduino_port = QtWidgets.QComboBox(self)
        self.init_ports()
        self.arduino_port.currentTextChanged.connect(self.connect_arduino)
        options_lyt.addRow('Arduino port', self.arduino_port)

        # INFO
        self.rem_text = QtWidgets.QLabel()
        self.rem_text.setText('')
        options_lyt.addRow('Sleeping Info', self.rem_text)

        # Buttons
        v_lyt_right.addLayout(options_lyt)
        h_lyt_btn = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton('&Connect')
        self.connect_btn.clicked.connect(self.connect)
        self.stop_btn = QtWidgets.QPushButton('Sto&p')
        self.stop_btn.setEnabled(False)
        self.start_btn = QtWidgets.QPushButton('&Start')
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        h_lyt_btn.addWidget(self.connect_btn)
        h_lyt_btn.addWidget(self.stop_btn)
        h_lyt_btn.addWidget(self.start_btn)
        v_lyt_right.addLayout(h_lyt_btn)
        v_lyt_right.addSpacerItem(QtWidgets.QSpacerItem(1, 10, QtWidgets.QSizePolicy.Minimum,
                                                        QtWidgets.QSizePolicy.Expanding))

        # Finish placing stuff
        self._splitter.addWidget(left_wdg)
        self._splitter.addWidget(scroller)
        self._main_lyt.addWidget(self._splitter)

        # Graphs
        self.ratio_th_marker = pg.InfiniteLine(0, 0, movable=True)
        self.acc_th_marker = pg.InfiniteLine(0, 0, movable=True)
        self.lfp_curve = self.lfp_plot.getPlotItem().plot()
        self.ratio_curve = self.ratio_plot.getPlotItem().plot()
        self.acc_curve = self.acc_plot.getPlotItem().plot()
        self.ratio_plot.addItem(self.ratio_th_marker)
        self.acc_plot.addItem(self.acc_th_marker)
        self.ratio_th_marker.sigPositionChangeFinished.connect(
            self.ratio_marker_moved)
        self.acc_th_marker.sigPositionChangeFinished.connect(
            self.acc_marker_moved)

        # Logging end of init
        self.logger.info('Initialization of interface done.')
        # Start
        self.show()

    def acc_marker_moved(self):
        self.update_acc(self.acc_th_marker.value())

    def buffer_dur_update(self, value):
        # TBD in main class
        pass

    def connect(self):
        # TBD in main class
        pass

    def connect_arduino(self):
        # TBD in main class
        pass

    def init_ports(self):
        ports = get_ports()
        for p in ports:
            self.arduino_port.addItem(f'{p.product} ({p.device})', p)

    def ratio_marker_moved(self):
        self.update_ratio(self.ratio_th_marker.value())

    def start(self):
        # TBD in main class
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)

    def stop(self):
        # TBD in main class
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_acc(self, value: float):
        # TBD in main class
        self.acc_th.blockSignals(True)
        self.acc_th.setValue(value)
        self.acc_th_marker.setValue(value)
        self.acc_th.blockSignals(False)

    def update_ch_num(self, ch_ix: int):
        # TBD in main class
        pass

    def update_filter(self, frequency: str, value: float):
        # TBD in main class
        pass

    def update_headstage(self, value: str):
        # TBD in main class
        pass

    def update_ratio(self, value: float):
        # TBD in main class
        self.ratio_th.blockSignals(True)
        self.ratio_th.setValue(value)
        self.ratio_th_marker.setValue(value)
        self.ratio_th.blockSignals(False)

    def win_dur_update(self, value):
        # TBD in main class
        pass
    
    def write_session_stimulations(self):
        # In main class
        pass

class LRD(UI):
    data_ready = QtCore.pyqtSignal(dict)
    sleeping = QtCore.pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.data_queue = Queue()
        # Connection to Intan software for parameters
        self.intan_master = IntanMaster(auto_retry=True)
        self.intan_cmd_th = QtCore.QThread(self)
        self.intan_cmd_th.finished.connect(self.finished_cmd_th)
        self.intan_cmd_th.start()
        self.intan_master.moveToThread(self.intan_cmd_th)
        self.intan_master.connect_ev.connect(self.master_connected)
        self.intan_master.disconnect_ev.connect(self.master_disconnected)
        # Connection to Intan software for data
        self.streamer = Streamer(self.data_queue)
        self.stream_th = QtCore.QThread(self)
        self.stream_th.finished.connect(self.finished_stream_th)
        self.stream_th.start()
        self.streamer.moveToThread(self.stream_th)
        self.streamer.connect_ev.connect(self.streamer_connected)
        self.streamer.data_error.connect(self.data_error)
        self.queue_timer = QtCore.QTimer()
        self.queue_timer.timeout.connect(self.fetch_data)
        # Data analysis
        self.comp_th = QtCore.QThread(self)
        self.rem_comp = REMDetector(None, self.delta_low.value(), self.delta_high.value(),
                                    self.theta_low.value(), self.theta_high.value(), fs=1250)
        self.comp_th.finished.connect(self.finish_comp_th)
        self.comp_th.start()
        self.rem_comp.moveToThread(self.comp_th)
        self.data_ready.connect(self.rem_comp.analyze_dict)
        self.rem_comp.data_ready.connect(self.comp_done)

        # Data buffers
        self.rolled_in = 0
        self.buf_size = int(2 * 20000 * 2)     # FIXME: to parametrize
        self.short_buff_size = int(1800)
        short_buff = np.zeros(self.short_buff_size)
        buff = np.zeros(self.buf_size)
        self.buffers = {'ratio': short_buff.copy(), 'motion': short_buff.copy(),
                        'theta': buff.copy(), 'delta': buff.copy(),
                        'lfp': buff.copy(), 'acc':  np.zeros((self.buf_size, 3)),
                        'time': buff.copy(), 't_ratio': short_buff.copy()}
        # Arduino
        self.port = None
        self.arduino = None
        # Check periodically if port is open
        self.open_timer = QtCore.QTimer(self)
        self.open_timer.timeout.connect(self.is_port_open)
        self._is_port_open = False
        self.ard_th = QtCore.QThread()
        self.ard_th.finished.connect(self.finished_ard_th)
        self.connect_arduino()

        # Stimulations
        self.REM = False
        self.current_rem_start = None
        self.current_rem_end = None
        self.REM_intervals = []

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        self.intan_master.restore_color()
        self.finished_cmd_th()
        self.finished_stream_th()
        self.finish_comp_th()
        self.finished_ard_th()

    def comp_done(self, data):
        # Fixme: TBD
        # self.logger.debug('REM detection is done')
        self.lfp_curve.setData(self.buffers['time'], self.buffers['lfp'])
        last_time = self.buffers['t_ratio'][-1]
        n_pts = len(data['lfp'])
        dt = 1 / 1250

        new_ts = np.linspace(dt, n_pts*dt, n_pts) + last_time
        self.add_to_buffer('t_ratio', [new_ts.mean()])
        self.add_to_buffer('ratio', [data['ratio']])
        self.add_to_buffer('motion', [data['motion']])

        self.ratio_curve.setData(
            self.buffers['t_ratio'], self.buffers['ratio'])
        self.acc_curve.setData(self.buffers['t_ratio'], self.buffers['motion'])

        ''' 
        Here I define a set of rules that will define if we are stimulating or not. 
        '''

        if np.all(self.buffers['motion'][-5:-1] < self.acc_th.value()) and np.any(self.buffers['ratio'][-4:-1] > self.ratio_th.value()):
            self.logger.info(
                'The animal seems to REM Sleep according to data')
            self.sleeping.emit(True)
            if self.REM is False:
                self.REM = True
                self.current_rem_start = self.buffers['time'][-1]
            self.rem_text.setText('REM SLEEP')

        else:
            if self.REM is True:
                self.REM = False
                self.current_rem_end = self.buffers['time'][-1]
                self.REM_intervals.append([self.current_rem_start,self.current_rem_end])
            self.sleeping.emit(False)
            self.rem_text.setText('')

    def connect(self):
        self.intan_master.headstage = self.headstage.currentText().lower()
        self.intan_master.connect()
        self.streamer.connect()
        self.sr_lbl.setText(self.intan_master.sampling_rate)

    def connect_arduino(self):
        super().connect_arduino()
        self.close_arduino()
        port = self.arduino_port.currentData()
        if port is None:
            self.logger.warning('No arduino plugged')
            return
        device = port.device
        try:
            self.logger.info(
                f'Tries to connect to {self.arduino_port.currentData()}')
            self.port = serial.Serial(device, timeout=3)
            self.open_timer.start(1000)
        except serial.SerialException as e:
            msg = f'Opening of serial port {device} impossible: {e.strerror}'
            self.logger.error(msg)

    def close_arduino(self):
        if self.port is not None:
            self.port.close()

    def is_port_open(self):
        if self.port is not None:
            self._is_port_open = self.port.is_open
        if self._is_port_open:
            self.open_timer.stop()
            self.arduino = Trigger(self.port)
            self.logger.info('Trigger instances created')
            self.sleeping.connect(self.arduino.trig)
            self.arduino.moveToThread(self.ard_th)

    def data_error(self):
        self.stop()

    def add_to_buffer(self, buffer_name, data):
        n_pts = len(data)
        self.buffers[buffer_name] = np.roll(self.buffers[buffer_name], -n_pts)
        self.buffers[buffer_name][-n_pts:] = data

    def fetch_data(self):
        try:
            data = self.data_queue.get_nowait()
        except Empty:
            return
        ts = data[0, :]
        lfp = data[1, :]
        acc = data[2:, :]
        n_pts = len(lfp)
        self.add_to_buffer('time', ts)
        self.add_to_buffer('lfp', lfp)
        self.add_to_buffer('acc', acc.T)
        self.rolled_in += n_pts
        # self.logger.debug('Drawing')
        self.lfp_curve.setData(self.buffers['time'], self.buffers['lfp'])
        # FIXME: Windows should overlap
        n_pts_analysis = 20000 * 2
        if self.rolled_in >= n_pts_analysis:  # FIXME: to parametrize
            self.data_ready.emit({'lfp': self.buffers['lfp'][-self.rolled_in:],
                                  'acc': self.buffers['acc'][-self.rolled_in:, :]})
            self.rolled_in = 0

    def finished_ard_th(self):
        self.ard_th.quit()
        self.ard_th.wait()

    def finished_cmd_th(self):
        self.intan_cmd_th.quit()
        self.intan_cmd_th.wait()

    def finish_comp_th(self):
        self.comp_th.quit()
        self.comp_th.wait()

    def finished_stream_th(self):
        self.stream_th.quit()
        self.stream_th.wait()

    def master_connected(self, status: bool):
        if status:
            self.intan_master.set_ch_ix(self.ch_num.value())
            self.intan_master.start_pinging()
            self.connect_btn.setEnabled(False)
            self.headstage.setEnabled(False)

    def master_disconnected(self):
        self.logger.error('Intan disconnected')
        self.stop()
        self.connect_btn.setEnabled(True)
        self.headstage.setEnabled(True)
        self.start_btn.setEnabled(False)

    def start(self):
        super(LRD, self).start()
        self.intan_master.stop_pinging()
        self.intan_master.set_ch_ix(self.ch_num.value())
        self.intan_master.run()
        self.queue_timer.start(1)
        self.streamer.start_stream()

        self.start_time = time.strftime('%y%m%d_%H%M%S')

    def stop(self):
        super().stop()
        self.intan_master.clear_all_data_outputs()
        self.intan_master.start_pinging()
        self.streamer.stop_stream()

        self.write_session_stimulations()

    def streamer_connected(self, status: bool):
        if status:
            self.start_btn.setEnabled(True)

    def update_ch_num(self, ch_ix: int):
        super().update_ch_num(ch_ix)
        self.intan_master.set_ch_ix(ch_ix)

    def write_session_stimulations(self):
        super().write_session_stimulations()
        intervals_array = np.array(self.REM_intervals)
        np.save(f'REMStim-{self.start_time}',intervals_array)
        self.REM_intervals = []

def handle_exception(logname, exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions and print in logger."""
    logger = logging.getLogger(logname)
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(
        exc_type, exc_value, exc_traceback))


if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    win = LRD()
    sys.exit(qApp.exec_())
