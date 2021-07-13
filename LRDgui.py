from typing import Dict
import pyqtgraph as pg
import sys
from functools import partial
from PyQt5 import QtWidgets, QtCore, QtGui
import logging
from logging.handlers import RotatingFileHandler


class UI(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Livre REM sleep detector - Girardeau lab')
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
        logo_lab.setPixmap(QtGui.QPixmap('RatHippoBLA.png').scaledToHeight(200))
        v_lyt_right.addWidget(logo_lab)
        # Parameters
        options_lyt = QtWidgets.QFormLayout()
        self.ch_num = QtWidgets.QSpinBox()
        self.ch_num.valueChanged.connect(self.update_ch_num)
        options_lyt.addRow('Channel to use', self.ch_num)
        self.theta_low = QtWidgets.QDoubleSpinBox()
        self.theta_low.valueChanged.connect(partial(self.update_filter, 'theta_low'))
        options_lyt.addRow('Theta low frequency (Hz)', self.theta_low)
        self.theta_high = QtWidgets.QDoubleSpinBox()
        self.theta_low.valueChanged.connect(partial(self.update_filter, 'theta_high'))
        options_lyt.addRow('Theta high frequency (Hz)', self.theta_high)
        self.delta_low = QtWidgets.QDoubleSpinBox()
        options_lyt.addRow('Delta low frequency (Hz)', self.delta_low)
        self.theta_low.valueChanged.connect(partial(self.update_filter, 'delta_low'))
        self.delta_high = QtWidgets.QDoubleSpinBox()
        options_lyt.addRow('Delta high frequency (Hz)', self.delta_high)
        self.theta_low.valueChanged.connect(partial(self.update_filter, 'delta_high'))
        self.ratio_th = QtWidgets.QDoubleSpinBox()
        self.ratio_th.setSingleStep(0.1)
        self.ratio_th.valueChanged.connect(self.update_ratio)
        options_lyt.addRow('Delta / Theta ratio threshold', self.ratio_th)
        self.acc_th = QtWidgets.QDoubleSpinBox()
        self.acc_th.valueChanged.connect(self.update_acc)
        options_lyt.addRow('Motion threshold', self.acc_th)
        # Buttons
        v_lyt_right.addLayout(options_lyt)
        h_lyt_btn = QtWidgets.QHBoxLayout()
        self.stop_btn = QtWidgets.QPushButton('Sto&p')
        self.start_btn = QtWidgets.QPushButton('&Start')
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
        self.ratio_th_marker.sigPositionChangeFinished.connect(self.ratio_marker_moved)
        self.acc_th_marker.sigPositionChangeFinished.connect(self.acc_marker_moved)

        # Logging end of init
        self.logger.info('Initialization of interface done.')
        # Start
        self.show()

    def acc_marker_moved(self):
        self.update_acc(self.acc_th_marker.value())

    def ratio_marker_moved(self):
        self.update_ratio(self.ratio_th_marker.value())

    def start(self):
        # TBD in main class
        pass

    def stop(self):
        # TBD in main class
        pass

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

    def update_ratio(self, value: float):
        # TBD in main class
        self.ratio_th.blockSignals(True)
        self.ratio_th.setValue(value)
        self.ratio_th_marker.setValue(value)
        self.ratio_th.blockSignals(False)


class LRD(UI):
    pass


def handle_exception(logname, exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions and print in logger."""
    logger = logging.getLogger(logname)
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    win = LRD()
    sys.exit(qApp.exec_())
