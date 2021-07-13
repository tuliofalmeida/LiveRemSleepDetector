# Imports
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread
from rem_obj import REMDetector
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.lines import Line2D
import matplotlib
import neuroseries as nts
import numpy as np
import bk.load
import bk.signal
import scipy
import scipy.signal
import scipy.stats

import time
import rem

# Ensure using PyQt5 backend
matplotlib.use('QT5Agg')

COLOR = 'BLACK'
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['axes.edgecolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR
matplotlib.rcParams['figure.facecolor'] = '#efefef'
matplotlib.rcParams['axes.facecolor'] = '#efefef'
matplotlib.rcParams['svg.fonttype'] = 'none'


# Matplotlib canvas class to create figure
class MplCanvas(Canvas):
    def __init__(self):
        self.fig = Figure()
        # Setting up axes
        self.ax_lfp = self.fig.add_subplot(311)
        self.ax_lfp.set_title('LFP')
        self.ax_spectro = self.fig.add_subplot(312, sharex=self.ax_lfp)
        self.ax_spectro.set_title('Power')
        self.ax_motion = self.fig.add_subplot(313, sharex=self.ax_lfp)
        self.ax_motion.set_title('Motion')
        # Plots on the LFP graph
        self.lfp_line: Line2D = self.ax_lfp.plot(0, 0, 'grey')[0]
        self.theta_line: Line2D = self.ax_lfp.plot(0, 0, 'red', alpha=.7, linewidth=.5)[0]
        self.delta_line: Line2D = self.ax_lfp.plot(0, 0, 'green', alpha=.7, linewidth=.5)[0]
        self.ax_lfp.set_ylim(-35000, 35000)
        # Plots on the theta / delta ratio axis
        self.ratio_line: Line2D = self.ax_spectro.plot(0, 0, 'red')[0]
        self.ratio_th_line: Line2D = self.ax_spectro.axhline(0, color='r', picker=True)
        self.ax_spectro.set_ylim([0, 5000])
        # Plots on the motion axis
        self.motion_line: Line2D = self.ax_motion.plot(0, 0)[0]
        self.motion_th_line: Line2D = self.ax_motion.axhline(0, color='r')
        self.ax_motion.set_ylim([-250, 250])
        # All plots
        self.all_lines = (self.lfp_line, self.theta_line, self.delta_line,
                          self.ratio_line, self.motion_line)

        self.fig.set_tight_layout(True)
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

    def plot(self, lfp, theta, delta, ratio, motion, **kwargs):
        if len(self.lfp_line.get_xdata()) < 10:
            n_pts = len(lfp)
            for l in self.all_lines:
                l.set_xdata(np.arange(n_pts))  # Fixme: add the sampling rate here
                l.axes.set_xlim(0, n_pts)
        self.lfp_line.set_ydata(lfp)
        self.theta_line.set_ydata(theta)
        self.delta_line.set_ydata(delta)
        self.ratio_line.set_ydata(ratio)
        self.motion_line.set_ydata(motion)
        self.fig.canvas.draw()


# Matplotlib widget
class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, ui=None):
        self.ui = ui

        self.lfp_channel = 0
        self.motion_channel = 0

        self.low_delta = 0.1
        self.high_delta = 4

        self.low_theta = 4
        self.high_theta = 12

        self.ratio_treshold = 0
        self.motion_treshold = 0

        self.window_length = 1

        # self.update_params()

        QtWidgets.QWidget.__init__(self, parent)  # Inherit from QWidget
        self.canvas = MplCanvas()  # Create canvas object
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl = QtWidgets.QVBoxLayout()  # Set box for plotting
        self.vbl.addWidget(self.toolbar)
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)

        self.comp_th = QThread(self)
        self.rem_comp = REMDetector(None, self.low_delta, self.high_delta,
                                    self.low_theta, self.high_theta, fs=20000)
        self.comp_th.finished.connect(self.finish_comp_th)
        self.rem_comp.moveToThread(self.comp_th)
        self.comp_th.start()
        self.rem_comp.data_ready.connect(self.comp_done)
        self.buf_size = 2000 * 5    # FIXME: to parametrize
        buff = np.zeros(self.buf_size)
        self.buffers = {'ratio': buff.copy(), 'motion': buff.copy(),
                        'theta': buff.copy(), 'delta': buff.copy(),
                        'lfp': buff.copy(), 'acc': buff.copy()}
        self.c_ix = 0
        self.start = 200

    def comp_done(self, data):
        if self.c_ix >= self.buf_size:
            self.c_ix = 0
        data_ex = list(data.values())[0]
        n_pts = len(data_ex)
        space_left = self.buf_size - self.c_ix
        remaining = n_pts - space_left
        for k, v in data.items():
            if k !='lfp':
                continue
            first_part = min(space_left, len(v))
            self.buffers[k][self.c_ix:self.c_ix+first_part] = v[:first_part]
            if remaining > 0:
                self.buffers[k][:remaining] = v[space_left:]
        self.c_ix += n_pts
        self.canvas.plot(**self.buffers)

    def finish_comp_th(self):
        self.comp_th.quit()
        self.comp_th.wait()

    def update_params(self):
        self.lfp_channel = np.int(self.ui.lfp_channel.text())
        self.motion_channel = np.int(self.ui.motion_channel.text())

        self.low_delta = np.int(self.ui.low_delta.text())
        self.high_delta = np.int(self.ui.high_delta.text())

        self.low_theta = np.int(self.ui.low_theta.text())
        self.high_theta = np.int(self.ui.high_theta.text())

        self.ratio_treshold = np.int(self.ui.ratio_lfp.value())
        self.motion_treshold = np.int(self.ui.ratio_motion.value())

        self.window_length = np.float(self.ui.window_length.text())





#
# def compute_graph(path, lfp_channel, motion_channel, start, end, low_delta, high_delta, low_theta,
#                   high_theta):
#     data = np.memmap(path, dtype=np.int16)
#     data = data.reshape((-1, 137))
#     t = np.arange(start, end, 1 / 20_000, dtype=np.float64)
#     print(t.shape)
#     lfp = nts.Tsd(t, data[np.int(start * 20_000):np.int(end * 20_000), lfp_channel], time_units='s')
#
#     motion = nts.Tsd(t, data[np.int(start * 20_000):np.int(end * 20_000), motion_channel],
#                      time_units='s')
#     # lfp = bk.load.lfp(self.lfp_channel,self.start,self.end,dat = True,frequency = 20_000)
#     # motion = bk.load.lfp(self.motion_channel,self.start,self.end,dat = True,frequency = 20_000)
#
#     lfp = scipy.signal.decimate(lfp.values, 16)
#     t_down = np.linspace(start, end, len(lfp))
#
#     print(lfp.shape)
#     lfp = nts.Tsd(t_down, lfp, time_units='s')
#
#     motion = scipy.signal.decimate(motion.values, 16)
#     motion = np.diff(motion, append=motion[-1])
#     motion = nts.Tsd(t_down, motion, time_units='s')
#
#     filt_theta = bk.signal.passband(lfp, low_theta, high_theta)
#     # filt_delta = bk.signal.passband(lfp,low_delta,high_delta)
#     filt_delta = bk.signal.lowpass(lfp, high_delta)
#
#     lfp = nts.Tsd(lfp.index.values, scipy.stats.zscore(lfp.values))
#     filt_theta_z = nts.Tsd(filt_theta.index.values, scipy.stats.zscore(filt_theta.values))
#     filt_delta_z = nts.Tsd(filt_delta.index.values, scipy.stats.zscore(filt_delta.values))
#
#     power_theta, _ = bk.signal.hilbert(filt_theta)
#     power_delta, _ = bk.signal.hilbert(filt_delta)
#
#     ratio = power_theta.values / power_delta.values
#     ratio = nts.Tsd(t_down, ratio, time_units='s')
#     # t = t+lfp.as_units('s').index.values[0]
#
#     return (lfp, filt_theta_z, filt_delta_z, ratio, motion)
