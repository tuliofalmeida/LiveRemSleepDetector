# Imports
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer,QDateTime

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib
import neuroseries as nts
import numpy as np
import bk.load
import bk.signal
import scipy
import scipy.signal
import scipy.stats

import time

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
        
        self.ax_lfp = self.fig.add_subplot(311)
        self.ax_lfp.set_title('LFP')
        self.ax_spectro = self.fig.add_subplot(312,sharex = self.ax_lfp)
        self.ax_spectro.set_title('Power')
        self.ax_motion = self.fig.add_subplot(313,sharex = self.ax_lfp)
        self.ax_motion.set_title('Motion')

        self.fig.tight_layout()

        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)
    def plot(self,lfp,filt_theta,filt_delta,ratio,motion,ratio_treshold,motion_treshold):

        self.ax_lfp.clear()
        self.ax_spectro.clear()
        self.ax_motion.clear()

        self.ax_lfp.plot(lfp.as_units('s'),'grey')
        self.ax_lfp.plot(filt_theta.as_units('s'),'red',alpha = 0.3)
        self.ax_lfp.plot(filt_delta.as_units('s'),'green',alpha = 0.3)
        self.ax_lfp.set_ylim(-3,3)
        self.ax_lfp.set_title('LFP')
        
        self.ax_spectro.clear()
        # self.ax_spectro.set_ylim([0,0.001])
        # img = self.ax_spectro.pcolormesh(t_spectrogram, f, Sxx)
        # img.set_clim(0,20_00)
        # self.ax_spectro.set_ylim(0,20)
        # self.ax_spectro.set_title('Power')

        self.ax_spectro.plot(ratio.as_units('s'))
        self.ax_spectro.axhline(ratio_treshold, color = 'r')
        self.ax_spectro.set_ylim([0,50])




        self.ax_motion.clear()
        self.ax_motion.plot(motion.as_units('s'))
        self.ax_motion.axhline(motion_treshold, color = 'r')
        self.ax_motion.set_ylim([-250,250])



        self.fig.canvas.draw()
        self.ax_motion.set_title('Motion')



# Matplotlib widget
class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None,ui = None):
        
        self.ui = ui

        self.lfp_channel = 0
        self.motion_channel = 0

        self.low_delta = 0
        self.high_delta = 4

        self.low_theta = 4
        self.high_theta = 12


        self.ratio_treshold = 0
        self.motion_treshold = 0

        self.window_length = 1

        # self.update_params()

        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvas()                  # Create canvas object
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.toolbar)
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)


        self.start = 200

    def update_plot(self):
        
        t_t = time.time()
        self.end = self.start + self.window_length
        # bk.load.current_session_linux()
        
        # f,t,Sxx = scipy.signal.spectrogram(lfp.values,fs = 1250,nperseg = 200, noverlap = 100)
        # ratio = np.mean(np.mean(Sxx[(4<f) & (f<12),:],0)/np.mean(Sxx[f<4,:],0))


        lfp,filt_theta_z,filt_delta_z,ratio,motion = compute_graph(self.ui.dat_path,
            self.lfp_channel,self.motion_channel,
            self.start,self.end,
            self.low_delta,self.high_delta,
            self.low_theta,self.high_theta)

        self.canvas.plot(lfp,filt_theta_z,filt_delta_z,ratio,motion,self.ratio_treshold,self.motion_treshold)
        self.start += 1
        self.end +=1
        print(time.time()-t_t)
        QTimer.singleShot(2000,self.update_plot)

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


def compute_graph(path,lfp_channel,motion_channel,start,end,low_delta,high_delta,low_theta,high_theta):
    
    data = np.memmap(path,dtype = np.int16)
    data = data.reshape((-1,137))
    t = np.arange(start,end,1/20_000,dtype = np.float64)
    print(t.shape)
    lfp = nts.Tsd(t ,data[np.int(start*20_000):np.int(end*20_000),lfp_channel],time_units = 's')

    motion = nts.Tsd(t,data[np.int(start*20_000):np.int(end*20_000),motion_channel],time_units = 's')
    # lfp = bk.load.lfp(self.lfp_channel,self.start,self.end,dat = True,frequency = 20_000)
    # motion = bk.load.lfp(self.motion_channel,self.start,self.end,dat = True,frequency = 20_000)

    lfp = scipy.signal.decimate(lfp.values,16)
    t_down = np.linspace(start,end,len(lfp))

    print(lfp.shape)
    lfp = nts.Tsd(t_down,lfp,time_units = 's')

    motion = scipy.signal.decimate(motion.values,16)
    motion = np.diff(motion,append = motion[-1])
    motion = nts.Tsd(t_down,motion,time_units = 's')

    filt_theta = bk.signal.passband(lfp,low_theta,high_theta)
    # filt_delta = bk.signal.passband(lfp,low_delta,high_delta)
    filt_delta = bk.signal.lowpass(lfp,high_delta)


    lfp = nts.Tsd(lfp.index.values,scipy.stats.zscore(lfp.values))
    filt_theta_z = nts.Tsd(filt_theta.index.values,scipy.stats.zscore(filt_theta.values))
    filt_delta_z = nts.Tsd(filt_delta.index.values,scipy.stats.zscore(filt_delta.values))


    power_theta,_ = bk.signal.hilbert(filt_theta)
    power_delta,_ = bk.signal.hilbert(filt_delta)

    ratio = power_theta.values/power_delta.values
    ratio = nts.Tsd(t_down,ratio,time_units = 's')
    # t = t+lfp.as_units('s').index.values[0]

    return(lfp,filt_theta_z,filt_delta_z,ratio,motion)

