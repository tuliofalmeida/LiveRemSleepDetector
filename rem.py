from scipy.stats import zscore
from scipy import signal
import numpy as np
from numpy.random import Generator, MT19937
import matplotlib.pyplot as plt
import neuroseries as nts
import scipy
import bk

gen = Generator(MT19937(6))


def bandpass(sig, low, high, fs=1250, order=4):
    sos = signal.butter(order, [low, high], btype='bandpass', output='sos', fs=fs)
    filtered = signal.sosfiltfilt(sos, sig)
    return filtered


def lowpass(sig, cut, fs=1250, order=4):
    sos = signal.butter(order, cut, btype='lowpass', output='sos', fs=fs)
    filtered = signal.sosfiltfilt(sos, sig)
    return filtered


def downsample(raw_sig, factor=16):
    dwn = signal.decimate(raw_sig, factor, ftype='fir')
    return dwn


def get_band_power(raw_sig, low, high, fs=20000, factor=16, order=4):
    dwn_sig = downsample(raw_sig, factor)
    dwn_fs = int(fs // factor)
    f_sig = bandpass(dwn_sig, low, high, dwn_fs, order)
    h_transf = signal.hilbert(f_sig)
    power = np.abs(h_transf) ** 2
    return power


def delta_theta(lfp, low_delta, high_delta, low_theta, high_theta, fs=20000):
    theta_power = get_band_power(lfp, low_theta, high_theta, fs)
    delta_power = get_band_power(lfp, low_delta, high_delta, fs)
    ratio = zscore(theta_power) / zscore(delta_power)
    return ratio


def speed(acc_sig, factor=16, fs=20000):
    motion = downsample(acc_sig)
    dwn_fs = fs / factor
    n_pts = len(motion)
    end_time = n_pts / dwn_fs
    t = np.linspace(0, end_time, n_pts)
    d_motion = np.gradient(motion, t)
    return d_motion


def is_sleeping(lfp, acc, low_delta=.1, high_delta=3, low_theta=4, high_theta=10, fs=20000):
    ratio = delta_theta(lfp, low_delta, high_delta, low_theta, high_theta, fs=fs)
    motion = speed(acc, fs=fs)
    return ratio, motion


def generate_data(dur=60, fs=20000, noise=1):
    """
    Generates fake data to test analysis functions
    Noise during first third, then delta + theta for a third, then noise
    Accelaration is noise during first fourth, then activity during second 1/4th, then just noise
    during next  1/4 then activity again
    LFP: ____----____
    ACC: ___---___---
    REM: 000000100000

    Parameters
    ----------
    dur: int
        Duration in seconds
        Defaut to 60
    fs: int
        Sampling rate in Hz
        Defaults to 20000 Hz
    noise: float
        Scale of noise

    Returns
    -------

    """
    n_pts = dur * fs
    end_time = n_pts / fs
    t = np.linspace(0, end_time, n_pts)
    delta = np.zeros(n_pts)
    theta = np.zeros(n_pts)
    acc = np.zeros(n_pts) + gen.normal(scale=noise, size=(n_pts,))
    third = n_pts // 3
    fourth = n_pts // 4
    # delta
    for f in gen.uniform(low=1, high=3, size=(5, )):
        delta[third:2*third] += np.sin(2*np.pi*f*t[third:2*third])
    # theta
    for f in gen.uniform(low=4, high=7, size=(5, )):
        theta[third:2*third] += np.sin(2*np.pi*f*t[third:2*third])
    acc[fourth:2*fourth] = gen.normal(loc=3, size=(fourth, ))
    acc[3*fourth:] = gen.normal(loc=3, size=(fourth, ))

    lfp = delta + theta + gen.normal(scale=noise, size=(n_pts,))
    return lfp, acc


def compute_graph(path, lfp_channel, motion_channel, start, end, low_delta, high_delta, low_theta,
                  high_theta):
    data = np.memmap(path, dtype=np.int16)
    data = data.reshape((-1, 137))
    t = np.arange(start, end, 1 / 20_000, dtype=np.float64)
    print(t.shape)
    lfp = nts.Tsd(t, data[np.int(start * 20_000):np.int(end * 20_000), lfp_channel], time_units='s')

    motion = nts.Tsd(t, data[np.int(start * 20_000):np.int(end * 20_000), motion_channel],
                     time_units='s')
    # lfp = bk.load.lfp(self.lfp_channel,self.start,self.end,dat = True,frequency = 20_000)
    # motion = bk.load.lfp(self.motion_channel,self.start,self.end,dat = True,frequency = 20_000)

    lfp = scipy.signal.decimate(lfp.values, 16)
    t_down = np.linspace(start, end, len(lfp))

    print(lfp.shape)
    lfp = nts.Tsd(t_down, lfp, time_units='s')

    motion = scipy.signal.decimate(motion.values, 16)
    motion = np.diff(motion, append=motion[-1])
    motion = nts.Tsd(t_down, motion, time_units='s')

    filt_theta = bk.signal.passband(lfp, low_theta, high_theta)
    # filt_delta = bk.signal.passband(lfp,low_delta,high_delta)
    filt_delta = bk.signal.lowpass(lfp, high_delta)

    lfp = nts.Tsd(lfp.index.values, scipy.stats.zscore(lfp.values))
    filt_theta_z = nts.Tsd(filt_theta.index.values, scipy.stats.zscore(filt_theta.values))
    filt_delta_z = nts.Tsd(filt_delta.index.values, scipy.stats.zscore(filt_delta.values))

    power_theta, _ = bk.signal.hilbert(filt_theta)
    power_delta, _ = bk.signal.hilbert(filt_delta)

    ratio = power_theta.values / power_delta.values
    ratio = nts.Tsd(t_down, ratio, time_units='s')
    # t = t+lfp.as_units('s').index.values[0]

    return (lfp, filt_theta_z, filt_delta_z, ratio, motion)

