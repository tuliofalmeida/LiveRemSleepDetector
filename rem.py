from scipy.stats import zscore
from scipy import signal
import numpy as np
from numpy.random import Generator, MT19937
from functools import lru_cache
from scipy.signal import welch
from scipy.integrate import simps

gen = Generator(MT19937(6))


@lru_cache(None)
def make_sos_filter(order, low, high, btype, fs):
    sos = signal.butter(order, [low, high], btype=btype, output='sos', fs=fs)
    return sos


def bandpass(sig, low, high, fs=1250, order=4):
    sos = make_sos_filter(order, low, high, 'bandpass', fs)
    filtered = signal.sosfiltfilt(sos, sig)
    return filtered


def lowpass(sig, cut, fs=1250, order=4):
    sos = signal.butter(order, cut, btype='lowpass', output='sos', fs=fs)
    filtered = signal.sosfiltfilt(sos, sig)
    return filtered


def downsample(raw_sig, factor=16):
    # This is the slowest part of it all
    dwn = signal.decimate(raw_sig, factor, ftype='fir')  # FIR is slower than IIR
    return dwn


def get_band_power(dwn_sig, low, high, dwn_fs=1250, order=4):
    f_sig = bandpass(dwn_sig, low, high, dwn_fs, order)
    h_transf = signal.hilbert(f_sig)
    power = np.abs(h_transf) ** 2
    return f_sig, power


def delta_theta(lfp, low_delta, high_delta, low_theta, high_theta, fs=20000, factor=16):
    dwn_sig = downsample(lfp, factor)
    dwn_fs = int(fs // factor)
    theta, theta_power = get_band_power(dwn_sig, low_theta, high_theta, dwn_fs)
    delta, delta_power = get_band_power(dwn_sig, low_delta, high_delta, dwn_fs)
    ratio = zscore(theta_power) / zscore(delta_power)
    return ratio, theta, delta, dwn_sig


def speed(acc_sig, factor=16, fs=20000):
    motion = downsample(acc_sig)
    dwn_fs = fs / factor
    n_pts = len(motion)
    end_time = n_pts / dwn_fs
    t = np.linspace(0, end_time, n_pts)
    d_motion = np.gradient(motion, t)
    return d_motion


def is_sleeping(lfp, acc, low_delta=.1, high_delta=3, low_theta=4, high_theta=10, fs=20000):
    ratio, theta, delta, dwn_sig = delta_theta(lfp, low_delta, high_delta, low_theta, high_theta, fs=fs)
    motion = speed(acc, fs=fs)
    return ratio, theta, delta, motion, dwn_sig


def bandpower(data, sf, band, window_sec=None, relative=False):
    """
    Compute the average power of the signal x in a specific frequency band.
    Taken from: https://raphaelvallat.com/bandpower.html

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """

    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def generate_data(dur=60, fs=20000, noise=1):
    """
    Generates fake data to test analysis functions
    Noise during first third, then delta + theta for a third, then noise
    Acceleration is noise during first fourth, then activity during second 1/4th, then just noise
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
    lfp: np.ndarray
    acc: np.ndarray
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


if __name__ == '__main__' and False:
    fake_lfp, fake_acc = generate_data(dur=5)
    for _ in range(1):
        r, t, d, m, dl = is_sleeping(fake_lfp, fake_acc)
