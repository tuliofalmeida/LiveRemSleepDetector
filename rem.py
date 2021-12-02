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
    # zero_phase = False ?
    dwn = signal.decimate(raw_sig, factor, ftype='iir', axis=0)  # FIR is slower than IIR
    return dwn


def theta_delta(lfp, low_delta, high_delta, low_theta, high_theta, fs=1250):
    bands = [(low_delta, high_delta), (low_theta, high_theta)]
    nperseg = (2 / low_delta) * fs
    freqs, psd = get_spectrum(lfp, fs, nperseg)
    delta, theta = bandpower(bands, freqs, psd)
    ratio = theta / delta
    return ratio, theta, delta

def speed(acc_sig):
    acc = np.linalg.norm(acc_sig, axis=1)
    # med = np.median(acc)
    # mad = np.median(np.abs(acc-med))
    # acc = np.abs(acc-med) / mad
    # acc = np.abs(acc-acc.mean()) / acc.std()
    # acc = np.mean(acc)
    acc = np.mean(np.abs(np.diff(acc)))
    return acc


def is_sleeping(lfp, acc, low_delta=.1, high_delta=3, low_theta=4, high_theta=10, fs=1250):
    ratio, theta, delta = theta_delta(lfp, low_delta, high_delta, low_theta, high_theta, fs=fs)
    motion = speed(acc)
    return ratio, theta, delta, motion


def get_spectrum(data: np.ndarray, sf, nperseg):
    """

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    nperseg

    Returns
    -------
    freqs
    psd
    """

    # nperseg = (2 / low) * sf
    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)
    return freqs, psd


def bandpower(bands, freqs, psd):
    """
    Compute the average power of the signal x in a specific frequency band.
    Taken from: https://raphaelvallat.com/bandpower.html

    Parameters
    ----------
    bands : list of tuples
        List of tuples with Lower and upper frequencies of the band of interest.
    freqs
    psd

    Return
    ------
    bp : float
        Absolute or relative band power.
    """

    # band = np.asarray(band)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    powers = [None for _ in bands]
    for ix, band in enumerate(bands):
        low, high = band
        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Integral approximation of the spectrum using Simpson's rule.
        bp = simps(psd[idx_band], dx=freq_res)
        powers[ix] = bp

    return powers


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
