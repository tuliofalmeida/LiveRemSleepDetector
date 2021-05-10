import scipy
import scipy.signal
import neuroseries as nts
import numpy as np

def passband(lfp,low,high,fs = 1250,order = 4):
    b,a = scipy.signal.butter(order,[low, high],'band',fs = 1250)
    filtered = scipy.signal.filtfilt(b,a,lfp.values)
    return nts.Tsd(np.array(lfp.index),filtered)

def hilbert(lfp,deg = False):
    xa = scipy.signal.hilbert(lfp)
    power = nts.Tsd(np.array(lfp.index),np.abs(xa)**2)
    phase = nts.Tsd(np.array(lfp.index),np.angle(xa,deg = deg))
    
    return power,phase

def lowpass(lfp,low,fs = 1250, order = 4):
    b,a = scipy.signal.butter(order,low,'low',fs = 1250)
    filtered = scipy.signal.filtfilt(b,a,lfp.values)
    return nts.Tsd(np.array(lfp.index),filtered)