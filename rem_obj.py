import numpy as np
from typing import Optional
from PyQt5.QtCore import QObject, pyqtSignal
import rem


class REMDetector(QObject):
    data_ready = pyqtSignal(dict)

    def __init__(self, parent: Optional['QObject'] = None,
                 low_delta: float = .1, high_delta: float = 3,
                 low_theta: float = 4, high_theta: float = 10, fs: int = 1250) -> None:
        super().__init__(parent)
        self.low_delta = low_delta
        self.high_delta = high_delta
        self.low_theta = low_theta
        self.high_theta = high_theta
        self.fs = fs
        self._last_ratio: Optional[np.ndarray] = None
        self._last_motion: Optional[np.ndarray] = None

    def analyze_dict(self, data):
        self.analyze(**data)

    def analyze(self, lfp, acc):
        lfp = rem.downsample(lfp)
        acc = rem.downsample(acc)
        ratio, theta, delta, motion = rem.is_sleeping(lfp, acc,
                                                      self.low_delta, self.high_delta,
                                                      self.low_theta, self.high_theta,
                                                      self.fs)
        self._last_motion = motion
        self._last_ratio = ratio
        self.data_ready.emit({'ratio': ratio, 'motion': motion, 'theta': theta, 'delta': delta,
                              'lfp': lfp, 'acc': acc})
