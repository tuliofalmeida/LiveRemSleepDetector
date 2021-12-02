import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.io import loadmat
from pathlib import Path
from typing import Union
from rem import bandpower, get_spectrum


def load_data(dirpath: Union[str, Path], sr=1250):
    dirpath = Path(dirpath)
    lfp_path = dirpath / 'sample.lfp'
    states_path = dirpath / 'States.mat'
    lfp = np.memmap(lfp_path, dtype=np.int16)
    lfp = lfp.reshape((-1, 6))
    r = loadmat(states_path)
    states_keys = ('Rem', 'drowsy', 'sws', 'wake')
    states = {k: r[k] for k in states_keys}
    end_time = lfp.shape[0] / sr
    t = np.arange(0, end_time, 1/sr)
    return lfp, states, t


def plot_states(lfp, states, t, ch_ix: int = 0, tmin=0, tmax=None, ax=None):
    if tmax is None:
        tmax = t.max()
    gi = (t > tmin) & (t <= tmax)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(t[gi], lfp[gi, ch_ix], c='.2')
    ymin, ymax = ax.get_ylim()
    states_names = sorted(list(states.keys()))
    colors = ('#ef476f55', '#ffd16655', '#06d6a055', '#118ab255')
    states_colors = dict(zip(states_names, colors))
    names_col = list(states_colors.items())
    leg = [Rectangle((0, 0), 5, 5, ec='none', fc=it[1])
           for it in names_col]
    for st_name, st_lims in states.items():
        for lim in st_lims:
            t_start = t[t.searchsorted(lim[0])]
            if t_start < tmin:
                continue
            if t_start > tmax:
                break
            t_end = t[t.searchsorted(lim[1])]
            t_width = t_end - t_start
            r = Rectangle((t_start, ymin), t_width, ymax-ymin, ec='none', fc=states_colors[st_name])
            ax.add_patch(r)
    ax.legend(leg, [it[0] for it in names_col])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('LFP (AU)')


def reshape_window(sig, t, sr, window_dur):
    n_pts_win = window_dur * sr
    n_windows = (len(lfp) // n_pts_win)
    n_pts_total = n_windows * n_pts_win
    sig_seg = sig[:n_pts_total].reshape((n_windows, n_pts_win))
    t_seg = t[:n_pts_total].reshape((n_windows, n_pts_win))
    t_ratio = t_seg.mean(1)
    return sig_seg, t_ratio


def compute_power(lfp, t, sr=1250, window_dur=5):
    low_delta = 1
    high_delta = 3
    low_theta = 7
    high_theta = 10
    lfp_seg, t_ratio = reshape_window(lfp, t, sr, window_dur)

    all_theta = []
    all_delta = []
    for sig in lfp_seg:
        freqs, psd = get_spectrum(sig, sr, (2 / low_delta) * sr)
        theta_power, delta_power = bandpower([(low_theta, high_theta), (low_delta, high_delta)],
                                             freqs, psd)
        # delta_power = bandpower((low_delta, high_delta), freqs, psd)
        all_theta.append(theta_power)
        all_delta.append(delta_power)

    theta = np.hstack(all_theta)
    delta = np.hstack(all_delta)

    return theta, delta, t_ratio


def get_starts_stops(is_state):
    is_state = is_state.astype(np.int8)
    ix_starts, = np.nonzero(np.diff(is_state) == 1)
    ix_stops, = np.nonzero(np.diff(is_state) == -1)
    if len(ix_starts) < len(ix_stops):
        if ix_starts[0] > ix_stops[0]:
            ix_stops = ix_stops[1:]
        else:
            ix_starts = ix_starts[:-1]

    return ix_starts, ix_stops


def is_sleep(theta, delta, t, th=.3, max_gap=.1, min_dur=5):
    dt = np.median(np.diff(t))
    max_n_gap = int(max_gap / dt) + 1
    min_n_ep = int(min_dur / dt) + 1
    ratio = delta / theta
    is_rem = ratio < th
    in_rem = False
    n_out = 0
    to_replace = []
    for ix, v in enumerate(is_rem):
        if v:
            # If ratio < th: in rem
            in_rem = True
            if len(to_replace) > 0:
                # If ratio < th but we have some gap to fill
                for repix in to_replace:
                    is_rem[repix] = True
                to_replace = []
                n_out = 0
        elif in_rem and n_out < max_n_gap:
            # if ratio > th but not for long we will fill the gap
            n_out += 1
            to_replace.append(ix)
        elif in_rem and n_out >= max_n_gap:
            # if ratio > th for too long, we forget, we no longer are in rem
            n_out = 0
            to_replace = []
            in_rem = False
    ix_starts, ix_stops = get_starts_stops(is_rem)
    ep_len = ix_stops - ix_starts
    to_rm = ep_len < min_n_ep
    for start, stop in zip(ix_starts[to_rm], ix_stops[to_rm]):
        is_rem[start:stop+1] = False
    ix_starts, ix_stops = get_starts_stops(is_rem)
    rem_ep = np.vstack((t[ix_starts], t[ix_stops])).T
    return is_rem, rem_ep


def is_still(all_acc, t, sr=1250, window_dur=5):
    acc_norm = np.linalg.norm(all_acc, axis=1)
    acc_seg, t_seg = reshape_window(acc_norm, t, sr, window_dur)
    acc = np.mean(acc_seg, 1)
    med = np.median(acc)
    mad = np.median(np.abs(acc-med))
    acc = np.abs(acc-med) / mad
    still = acc < 3
    return still


def combine_power_acc(is_rem, is_still, t):
    full_cond = is_rem & is_still
    starts, stops = get_starts_stops(full_cond)
    ep = np.vstack((t[starts], t[stops])).T
    return ep


if __name__ == '__main__':
    test_path = '/home/remi/TDS/Programmation/InProgress/LiveRemSleepDetector/lfp_sample_and_sleep_scoring'
    plt.ion()
    lfp, st, t = load_data(test_path)
    s = lfp[:, 1]
    theta, delta, tseg = compute_power(s, t, window_dur=6)
    is_rem, _ = is_sleep(theta, delta, tseg, max_gap=0, min_dur=0, th=0.45)
    still = is_still(lfp[:, 3:], t, window_dur=6)
    ep = combine_power_acc(is_rem, still, tseg)
    # FIXME: Fill gaps / delete small episodes also after combinations
    rem_st = {'Rem': ep}
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(19, 5))
    plot_states(lfp, st, t, 1, 4000, 6000, axs[0])
    plot_states(lfp, rem_st, t, 1, 4000, 6000, axs[1])
    axs[0].set_title('Manual scoring')
    axs[1].set_title(r'Automated scoring on $\delta\ /\ \theta$ and accelerometer')
    fig.set_tight_layout(True)
