import numpy as np
import scipy.ndimage
import scipy.signal

from .. import utils


def get_speeds(strips, spatial_axis=None, temporal_axis=None, dx=None,
        dt=None, apodize_rolloff=False):
    if apodize_rolloff:
        try:
            len(apodize_rolloff)
        except:
            apodize_rolloff = [apodize_rolloff] * 2
        rolloff = utils.get_hann_rolloff(strips.shape, apodize_rolloff)
        strips = strips * rolloff
    fstrips = np.fft.fft2(strips)
    if dx is None:
        dx = spatial_axis[1] - spatial_axis[0]
    if dt is None:
        dt = temporal_axis[1] - temporal_axis[0]
    omega = np.fft.fftfreq(strips.shape[0], dt)
    k = np.fft.fftfreq(strips.shape[1], dx)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = -omega[:, None] / k[None, :]
    return v, fstrips


def select_speed_range(vmin, vmax, strips, spatial_axis=None,
        temporal_axis=None, dx=None, dt=None, apodize_rolloff=0,
        filter_rolloff=0):
    v, fstrips = get_speeds(
            strips, spatial_axis, temporal_axis, dx, dt,
            apodize_rolloff=apodize_rolloff)
    mask = (v < vmax) * (v > vmin) * np.isfinite(v)
    if filter_rolloff > 1:
        window = scipy.signal.windows.hann(filter_rolloff*2)
        window = window[:, None] * window[None, :]
        window /= np.sum(window)
        mask = np.fft.fftshift(mask)
        mask = scipy.ndimage.convolve(mask.astype(float), window,
                mode='nearest')
        mask = np.fft.ifftshift(mask)
    fstrips *= mask
    return np.abs(np.fft.ifft2(fstrips))


def find_radiant(strips, t, fov_angles, window_size=41, v_halfwindow=1,
        ret_extra=False):
    if window_size % 2 != 1:
        raise ValueError("window_size should be odd")
    
    windows = np.lib.stride_tricks.sliding_window_view(
            strips, (window_size, window_size))

    best_vs = []
    best_fts = []
    best_fractions = []
    computed_radiants = []
    computed_radiants_ts = []
    all_fractions = np.zeros(windows.shape[:2])
    all_stat_energy = np.zeros(windows.shape[:2])
    for j in range(windows.shape[0]):
        best_fraction = -np.inf
        for i in range(windows.shape[1]):
            v, ft = get_speeds(
                    windows[j, i], dt=t[1] - t[0], spatial_axis=fov_angles,
                    apodize_rolloff=5)
            ft = np.abs(ft)
            good = np.isfinite(v) * (v > -v_halfwindow) * (v < v_halfwindow)
            stationary_energy = np.sum(np.abs(v*ft)[good])
            fraction = stationary_energy / np.sum(ft)
            all_fractions[j, i] = fraction
            all_stat_energy[j, i] = stationary_energy
            if fraction > best_fraction:
                best_fraction = fraction
                best_v, best_ft = v, ft
                best_idx = i + window_size // 2
        best_vs.append(best_v)
        best_fts.append(best_ft)
        best_fractions.append(best_fraction)
        computed_radiants.append(fov_angles[best_idx])
        computed_radiants_ts.append(t[j + window_size//2])
    if ret_extra:
        extras = dict(
                best_vs=best_vs,
                best_fts=best_fts,
                best_fractions=best_fractions,
                all_fractions=all_fractions,
                all_stat_energy=all_stat_energy)
        return computed_radiants, computed_radiants_ts, extras
    return computed_radiants, computed_radiants_ts
