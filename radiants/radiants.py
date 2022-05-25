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

