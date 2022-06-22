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


def get_speeds_2D(cube, x_axis=None, y_axis=None, temporal_axis=None, dx=None,
        dy=None, dt=None, apodize_rolloff=False):
    if apodize_rolloff:
        try:
            len(apodize_rolloff)
        except:
            apodize_rolloff = [apodize_rolloff] * len(cube.shape)
        rolloff = utils.get_hann_rolloff(cube.shape, apodize_rolloff)
        cube = cube * rolloff
    fcube = np.fft.fftn(cube)
    if dx is None:
        dx = x_axis[1] - x_axis[0]
    if dy is None:
        dy = y_axis[1] - y_axis[0]
    if dt is None:
        dt = temporal_axis[1] - temporal_axis[0]
    omega = np.fft.fftfreq(cube.shape[0], dt)[:, None, None]
    kx = np.fft.fftfreq(cube.shape[2], dx)[None, None, :]
    ky = np.fft.fftfreq(cube.shape[1], dy)[None, :, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        vx = -omega / kx
        vy = -omega / ky
    
    vx = np.broadcast_to(vx, fcube.shape)
    vy = np.broadcast_to(vy, fcube.shape)
    # In the above, leave kx, ky and omega as 1-D arrays so the vx and vy calcs
    # are 2D. Below, update those three to have the same shape as vx and vy.
    kx = np.broadcast_to(kx, fcube.shape)
    ky = np.broadcast_to(ky, fcube.shape)
    omega = np.broadcast_to(omega, fcube.shape)
    return vx, vy, fcube, kx, ky, omega


def select_speed_range(vmin, vmax, strips, spatial_axis=None,
        temporal_axis=None, dx=None, dt=None, apodize_rolloff=0,
        filter_rolloff=0):
    v, fstrips = get_speeds(
            strips, spatial_axis, temporal_axis, dx, dt,
            apodize_rolloff=apodize_rolloff)
    mask = (v < vmax) * (v > vmin) * np.isfinite(v)
    return apply_velocity_mask(mask, fstrips, filter_rolloff=filter_rolloff)


def apply_velocity_mask(mask, data, filter_rolloff=0):
    if filter_rolloff > 1:
        # Make a 1D window of the correct shape
        window_1d = scipy.signal.windows.hann(filter_rolloff*2 + 3)
        window_1d = window_1d[1:-1]
        # Make a window of the appropriate shape by making an N-D cube of 1s,
        # and iteratively multiplying by the window oriented horizontally, then
        # vertically, then in depth, etc.
        window = np.ones([window_1d.size] * len(data.shape))
        for i in range(len(data.shape)):
            sel = [None] * len(data.shape)
            sel[i] = slice(None)
            window *= window_1d[tuple(sel)]
        window /= np.sum(window)
        
        # We want the window to smoothly roll off to zero at the edges. There's
        # no pre-defined boundary mode in `scipy.ndimage.convolve` that
        # achieves that---zero padding only gives a rolloff to 0.5 at the edges
        # (where the mask is 1 at the edge). And we want to use
        # `scipy.signal.fftconvolve` instead, since it's ridiculously faster
        # for large mask arrays. `fftconvolve` seems to implicitly zero-pad the
        # array, and to achieve a rolloff to zero, we need to also zero out the
        # outsides of our mask array.
        mask = mask.astype(float)
        # Shift so we're zeroing out and rolling off at the highest frequencies
        mask = np.fft.fftshift(mask)
        edge = window_1d.size // 2
        mask[:edge] = 0
        mask[-edge:] = 0
        mask[:, :edge] = 0
        mask[:, -edge:] = 0
        mask = scipy.signal.fftconvolve(mask.astype(float), window, mode='same')
        mask = np.fft.ifftshift(mask)
    return np.abs(np.fft.ifftn(data * mask))


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
