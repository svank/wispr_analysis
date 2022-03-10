from math import ceil
import warnings

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import reproject
import scipy.ndimage

from . import utils


def dust_streak_filter(img1, img2, img3, radec=True, greatest_allowed_gap=2.5*60*60,
        window_width=9, return_mask=False, sliding_window_stride=1):
    if window_width % 2 != 1:
        raise ValueError("`window_width` should be odd")
    
    with warnings.catch_warnings():
        warnings.filterwarnings(
                action='ignore', message=".*'BLANK' keyword.*")
        warnings.filterwarnings(
                action='ignore', message=".*datfix.*")
        if isinstance(img1, tuple):
            img1, hdr1 = img1
        else:
            img1, hdr1 = fits.getdata(img1, header=True)
        if isinstance(img2, tuple):
            img2, hdr2 = img2
        else:
            img2, hdr2 = fits.getdata(img2, header=True)
        if isinstance(img3, tuple):
            img3, hdr3 = img3
        else:
            img3, hdr3 = fits.getdata(img3, header=True)
    
    if (greatest_allowed_gap and 
            utils.to_timestamp(hdr3['date-avg'])
            - utils.to_timestamp(hdr1['date-avg']) > greatest_allowed_gap):
        if return_mask:
            if return_mask == 'also':
                return img2, np.zeros_like(img2, dtype=bool)
            else:
                return np.zeros_like(img2, dtype=bool)
        return img2
    
    if radec:
        wcs2 = WCS(hdr2, key='A')
        hdr1_out = wcs2
        hdr2_out = wcs2
        hdr3_out = wcs2
        img1_r = reproject.reproject_adaptive((img1, wcs1), hdr1_out, img2.shape,
                kernel='gaussian', return_footprint=False,
                boundary_mode='constant', roundtrip_coords=False)
        img2_r = reproject.reproject_adaptive((img2, wcs2), hdr2_out, img2.shape,
                kernel='gaussian', return_footprint=False,
                boundary_mode='constant', roundtrip_coords=False)
        img3_r = reproject.reproject_adaptive((img3, wcs3), hdr3_out, img2.shape,
                kernel='gaussian', return_footprint=False,
                boundary_mode='constant', roundtrip_coords=False)
        img_fill = np.mean((
            reproject.reproject_adaptive((img1, wcs1), hdr1_out, img2.shape,
                kernel='hann', return_footprint=False,
                boundary_mode='constant', roundtrip_coords=False),
            reproject.reproject_adaptive((img3, wcs3), hdr3_out, img2.shape,
                kernel='hann', return_footprint=False,
                boundary_mode='constant', roundtrip_coords=False)),
            axis=0)
    else:
        img1_r = img1
        img2_r = img2
        img3_r = img3
        img_fill = np.mean((img1, img3), axis=0)
    
    bad_px = np.isnan(img2) + (img2 < 0)
    rows_filter = np.sum(bad_px, axis=1)
    cols_filter = np.sum(bad_px, axis=0)

    trim = (np.sum(rows_filter[:bad_px.shape[0]//2] > .9 * bad_px.shape[0]),
            np.sum(rows_filter[bad_px.shape[0]//2:] > .9 * bad_px.shape[0]),
            np.sum(cols_filter[:bad_px.shape[1]//2] > .9 * bad_px.shape[1]),
            np.sum(cols_filter[bad_px.shape[1]//2:] > .9 * bad_px.shape[1]))
    if np.any(np.array(trim) > 100):
        warnings.warn(f"File {hdr2['filename']} appears to be very bad. Skipping...")
        if return_mask:
            if return_mask == 'also':
                return img2, np.zeros_like(img2, dtype=bool)
            else:
                return np.zeros_like(img2, dtype=bool)
        return img2

    mean_diffs, std_diffs = gen_diffs_distribution(
            img1_r, img3_r, trim, sliding_window_stride, window_width)
    
    larger = np.max((img1_r, img3_r), axis=0)
    sig_excess = ((img2_r - larger) - mean_diffs) / std_diffs

    filter = sig_excess > 1
    basic_mask = sig_excess > .25
    
    filter = scipy.ndimage.binary_closing(filter, np.ones((5,5)))
    
    filter *= basic_mask
    
    min_size = 40
    high_sig_req = 6.5
    low_sig_req = 5
    low_sig_size = 80
    
    labels, n_labels = scipy.ndimage.label(filter, structure=np.ones((3,3)))
    def apply_min_size(input, pos):
        pos = np.unravel_index(pos, filter.shape)
        if len(input) < min_size:
            filter[pos] = 0
    if n_labels > 0:
        scipy.ndimage.labeled_comprehension(filter, labels,
                np.arange(1, n_labels+1), apply_min_size, float, 0,
                pass_positions=True);
    
    labels, n_labels = scipy.ndimage.label(filter, structure=np.ones((3,3)))
    def apply_sig_thresholds(input, pos):
        pos = np.unravel_index(pos, filter.shape)
        max_sig = np.max(sig_excess[pos])
        size = len(input)
        req_sig = max(low_sig_req + (high_sig_req - low_sig_req)
                * (low_sig_size - size) / (low_sig_size - min_size),
                low_sig_req)
        if max_sig < req_sig:
            filter[pos] = 0
    if n_labels > 0:
        scipy.ndimage.labeled_comprehension(filter, labels,
                np.arange(1, n_labels+1), apply_sig_thresholds, float, 0,
                pass_positions=True);
    
    labels, n_labels = scipy.ndimage.label(filter, structure=np.ones((3,3)))
    def defuzz(input, pos):
        pos_unraveled = np.unravel_index(pos, filter.shape)
        max_sig = np.max(sig_excess[pos_unraveled])
        fuzz_level = min(1.5, max(.2, .2 + max_sig / 10))
        pos = pos[sig_excess[pos_unraveled] < fuzz_level]
        pos_unraveled = np.unravel_index(pos, filter.shape)
        filter[pos_unraveled] = 0
    if n_labels > 0:
        scipy.ndimage.labeled_comprehension(filter, labels,
                np.arange(1, n_labels+1), defuzz, float, 0, pass_positions=True);
    
    labels, n_labels = scipy.ndimage.label(filter, structure=np.ones((3,3)))
    def remove_small_features(input, pos):
        if len(input) <= 3:
            filter[np.unravel_index(pos, filter.shape)] = 0
    if n_labels > 0:
        scipy.ndimage.labeled_comprehension(filter, labels,
                np.arange(1, n_labels+1), remove_small_features, float, 0,
                pass_positions=True);
    
    if return_mask and return_mask != 'also':
        return filter

    filtered = np.where(filter, img_fill, img2)
    
    if return_mask == 'also':
        return filtered, filter
    return filtered


def gen_diffs_distribution(img1_r, img3_r, trim, sliding_window_stride, window_width):
    diffs = np.abs(img3_r - img1_r)
    diffs = diffs[trim[0]:diffs.shape[0] - trim[1], trim[2]:diffs.shape[1] - trim[3]]
    sliding_window = np.lib.stride_tricks.sliding_window_view(
            diffs, (window_width, window_width))
    sliding_window = sliding_window[::sliding_window_stride,
            ::sliding_window_stride]
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore',
                message=".*empty slice.*")
        warnings.filterwarnings(action='ignore',
                message=".*degrees of freedom <= 0.*")
        mean_diffs = np.nanmean(sliding_window, axis=(2, 3))
        std_diffs = np.nanstd(sliding_window, axis=(2, 3))
    if sliding_window_stride > 1:
        mean_diffs = np.repeat(mean_diffs, sliding_window_stride, axis=0)
        mean_diffs = np.repeat(mean_diffs, sliding_window_stride, axis=1)
        std_diffs = np.repeat(std_diffs, sliding_window_stride, axis=0)
        std_diffs = np.repeat(std_diffs, sliding_window_stride, axis=1)
    vpad = img1_r.shape[0] - mean_diffs.shape[0] - trim[0] - trim[1]
    hpad = img1_r.shape[1] - mean_diffs.shape[1] - trim[2] - trim[3]
    padding = ((vpad//2 + trim[0], ceil(vpad/2) + trim[1]),
            (hpad//2 + trim[2], ceil(hpad/2) + trim[3]))
    mean_diffs = np.pad(mean_diffs, padding, mode='edge')
    std_diffs = np.pad(std_diffs, padding, mode='edge')
    
    return mean_diffs, std_diffs


def median_filter(img1, img2, img3, radec=True, greatest_allowed_gap=2.5*60*60):
    with warnings.catch_warnings():
        warnings.filterwarnings(
                action='ignore', message=".*'BLANK' keyword.*")
        warnings.filterwarnings(
                action='ignore', message=".*datfix.*")
        warnings.filterwarnings(
                action='ignore', message=".*All-NaN slice.*")
        if isinstance(img1, tuple):
            img1, hdr1 = img1
        else:
            img1, hdr1 = fits.getdata(img1, header=True)
        if isinstance(img2, tuple):
            img2, hdr2 = img2
        else:
            img2, hdr2 = fits.getdata(img2, header=True)
        if isinstance(img3, tuple):
            img3, hdr3 = img3
        else:
            img3, hdr3 = fits.getdata(img3, header=True)
        
        if (utils.to_timestamp(hdr3['date-avg'])
                - utils.to_timestamp(hdr1['date-avg']) > greatest_allowed_gap):
            return img2, hdr2
        
        if radec:
            hdr_out = WCS(hdr2, key='A')
            img1r = reproject.reproject_adaptive((img1, WCS(hdr1, key='A')),
                    hdr_out, img2.shape, kernel='hann', return_footprint=False,
                    boundary_mode='constant', roundtrip_coords=False)
            img2r = img2
            img3r = reproject.reproject_adaptive((img3, WCS(hdr3, key='A')),
                    hdr_out, img2.shape, kernel='hann', return_footprint=False,
                    boundary_mode='constant', roundtrip_coords=False)
            # Make sure we return a header with the helioprojective coords included
            hdr_out = hdr2
        else:
            img1r, img2r, img3r = img1, img2, img3
            hdr_out = hdr2
        return np.nanmedian((img1r, img2r, img3r), axis=0), hdr_out
