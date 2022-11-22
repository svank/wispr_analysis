from itertools import repeat
from math import ceil
import os
import subprocess
import warnings

import multiprocessing

from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import reproject
import scipy.ndimage
from tqdm.auto import tqdm

from . import plot_utils
from . import utils


def dust_streak_filter(img1, img2, img3, radec=True,
        greatest_allowed_gap=2.5*60*60,
        window_width=9, return_mask=False, return_header=False,
        sliding_window_stride=1):
    if window_width % 2 != 1:
        raise ValueError("`window_width` should be odd")
    
    with utils.ignore_fits_warnings():
        if isinstance(img1, tuple):
            img1, hdr1 = img1
        else:
            img1 = os.path.expanduser(img1)
            img1, hdr1 = fits.getdata(img1, header=True)
        if isinstance(img2, tuple):
            img2, hdr2 = img2
        else:
            img2 = os.path.expanduser(img2)
            img2, hdr2 = fits.getdata(img2, header=True)
        if isinstance(img3, tuple):
            img3, hdr3 = img3
        else:
            img3 = os.path.expanduser(img3)
            img3, hdr3 = fits.getdata(img3, header=True)
    
    if return_header:
        try:
            commit = subprocess.run(
                    ["git", '-C', os.path.dirname(__file__),
                        "show", "-s", "--format=%H"],
                    check=True, capture_output=True,
                    ).stdout.decode()
        except subprocess.CalledProcessError:
            commit = "<commit hash couldn't be found>"
        hdr2.add_history(f"Processed by dust_streak_filter")
        hdr2.add_history(f" at commit {commit.strip()}")
        hdr2.add_history(f"  file1: {hdr1['filename']}")
        hdr2.add_history(f"  file3: {hdr3['filename']}")
        hdr2.add_history(f"  ra-dec alignment: {radec};"
                         f" window_width: {window_width}")
        hdr2.add_history(f"  sliding_window_stride: {sliding_window_stride}")
    
    gap = (utils.to_timestamp(hdr3['date-avg'])
            - utils.to_timestamp(hdr1['date-avg']))
    if (greatest_allowed_gap and gap > greatest_allowed_gap):
        ret = [img2]
        if return_mask:
            mask = np.zeros_like(img2, dtype=bool)
            if return_mask == 'also':
                ret.append(mask)
            else:
                ret[0] = mask
        if return_header:
            hdr2.add_history(f"Dust streak removal skipped; gap of {gap:.0f}")
            hdr2.add_history(
                    f" exceeded greatest allowable gap of {greatest_allowed_gap}")
            ret.append(hdr2)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    
    if radec:
        with utils.ignore_fits_warnings():
            wcs1 = WCS(hdr1, key='A')
            wcs2 = WCS(hdr2, key='A')
            wcs3 = WCS(hdr3, key='A')
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
    threshold = 100
    if (utils.to_timestamp(hdr2['DATE-END'])
            < utils.to_timestamp('psp_L3_wispr_20181201T000000_V3_2222.fits')
            and hdr2.get('detector', 1) == 2):
        # The outer-FOV images from the first encounter have extra thick
        # top/bottom margins, so allow that.
        threshold = 150
    if np.any(np.array(trim) > threshold):
        warnings.warn(f"File {hdr2['filename']} appears to have very large"
                      f" margins. Skipping...")
        ret = [img2]
        if return_mask:
            mask = np.zeros_like(img2, dtype=bool)
            if return_mask == 'also':
                ret.append(mask)
            else:
                ret[0] = mask
        if return_header:
            hdr2.add_history(
                    "Dust streak removal skipped; image appears to have very"
                    " large margins.")
            ret.append(hdr2)
        if len(ret) == 1:
            ret = ret[0]
        return ret

    mean_diffs, std_diffs = gen_diffs_distribution(
            img1_r, img3_r, trim, sliding_window_stride, window_width)
    
    # TODO: How can we better treat sigma values of zero (which occur when a
    # strip of constant values appears at the edge of the image, but with some
    # variations so the trimming steps above aren't activated, e.g.
    # psp_L3_wispr_20210809T023707_V1_1211
    std_diffs = np.where(std_diffs == 0, 1e-14, std_diffs)
    
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
    
    filtered = np.where(filter, img_fill, img2)
    
    ret = [filtered]
    if return_mask:
        if return_mask == 'also':
            ret.append(filter)
        else:
            ret[0] = filter
    if return_header:
        ret.append(hdr2)
    if len(ret) == 1:
        ret = ret[0]
    return ret


def gen_diffs_distribution(img1_r, img3_r, trim, sliding_window_stride, window_width):
    diffs = np.abs(img3_r - img1_r)
    return utils.sliding_window_stats(diffs, window_width, ['mean', 'std'],
            trim=trim, sliding_window_stride=sliding_window_stride)


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


def clean_file(data):
    """Utility function for clean_fits_files"""
    (fnames, input_dir, output_dir, plot_dir, save_masks,
            save_plots, overwrite, dust_streak_filter_kwargs) = data
    file_name = fnames[1]
    cleaned, mask, hdr = dust_streak_filter(
            *fnames, return_header=True, return_mask='also',
            **dust_streak_filter_kwargs)
    
    if output_dir is not None:
        fname = file_name.replace(input_dir, output_dir)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
    else:
        if save_masks:
            fname = file_name[:-5] + '_dust_streak_mask.fits'
        else:
            fname = file_name[:-5] + '_dust_streak_filtered.fits'
    
    if save_masks:
        hdr['bitpix'] = '8'
        del hdr['bunit']
        del hdr['blank']
    
    with utils.ignore_fits_warnings():
        if save_masks:
            output = mask.astype(np.int8)
        else:
            output = cleaned.astype(np.float32)
        fits.writeto(fname, output, hdr, overwrite=overwrite)
    
    if save_plots:
        if plot_dir is not None:
            fname = file_name.replace(input_dir, plot_dir)
            fname = fname[:-5] + '.jpg'
            os.makedirs(os.path.dirname(fname), exist_ok=True)
        else:
            fname = file_name[:-5] + '_dust_streak_removal.jpg'
        
        with utils.ignore_fits_warnings():
            data, header = fits.getdata(file_name, header=True)
        fig, axs = plt.subplots(1, 3, figsize=(20, 9), dpi=200)
        plt.suptitle(os.path.basename(file_name))
        plt.subplot(131)
        plot_utils.plot_WISPR((data, header), ax=plt.gca())
        plt.axis('off')
        plt.subplot(132)
        plot_utils.plot_WISPR((cleaned, header), ax=plt.gca())
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(mask, origin='lower')
        plt.axis('off')
        plt.subplots_adjust(wspace=.01, top=.98)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()


def clean_fits_files(input_dir, output_dir=None, plot_dir=None,
        save_masks=False, save_plots=False, overwrite=False,
        filters=None, dust_streak_filter_kwargs={}):
    input_dir = os.path.expanduser(input_dir)
    if not input_dir.endswith(os.sep):
        input_dir = input_dir + os.sep
    if output_dir is not None:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if not output_dir.endswith(os.sep):
            output_dir = output_dir + os.sep
    if plot_dir is not None:
        plot_dir = os.path.expanduser(plot_dir)
        os.makedirs(plot_dir, exist_ok=True)
        if not plot_dir.endswith(os.sep):
            plot_dir = plot_dir + os.sep
    else:
        plot_dir = output_dir
    
    ifiles, ofiles = utils.collect_files(input_dir, separate_detectors=True,
            filters=filters)
    
    with tqdm(total=len(ifiles) - 2 + len(ofiles) - 2) as pbar:
        for fits_files in ifiles, ofiles:
            triplets = zip(fits_files[:-2], fits_files[1:-1], fits_files[2:])
            iterable = zip(triplets, repeat(input_dir), repeat(output_dir),
                    repeat(plot_dir), repeat(save_masks), repeat(save_plots),
                    repeat(overwrite), repeat(dust_streak_filter_kwargs))
            with multiprocessing.Pool() as p:
                for x in p.imap_unordered(clean_file, iterable):
                    pbar.update(1)


def find_mask(masks_dir, fnames):
    """
    Finds a mask file for a given data file(s) by matching timestamps
    
    Arguments
    ---------
    masks_dir : str
        A directory containing mask files, with a directory structure parseable
        by `utils.collect_files`.
    fnames : str or list
        One or more data files.
    
    Returns
    -------
    found_masks : str or list
        The paths of the mask files for each data file, or ``None`` if a mask
        file was not found.
    """
    orig_fnames = fnames
    if isinstance(fnames, str):
        fnames = [fnames]
    masks = utils.collect_files(masks_dir, separate_detectors=False)
    masks = {utils.to_timestamp(f): f for f in masks}
    found_masks = []
    for f in fnames:
        try:
            found_masks.append(masks[utils.to_timestamp(f)])
        except KeyError:
            found_masks.append(None)
    
    if isinstance(orig_fnames, str):
        return found_masks[0]
    return found_masks

