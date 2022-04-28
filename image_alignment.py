from collections import defaultdict
from math import ceil, floor
from itertools import chain
import warnings

import multiprocessing
try:
    multiprocessing.set_start_method('fork')
except RuntimeError:
    pass

from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
import numpy as np
import scipy.ndimage
from tqdm.contrib.concurrent import process_map

from . import utils


def make_cutout(x, y, data, cutout_size, binning=2):
    cutout_size = int(round(cutout_size * 2 / binning))
    
    cutout_start_x = int(x) - cutout_size//2 + 1
    cutout_start_y = int(y) - cutout_size//2 + 1
    
    cutout = data[
            cutout_start_y:cutout_start_y + cutout_size,
            cutout_start_x:cutout_start_x + cutout_size]
    
    cutout = cutout - np.min(cutout)
    with np.errstate(invalid='raise'):
        cutout /= np.max(cutout)
    
    return cutout, cutout_start_x, cutout_start_y


def fit_star(x, y, data, all_stars_x, all_stars_y, cutout_size=8,
        ret_more=False, binning=2, start_at_max=True):
    try:
        cutout, cutout_start_x, cutout_start_y = make_cutout(
                x, y, data, cutout_size, binning=binning)
    except:
        err = ["Invalid values encountered"]
        if not ret_more:
            return np.nan, np.nan, err
    
    if all_stars_x is None:
        all_stars_x = np.array([x])
    if all_stars_y is None:
        all_stars_y = np.array([y])
    
    n_in_cutout = np.sum(
        (all_stars_x > cutout_start_x - .5)
        * (all_stars_x < cutout_start_x + cutout_size - .5)
        * (all_stars_y > cutout_start_y - .5)
        * (all_stars_y < cutout_start_y + cutout_size - .5))
    
    err = []
    if n_in_cutout > 1:
        err.append("Crowded frame")
        if not ret_more:
            return np.nan, np.nan, err
    
    fitter = fitting.LevMarLSQFitter()
    
    if start_at_max:
        i_max = np.argmax(cutout)
        y_start, x_start = np.unravel_index(i_max, cutout.shape)
    else:
        x_start = x - cutout_start_x
        y_start = y - cutout_start_y
    
    model = (models.Gaussian2D(
                amplitude=cutout.max(),
                x_mean=x_start,
                y_mean=y_start,
                x_stddev=2 / binning,
                y_stddev=2 / binning,
                bounds=dict(
                    amplitude=(0, np.inf),
                    x_mean=(-1, cutout_size),
                    y_mean=(-1, cutout_size),
                    ))
             + models.Planar2D(
                 intercept=np.median(cutout),
                 slope_x=0,
                 slope_y=0))
    
    yy, xx = np.mgrid[:cutout_size, :cutout_size]
    
    with warnings.catch_warnings():
        warnings.filterwarnings(action='error')
        try:
            fitted = fitter(model, xx, yy, cutout, maxiter=500)
        except AstropyUserWarning:
            err.append("No solution found")
            if ret_more:
                return None, cutout, err, cutout_start_x, cutout_start_y
            else:
                return np.nan, np.nan, err
    
    max_std = 2 / binning
    min_std = .1 / binning
    
    if fitted.amplitude_0 < 0.5 * (np.max(cutout) - fitted.intercept_1):
        err.append("No peak found")
    if fitted.x_stddev_0 > max_std or fitted.y_stddev_0 > max_std:
        err.append("Fit too wide")
    if fitted.x_stddev_0 < min_std or fitted.y_stddev_0 < min_std:
        err.append("Fit too narrow")
    if (not (0 < fitted.x_mean_0.value < cutout_size - 1)
            or not (0 < fitted.y_mean_0.value < cutout_size - 1)):
        err.append("Fitted peak too close to edge")
    
    if ret_more:
        return fitted, cutout, err, cutout_start_x, cutout_start_y
    return fitted.x_mean_0.value + cutout_start_x, fitted.y_mean_0.value + cutout_start_y, err


bad_pixel_map = np.load(
        '/Users/svankooten/Nextcloud/Parker Solar Probe/WISPR Perihelion Data/bad_pixel_mask.npz')['map']
good_pixel_map = ~scipy.ndimage.binary_dilation(bad_pixel_map, iterations=3)

DIM_CUTOFF = 8
BRIGHT_CUTOFF = 2

def prep_frame_for_star_finding(fname):
    with utils.ignore_fits_warnings():
        data, hdr = fits.getdata(fname, header=True)
        w = WCS(hdr, key='A')
    
    if data.shape not in ((1024, 960), (2048, 1920)):
        raise ValueError(
                f"Strange image shape {data.shape} in {fname}---skipping")
    
    if hdr['nbin1'] != hdr['nbin2']:
        raise ValueError(f"There's some weird binning going on in {fname}")
    binning = hdr['nbin1']

    trim = (40, 20, 20, 20)
    trim = [int(round(t * 2 / binning)) for t in trim]
    
    left = 0 + trim[0]
    right = hdr['naxis1'] - trim[1]
    bottom = 0 + trim[2]
    top = hdr['naxis2'] - trim[3]
    xs = np.concatenate((
        np.arange(left, right),
        np.full(top-bottom, right - 1),
        np.arange(right - 1, left - 1, -1),
        np.full(top-bottom, left)))
    ys = np.concatenate((
        np.full(right - left, bottom),
        np.arange(bottom, top),
        np.full(right - left, top - 1),
        np.arange(top - 1, bottom - 1, -1)))

    ra, dec = w.all_pix2world(xs, ys, 0)
    assert not np.any(np.isnan(ra)) and not np.any(np.isnan(dec))
    ra_min = np.min(ra)
    ra_max = np.max(ra)
    if ra_min < 100 and ra_max > 300:
        ras1 = ra[ra < 180]
        ras2 = ra[ra > 180]
        ra_segments = [(np.min(ras1), np.max(ras1)),
                       (np.min(ras2), np.max(ras2))]
    else:
        ra_segments = [(ra_min, ra_max)]
    dec_min = np.min(dec)
    dec_max = np.max(dec)

    stars_ra, stars_dec, stars_vmag = zip(*stars.stars_between(
        ra_segments, dec_min, dec_max))
    stars_vmag = np.array(stars_vmag)
    stars_ra, stars_dec = np.array(stars_ra), np.array(stars_dec)

    stars_x, stars_y = w.all_world2pix(stars_ra, stars_dec, 0)
    filter = ((left < stars_x) * (stars_x < right)
              * (bottom < stars_y) * (stars_y < top))
    stars_x = stars_x[filter]
    stars_y = stars_y[filter]
    stars_vmag = stars_vmag[filter]
    stars_ra = stars_ra[filter]
    stars_dec = stars_dec[filter]
    
    all_stars_x, all_stars_y = stars_x, stars_y
    
    filter = (stars_vmag < DIM_CUTOFF) * (stars_vmag > BRIGHT_CUTOFF)
    if hdr['detector'] == 1:
        filter *= good_pixel_map[np.round(stars_y * binning / 2).astype(int),
                                 np.round(stars_x * binning / 2).astype(int)]
    stars_x = stars_x[filter]
    stars_y = stars_y[filter]
    stars_vmag = stars_vmag[filter]
    stars_ra = stars_ra[filter]
    stars_dec = stars_dec[filter]
    
    return (stars_x, stars_y, stars_vmag, stars_ra, stars_dec,
            all_stars_x, all_stars_y, data, binning)


class StarBins:
    def __init__(self, RA_bin_size, dec_bin_size):
        
        self.n_RA_bins = int(ceil(360 / RA_bin_size))
        self.n_dec_bins = int(ceil(180 / dec_bin_size))
        
        self.bins = []
        for i in range(self.n_RA_bins):
            self.bins.append([[] for j in range(self.n_dec_bins)])
    
    def get_ra_bin(self, ra):
        ra = ra % 360
        ra_frac = ra / 360
        
        ra_bin = int(floor(ra_frac * self.n_RA_bins))
        
        return ra_bin
    
    def get_dec_bin(self, dec):
        dec_frac = (dec + 90) / 180
        
        dec_bin = int(floor(dec_frac * self.n_dec_bins))
        
        return dec_bin
    
    def get_bin(self, ra, dec):
        return self.get_ra_bin(ra), self.get_dec_bin(dec)
    
    def add_star(self, ra, dec, data):
        ra_bin, dec_bin = self.get_bin(ra, dec)
        
        self.bins[ra_bin][dec_bin].append(data)
    
    def get_stars(self, ra, dec):
        ra_bin, dec_bin = self.get_bin(ra, dec)
        
        return self.bins[ra_bin][dec_bin]
    
    def stars_between(self, ra_segments, dec_min, dec_max):
        ra_seqs = []
        for ra_seg in ra_segments:
            bin_start = self.get_ra_bin(ra_seg[0])
            bin_end = self.get_ra_bin(ra_seg[1])
            ra_seqs.append(range(bin_start, bin_end+1))
        ra_bins = chain(*ra_seqs)
        bin_start = self.get_dec_bin(dec_min)
        bin_end = self.get_dec_bin(dec_max)
        dec_bins = range(bin_start, bin_end+1)
        
        for ra_bin in ra_bins:
            for dec_bin in dec_bins:
                yield from self.bins[ra_bin][dec_bin]


stars = StarBins(3, 3)


star_dat = open("hipparchos_catalog.tsv").readlines()
for line in star_dat[43:-1]:
    try:
        id, RA, dec, Vmag = line.split(";")
    except ValueError:
        continue
    try:
        Vmag = float(Vmag)
    except ValueError:
        continue
    
    # Convert RA to floating-point degrees
    h, m, s = RA.split(" ")
    h = int(h) + int(m) / 60 + float(s) / 60 / 60
    RA = h / 24 * 360
    
    # Convert declination to floating-point degrees
    d, m, s = dec.split(" ")
    sign = 1 if d.startswith("+") else -1
    d = abs(int(d)) + int(m) / 60 + float(s) / 60 / 60
    dec = d * sign
    
    stars.add_star(RA, dec, (RA, dec, Vmag))


def find_stars_in_frame(fname):
    t = utils.to_timestamp(fname)
    try:
        (stars_x, stars_y, stars_vmag, stars_ra, stars_dec,
                all_stars_x, all_stars_y, data,
                binning) = prep_frame_for_star_finding(fname)
    except ValueError as e:
        print(e)
        return [], [], [], {}, {}

    good = []
    crowded_out = []
    bad = []
    codes = {}
    
    mapping = {}
    
    for x, y, ra, dec in zip(stars_x, stars_y, stars_ra, stars_dec):
        fx, fy, err = fit_star(
                x, y, data, all_stars_x, all_stars_y,
                ret_more=False, binning=binning)

        if len(err) == 0:
            good.append((fx, fy))
            mapping[(ra, dec)] = (fx, fy, t)
        elif 'Crowded frame' in err:
            crowded_out.append((fx, fy))
        else:
            bad.append((fx, fy))
        for e in err:
            codes[e] = codes.get(e, 0) + 1
    return good, bad, crowded_out, codes, mapping


def find_all_stars(ifiles, ret_all=False):
    # res = map(find_stars_in_frame, tqdm(ifiles))
    res = process_map(find_stars_in_frame, ifiles)

    good = []
    crowded_out = []
    bad = []
    codes = {}
    mapping = defaultdict(list)
    mapping_by_frame = {}

    for fname, (good_in_frame, bad_in_frame, crowded_in_frame,
            codes_in_frame, mapping_in_frame) in zip(ifiles, res):
        t = utils.to_timestamp(fname)
        good.extend(good_in_frame)
        bad.extend(bad_in_frame)
        crowded_out.extend(crowded_in_frame)
        for code, count in codes_in_frame.items():
            codes[code] = codes.get(code, 0) + count
        
        mapping_by_frame[t] = []
        for celest_coords, px_coords_and_time in mapping_in_frame.items():
            mapping[celest_coords].append(px_coords_and_time)
            mapping_by_frame[t].append(
                    (*celest_coords, *px_coords_and_time[:2]))

    # The `reshape` calls handle the case that the input list is empty
    good_x, good_y = np.array(good).T.reshape((2, -1))
    crowded_out_x, crowded_out_y = np.array(crowded_out).T.reshape((2, -1))
    bad_x, bad_y = np.array(bad).T.reshape((2, -1))
    
    if ret_all:
        return (mapping, mapping_by_frame, good_x, good_y, crowded_out_x,
                crowded_out_y, bad_x, bad_y, codes)
    # Change the defaultdict to just a dict
    return {k:v for k, v in mapping}, mapping_by_frame
