from itertools import chain
from math import ceil, floor
import os

from astropy.io import fits
from astropy.wcs import WCS
from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from . import plot_utils, utils


def extract_flux(img_sequence, ra, dec, aperture_r=5, gap=2, annulus_thickness=3, ret_all=False, extra_backgrounds=None):
    cutout_width = aperture_r + gap + annulus_thickness + 1
    if extra_backgrounds is None:
        extra_backgrounds = [None] * len(img_sequence)
    bgs, fluxes, cutouts, coords = [], [], [], []
    
    for img, extra_bg in zip(img_sequence, extra_backgrounds):
        if isinstance(img, str):
            with utils.ignore_fits_warnings(), fits.open(img) as hdul:
                img = hdul[0].data
                wcs = WCS(hdul[0].header, hdul, key='A')
                if extra_bg is not None:
                    img -= fits.getdata(extra_bg)
        else:
            img, wcs = img
        
        x, y = wcs.world_to_pixel_values(ra, dec)
        x, y = int(np.round(x)), int(np.round(y))
        if (y - cutout_width < 0 or x - cutout_width < 0
                or y + cutout_width + 1 > img.shape[0]
                or x + cutout_width + 1 > img.shape[1]):
            raise ValueError("Does not fit!")
        cutout = img[y - cutout_width : y + cutout_width + 1,
                     x - cutout_width : x + cutout_width + 1]
        x0 = cutout_width
        y0 = cutout_width
        xs = np.arange(cutout.shape[1]) - x0
        ys = np.arange(cutout.shape[0]) - y0
        xs, ys = np.meshgrid(xs, ys)
        r = np.sqrt(xs**2 + ys**2)
        
        bgs.append(np.nanmedian(
            cutout[(r > aperture_r + gap)
                   * (r <= aperture_r + gap + annulus_thickness)]))
        
        center = (r <= aperture_r)
        central_flux = np.sum(cutout[center])
        fluxes.append(central_flux - bgs[-1] * np.sum(center))
        cutouts.append(cutout)
        coords.append((x, y))
    
    fluxes = np.array(fluxes)
    bgs = np.array(bgs)
    cutouts = np.array(cutouts)
    coords = np.array(coords)
    if ret_all:
        return fluxes, bgs, cutouts, coords
    return fluxes


def illustrate_flux(img_sequence, ra, dec, aperture_r=2, gap=1, annulus_thickness=2):
    def f(i, aperture_r=aperture_r, gap=gap,
            annulus_thickness=annulus_thickness):
        flux, bg, cutout, coords = extract_flux(
                img_sequence[i:i+1], ra, dec, aperture_r, gap,
                annulus_thickness, ret_all=True)
        x, y = coords[0]
        img = img_sequence[i]
        if not isinstance(img, str):
            img = img[0]
        plot_utils.plot_WISPR(img)
        plt.ylim(y-10, y+10)
        plt.xlim(x-10, x+10)
        plt.scatter([x], [y])
        
        thetas = np.linspace(0, 2*np.pi)
        r = aperture_r
        plt.plot(x + r * np.cos(thetas), y + r * np.sin(thetas))
        r = aperture_r + gap
        plt.plot(x + r * np.cos(thetas), y + r * np.sin(thetas))
        r = aperture_r + gap + annulus_thickness
        plt.plot(x + r * np.cos(thetas), y + r * np.sin(thetas))
        plt.title(f"Flux: {flux[0]}")
    return interactive(
            f,
            i=(0, len(img_sequence)-1),
            aperture_r=(1, 10),
            gap=(0, 10),
            annulus_thickness=(1, 10),
    )


class StarBins:
    """
    Class to allow efficient access to stars within an RA/Dec range
    
    Works by dividing RA/Dec space into bins and placing stars within a bin.
    Querying an RA/Dec range can then access only the relevant bins, cutting
    down the search space.
    """
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
        """
        Add a star to the bins
        
        Parameters
        ----------
        ra, dec : float
            The coordinates of the star
        data
            Any arbitrary object to be stored for this star
        """
        ra_bin, dec_bin = self.get_bin(ra, dec)
        
        self.bins[ra_bin][dec_bin].append(data)
    
    def get_stars(self, ra, dec):
        ra_bin, dec_bin = self.get_bin(ra, dec)
        
        return self.bins[ra_bin][dec_bin]
    
    def stars_between(self, ra_segments, dec_min, dec_max):
        """
        Generator to access stars within an RA/Dec range
        
        As a generator, it can be used like:
        
        for star_data in star_bins.stars_between(...):
            ...
        
        Parameters
        ----------
        ra_segments : list of tuples
            The segments in right ascension to access. To handle wrapping,
            multiple segments are supported. Each tuple in this list consists
            of (ra_start, ra_stop).
        dec_min, dec_max : float
            The declination range to search.
        """
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


def load_stars():
    stars = StarBins(3, 3)
    
    catalog_path = os.path.join(
            os.path.dirname(__file__), "data", "hipparchos_catalog.tsv")
    star_dat = open(catalog_path).readlines()
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
    return stars
stars = load_stars()


try:
    bad_pixel_map = np.load(
            '/home/svankooten/psp/WISPR/bad_pixel_mask.npz')['map']
except:
    bad_pixel_map = np.zeros((1024, 960), dtype=bool)
    import warnings
    warnings.warn('Did not find bad pixel map')

good_pixel_map = ~scipy.ndimage.binary_dilation(bad_pixel_map, iterations=3)


DIM_CUTOFF = 8
BRIGHT_CUTOFF = 2


def find_expected_stars_in_frame(fname, dim_cutoff=DIM_CUTOFF,
        bright_cutoff=BRIGHT_CUTOFF, trim=(0,0,0,0)):
    if isinstance(trim, int):
        trim = [trim] * 4
    
    if isinstance(fname, str):
        with utils.ignore_fits_warnings(), fits.open(fname) as hdul:
            hdr = hdul[0].header
            w = WCS(hdr, hdul, key='A')
    else:
        hdr, w = fname
    
    if hdr['nbin1'] != hdr['nbin2']:
        raise ValueError(f"There's some weird binning going on in {fname}")
    binning = hdr['nbin1']
    bin_factor = 2 / binning
    
    trim = [int(round(t * bin_factor)) for t in trim]
    
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
    
    filter = (stars_vmag < dim_cutoff) * (stars_vmag > bright_cutoff)
    if hdr['detector'] == 1:
        filter *= good_pixel_map[np.round(stars_y / bin_factor).astype(int),
                                 np.round(stars_x / bin_factor).astype(int)]
    stars_x = stars_x[filter]
    stars_y = stars_y[filter]
    stars_vmag = stars_vmag[filter]
    stars_ra = stars_ra[filter]
    stars_dec = stars_dec[filter]
    
    return (stars_x, stars_y, stars_vmag, stars_ra, stars_dec,
            all_stars_x, all_stars_y)
