from collections.abc import Iterable
import copy
import warnings

from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import reproject

from . import utils


def find_bounds(hdr, wcs_target, trim=(0, 0, 0, 0), key=' '):
    """Finds the pixel bounds of a FITS header in an output WCS.
    
    The edges of the input image are transformed to the coordinate system of
    `wcs_target`, and the extrema of these transformed coordinates are found.
    In other words, this finds the size of the output image that is required to
    bound the reprojected input image.
    
    Parameters
    ----------
    hdr : astropy.io.fits.header.Header or str
        A FITS header describing an input image's size and coordinate system,
        or the path to a FITS file whose header will be loaded.
    wcs_target : astropy.io.fits.header.Header or astropy.wcs.WCS
        A WCS object describing an output coordinate system.
    trim : tuple
        How many rows/columns to ignore from the input image. In order,
        (left, right, bottom, top).
    hdr_key : str
        The key argument passed to WCS, to select which of a header's
        coordinate systems to use.
    
    Returns
    -------
    bounds : tuple
        The bounding coordinates. In order, (left, right, bottom, top).
    """
    with utils.ignore_fits_warnings():
        if isinstance(hdr, str):
            with fits.open(hdr) as hdul:
                hdr = hdul[0].header
                wcs = WCS(hdr, hdul, key=key)
        else:
            wcs = WCS(hdr, key=key)
        if not isinstance(wcs_target, WCS):
            wcs_target = WCS(wcs_target)
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
    
    lon, lat = wcs.all_pix2world(xs, ys, 0)
    assert not np.any(np.isnan(lon)) and not np.any(np.isnan(lat))
    cx, cy = wcs_target.all_world2pix(lon, lat, 0)
    
    return (int(np.floor(np.min(cx))),
            int(np.ceil(np.max(cx))),
            int(np.floor(np.min(cy))),
            int(np.ceil(np.max(cy))))


def find_collective_bounds(hdrs, wcs_target, trim=(0, 0, 0, 0), key=' '):
    """
    Finds the bounding coordinates for a set of input images.
    
    Calls `find_bounds` for each provided header, and finds the bounding box in
    the output coordinate system that will contain all of the input images.
    
    Paramters
    ---------
    hdrs : Iterable
        Either a list of Headers, or a list of lists of Headers. If the latter,
        `trim` can be a list of trim values, one for each of the lists of
        Headers. Instead of Headers, each instance can be the path to a FITS
        file.
    wcs_target : astropy.wcs.WCS
        A WCS object describing an output coordinate system.
    trim : tuple or list
        How many rows/columns to ignore from the input image. In order,
        (left, right, bottom, top). If `hdrs` is a list of lists of Headers,
        this can be (but does not have to be) be a list of tuples of trim values,
        one for each list of Headers.
    hdr_key : str
        The key argument passed to WCS, to select which of a header's
        coordinate systems to use.
    
    Returns
    -------
    bounds : tuple
        The bounding coordinates. In order, (left, right, bottom, top).
    """
    
    if isinstance(hdrs, (fits.header.Header, str)):
        hdrs = [[hdrs]]
    if isinstance(hdrs[0], (fits.header.Header, str)):
        hdrs = [hdrs]
    if not isinstance(trim[0], Iterable):
        trim = [trim] * len(hdrs)
    
    bounds = []
    for h, t in zip(hdrs, trim):
        bounds += [find_bounds(hdr, wcs_target, trim=t, key=key) for hdr in h]
    bounds = np.array(bounds).T
    return np.min(bounds[0]), np.max(bounds[1]), np.min(bounds[2]), np.max(bounds[3])


def gen_header(hdr_i, hdr_o, proj='ARC', level=False, key=' '):
    naxis1, naxis2 = int(hdr_i['NAXIS1'] * 3), int(hdr_i['NAXIS2'] * 2)
    with warnings.catch_warnings():
        warnings.filterwarnings(
                action='ignore', message=".*'datfix' made the change.*")
        wcs_i = WCS(hdr_i, key=key)
        wcs_o = WCS(hdr_o, key=key)

    crval1 = (wcs_i.wcs.crval[0] + (wcs_o.wcs.crval[0] * 1.15)) / 2.
    crval2 = (wcs_i.wcs.crval[1] + wcs_o.wcs.crval[1]) / 2.
    ctype1, ctype2 = (f"{wcs_i.wcs.ctype[0].split('-')[0]}-{proj}",
            f"{wcs_i.wcs.ctype[1].split('-')[0]}-{proj}")

    crpix1 = naxis1 / 2
    crpix2 = naxis2 / 2
    
    if level:
        # Make the projection relative to the Sun (make the zero-longitude
        # line straight)
        crval1 = 0
        crpix1 = 10

        # Make the zero-latitude line straight
        crval2 = 0
        crpix2 = naxis2 // 2
    
    wcsh = WCS(naxis=2)
    wcsh.wcs.crpix = crpix1, crpix2
    wcsh.wcs.cdelt = wcs_i.wcs.cdelt
    wcsh.wcs.crval = crval1, crval2
    wcsh.wcs.ctype = ctype1, ctype2
    wcsh.wcs.cunit = wcs_i.wcs.cunit
    wcsh.wcs.pc = np.array([[1., 0.], [0., 1.]])
    
    return wcsh, naxis1, naxis2


def gen_composite(fname_i, fname_o, proj='ARC', level=False,
        return_both=False, bounds=None, wcsh=None, naxis1=None, naxis2=None,
        blank_i=False, blank_o=False, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings(
                action='ignore', message=".*'BLANK' keyword.*")
        if isinstance(fname_i, tuple):
            img_i, hdr_i = fname_i
            hdr_i = hdr_i.copy()
        else:
            img_i, hdr_i = fits.getdata(fname_i, header=True)
        if isinstance(fname_o, tuple):
            img_o, hdr_o = fname_o
            hdr_o = hdr_o.copy()
        else:
            img_o, hdr_o = fits.getdata(fname_o, header=True)
    
    if wcsh is None:
        wcsh, naxis1, naxis2 = gen_header(
                hdr_i, hdr_o, proj=proj, level=level)
    else:
        wcsh = wcsh.deepcopy()
    
    # Pixels to trim from the image edges.
    # (outer, inner), then (left, right, bottom, top)
    xy_bounds = [[20, 25, 0, 0], [33, 40, 42, 39]]
    
    imgs = []
    for img, hdr, trim in zip((img_o, img_i), (hdr_o, hdr_i), xy_bounds):
        imgs.append(img[trim[2]:img.shape[0]-trim[3],
                        trim[0]:img.shape[1]-trim[1]])
        hdr['NAXIS1'] -= trim[0] + trim[1]
        hdr['NAXIS2'] -= trim[2] + trim[3]
        hdr['CRPIX1'] -= trim[0]
        hdr['CRPIX2'] -= trim[2]
    img_o, img_i = imgs
    
    if bounds is None:
        img_i_bounds = find_bounds(hdr_i, wcsh)
        img_o_bounds = find_bounds(hdr_o, wcsh)
        bounds = (min(img_i_bounds[0], img_o_bounds[0]),
                  max(img_i_bounds[1], img_o_bounds[1]),
                  min(img_i_bounds[2], img_o_bounds[2]),
                  max(img_i_bounds[3], img_o_bounds[3]))
    elif not bounds:
        bounds = (0, naxis1, 0, naxis2)
    
    naxis1 += bounds[1] - naxis1
    naxis2 += bounds[3] - naxis2
    naxis1 -= bounds[0]
    naxis2 -= bounds[2]
    crpix1, crpix2 = wcsh.wcs.crpix
    wcsh.wcs.crpix = crpix1 - bounds[0], crpix2 - bounds[2]
    
    for hdr in hdr_i, hdr_o:
        for k in 'date-obs', 'rsun_ref', 'dsun_obs', 'crln_obs', 'crlt_obs':
            del hdr[k]
    
    with warnings.catch_warnings():
        warnings.filterwarnings(
                action='ignore', message=".*'datfix' made the change.*")
        if blank_i:
            o1 = np.full((naxis2, naxis1), np.nan)
        else:
            o1 = reproject.reproject_adaptive(
                    (img_i, hdr_i), wcsh, (naxis2, naxis1),
                    roundtrip_coords=False, return_footprint=False,
                    boundary_mode='ignore_threshold', **kwargs)
        if blank_o:
            o2 = np.full((naxis2, naxis1), np.nan)
        else:
            o2 = reproject.reproject_adaptive(
                    (img_o, hdr_o), wcsh, (naxis2, naxis1),
                    roundtrip_coords=False, return_footprint=False,
                    boundary_mode='ignore_threshold', **kwargs)
    if return_both:
        return o1, o2, wcsh
    replace = np.isnan(o1)
    o1[replace] = o2[replace]
    return o1, wcsh


def plot_map(map, wcsh, figsize=(15, 10)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=wcsh)
    ax.set_facecolor('black')
    lon, lat = ax.coords
    lat.set_ticks(np.arange(-90, 90, 10) * u.degree)
    lon.set_ticks(np.arange(-180, 180, 15) * u.degree)
    lat.set_major_formatter('dd')
    lon.set_major_formatter('dd')
    ax.coords.grid(color='white')
    ax.set_xlabel("Heliprojective Longitude")
    ax.set_ylabel("Heliprojective Latitude")
    cmap = copy.copy(plt.cm.Greys_r)
    cmap.set_bad('black')
    vmin, vmax = np.nanpercentile(map, [5, 95])
    ax.imshow(map, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    fig.tight_layout()
    return fig
