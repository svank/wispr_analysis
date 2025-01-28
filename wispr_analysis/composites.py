from collections.abc import Iterable
import copy
import os

from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import reproject
import reproject.mosaicking
# Ensure Helioprojective frames are treated
import sunpy.coordinates

from . import utils


def find_bounds(hdr, wcs_target, trim=(0, 0, 0, 0), key=' ', wrap_aware=False,
        world_coord_bounds=None):
    """Finds the pixel bounds of a FITS header in an output WCS.
    
    The edges of the input image are transformed to the coordinate system of
    ``wcs_target``, and the extrema of these transformed coordinates are found.
    In other words, this finds the size of the output image that is required to
    bound the reprojected input image.
    
    Optionally, handles the case that the x axis of the output WCS is periodic
    and the input WCS straddles the wrap point. Two sets of bounds are
    returned, for the two halves of the input WCS.
    
    Parameters
    ----------
    hdr : astropy.io.fits.header.Header or str or tuple
        A FITS header describing an input image's size and coordinate system,
        or the path to a FITS file whose header will be loaded, or a tuple of a
        Header and a WCS.
    wcs_target : astropy.io.fits.header.Header or astropy.wcs.WCS
        A WCS object describing an output coordinate system.
    trim : tuple
        How many rows/columns to ignore from the input image. In order,
        (left, right, bottom, top).
    hdr_key : str
        The key argument passed to WCS, to select which of a header's
        coordinate systems to use.
    wrap_aware : boolean
        Whether to heuristically check for and handle the case that the image
        straddles the wrap point of the periodic x axis.
    world_coord_bounds : list
        Edge pixels of the image that fall outside these world coordinates are
        ignored. Must be a list of four values [xmin, xmax, ymin, ymax]. Any
        value can be None to not provide a bound.
    
    Returns
    -------
    bounds : list of tuples
        The bounding coordinates. In order, (left, right, bottom, top). One or
        two such tuples are returned, depending on whether the input WCS
        straddles the output's wrap point.
    """
    with utils.ignore_fits_warnings():
        if isinstance(hdr, str):
            with fits.open(hdr) as hdul:
                if hdul[0].data is None:
                    hdr = hdul[1].header
                else:
                    hdr = hdul[0].header
                wcs = WCS(hdr, hdul, key=key)
        elif isinstance(hdr, tuple):
            hdr, wcs = hdr
        elif isinstance(hdr, fits.Header):
            wcs = WCS(hdr, key=key)
        elif isinstance(hdr, WCS):
            wcs = hdr
        if not isinstance(wcs_target, WCS):
            wcs_target = WCS(wcs_target)
    left = 0 + trim[0]
    right = wcs.pixel_shape[0] - trim[1]
    bottom = 0 + trim[2]
    top = wcs.pixel_shape[1] - trim[3]
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
    if world_coord_bounds is not None:
        assert len(world_coord_bounds) == 4
        if world_coord_bounds[0] is None:
            world_coord_bounds[0] = -np.inf
        if world_coord_bounds[2] is None:
            world_coord_bounds[2] = -np.inf
        if world_coord_bounds[1] is None:
            world_coord_bounds[1] = np.inf
        if world_coord_bounds[3] is None:
            world_coord_bounds[3] = np.inf
        f = ( (world_coord_bounds[0] <= lon)
            * (lon <= world_coord_bounds[1])
            * (world_coord_bounds[2] <= lat)
            * (lat <= world_coord_bounds[3]))
        if not np.any(f):
            return None
        lon = lon[f]
        lat = lat[f]
    cx, cy = wcs_target.all_world2pix(lon, lat, 0)
    
    if not wrap_aware:
        return (int(np.floor(np.min(cx))),
                int(np.ceil(np.max(cx))),
                int(np.floor(np.min(cy))),
                int(np.ceil(np.max(cy))))
    
    ranges = []
    
    cx_hist, cx_hist_edges = np.histogram(cx, bins=11)
    if cx_hist[0] > 0 and cx_hist[-1] > 0 and cx_hist[len(cx_hist)//2] == 0:
        divider = cx_hist_edges[len(cx_hist)//2]
        for f in (cx < divider, cx > divider):
            cxf = cx[f]
            cyf = cy[f]
            xmin = cxf.min()
            xmax = cxf.max()
            ymin = cyf.min()
            ymax = cyf.max()
            ranges.append((xmin, xmax, ymin, ymax))
    else:
        ranges.append((cx.min(), cx.max(), cy.min(), cy.max()))
    for i, r in enumerate(ranges):
        r = (
            int(np.floor(r[0])),
            int(np.ceil(r[1])),
            int(np.floor(r[2])),
            int(np.ceil(r[3])))
        ranges[i] = r
    return ranges


def find_bounds_wrap_aware(*args, **kwargs):
    """Finds the pixel bounds of a FITS header in an output WCS, handling wraps.
    
    The edges of the input image are transformed to the coordinate system of
    ``wcs_target``, and the extrema of these transformed coordinates are found.
    In other words, this finds the size of the output image that is required to
    bound the reprojected input image.
    
    This version of the function handles the case that the x axis of the output
    WCS is periodic and the input WCS straddles the wrap point. Two sets of
    bounds are returned, for the two halves of the input WCS.
    
    Parameters
    ----------
    hdr : astropy.io.fits.header.Header or str or tuple
        A FITS header describing an input image's size and coordinate system,
        or the path to a FITS file whose header will be loaded, or a tuple of a
        Header and a WCS.
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
    bounds : list of tuples
        The bounding coordinates. In order, (left, right, bottom, top). One or
        two such tuples are returned, depending on whether the input WCS
        straddles the output's wrap point.
    """
    return find_bounds(*args, **kwargs, wrap_aware=True)


def find_collective_bounds(hdrs, wcs_target, trim=(0, 0, 0, 0), key=' '):
    """
    Finds the bounding coordinates for a set of input images.
    
    Calls `find_bounds` for each provided header, and finds the bounding box in
    the output coordinate system that will contain all of the input images.
    
    Parameters
    ----------
    hdrs : Iterable
        Either a list of Headers, or a list of lists of Headers. If the latter,
        ``trim`` can be a list of trim values, one for each of the lists of
        Headers. Instead of Headers, each instance can be the path to a FITS
        file.
    wcs_target : astropy.wcs.WCS
        A WCS object describing an output coordinate system.
    trim : tuple or list
        How many rows/columns to ignore from the input image. In order,
        (left, right, bottom, top). If ``hdrs`` is a list of lists of Headers,
        this can be (but does not have to be) be a list of tuples of trim
        values, one for each list of Headers.
    key : str
        The key argument passed to WCS, to select which of a header's
        coordinate systems to use.
    
    Returns
    -------
    bounds : tuple
        The bounding coordinates. In order, (left, right, bottom, top).
    """
    
    if isinstance(hdrs, (fits.header.Header, str, WCS)):
        hdrs = [[hdrs]]
    if isinstance(hdrs[0], (fits.header.Header, str, WCS)):
        hdrs = [hdrs]
    if not isinstance(trim[0], Iterable):
        trim = [trim] * len(hdrs)
    
    bounds = []
    for h, t in zip(hdrs, trim):
        bounds += [find_bounds(hdr, wcs_target, trim=t, key=key) for hdr in h]
    bounds = np.array(bounds).T
    return np.min(bounds[0]), np.max(bounds[1]), np.min(bounds[2]), np.max(bounds[3])


def gen_header(hdr_i, hdr_o, proj='ARC', level=False, key=' '):
    """
    Generates a header suitable for a WISPR composite field of view.
    
    This is mostly a port from pyssw.
    
    The output WCS will have a reference pixel near the center of the composite
    field of view, with other values set accordingly.
    
    Parameters
    ----------
    hdr_i, hdr_o : astropy.io.fits.Header
        The FITS headers for an inner-FOV and outer-FOV image
    proj : str
        The output projection to use
    level : boolean
        If ``True``, the output coordinate system is shifted so that the lines of
        zero degrees latitude and longitude appear straight. Intended for use
        with helioprojective frames only.
    key : str
        The coordinate system associated with this key value will be loaded
        from the headers.
    
    Returns
    -------
    wcsh : WCS
        The output WCS object
    naxis1, naxis2 : int
        The size of the output image
    """
    with utils.ignore_fits_warnings():
        if isinstance(hdr_i, str):
            with fits.open(hdr_i) as hdul:
                hdu = 1 if hdul[0].data is None else 0
                hdr_i = hdul[hdu].header
                wcs_i = WCS(hdr_i, hdul, key=key)
        else:
            wcs_i = WCS(hdr_i, key=key)
        if isinstance(hdr_o, str):
            with fits.open(hdr_o) as hdul:
                hdu = 1 if hdul[0].data is None else 0
                hdr_o = hdul[hdu].header
                wcs_o = WCS(hdr_o, hdul, key=key)
        else:
            wcs_o = WCS(hdr_o, key=key)
    naxis1, naxis2 = int(hdr_i['NAXIS1'] * 3), int(hdr_i['NAXIS2'] * 2)
    
    crval1 = (wcs_i.wcs.crval[0] + (wcs_o.wcs.crval[0] * 1.15)) / 2.
    crval2 = (wcs_i.wcs.crval[1] + wcs_o.wcs.crval[1]) / 2.
    ctype1 = wcs_i.wcs.ctype[0].split('-')[0]
    ctype2 = wcs_i.wcs.ctype[1].split('-')[0]
    ctype1 += '-' * (5 - len(ctype1)) + proj
    ctype2 += '-' * (5 - len(ctype2)) + proj

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
    
    wcsh.pixel_shape = naxis1, naxis2
    
    set_wcs_observer_details(wcsh, wcs_i, wcs_o)
    
    return wcsh, naxis1, naxis2


def set_wcs_observer_details(target_wcs, wcs1, wcs2):
    if wcs1 is None:
        wcs1 = wcs2
    if wcs2 is None:
        wcs2 = wcs1
    date = np.mean([wcs1.wcs.mjdavg, wcs2.wcs.mjdavg])
    dsun_obs = np.mean([wcs1.wcs.aux.dsun_obs, wcs2.wcs.aux.dsun_obs])
    hglt_obs = np.mean([wcs1.wcs.aux.hglt_obs, wcs2.wcs.aux.hglt_obs])
    # Angular mean for hgln
    hglnx = 0
    hglny = 0
    for wcs in (wcs1, wcs2):
        x, y = np.cos(wcs.wcs.aux.hgln_obs * np.pi/180), np.sin(wcs.wcs.aux.hgln_obs * np.pi/180)
        hglnx += x / 2
        hglny += y / 2
    hgln_obs = np.arctan2(hglny, hglnx) * 180 / np.pi
    
    target_wcs.wcs.mjdobs = date
    target_wcs.wcs.mjdavg = date
    target_wcs.wcs.dateobs = ''
    target_wcs.wcs.dateavg = ''
    target_wcs.wcs.aux.hgln_obs = hgln_obs
    target_wcs.wcs.aux.hglt_obs = hglt_obs
    target_wcs.wcs.aux.dsun_obs = dsun_obs
    with utils.ignore_fits_warnings():
        target_wcs.fix()
    return target_wcs


def gen_composite(fname_i, fname_o, proj='ARC', level=False, key=' ',
        bounds=None, wcsh=None, image_trim='auto', **kwargs):
    """
    Generates a composite image from two WISPR images (inner and outer)
    
    Parameters
    ----------
    fname_i, fname_o : str
        The file names of the inner and outer images
    proj : str
        The projection to use for the composite field of view
    level : boolean
        If ``True``, the output coordinate system is shifted so that the lines of
        zero degrees latitude and longitude appear straight. Intended for use
        with helioprojective frames only.
    key : str
        The coordinate system associated with this key value will be loaded
        from the headers.
    bounds : tuple
        The composite image is cropped at these pixel locations. The ordering
        is (left, right, bottom, top). If None, the bounds are autodetected to
        tightly bound the reprojected image data. If False, cropping is not
        performed.
    wcsh : WCS
        The coordinate system for the composite field of view. If not provided,
        the system is automatically generated.
    image_trim : Iterable
        The edges of the input images are trimmed. ``image_trim`` is a list of
        two lists, one for the inner and outer images, respectively. Each
        sub-list is a number of pixels to trim from the left, right, bottom,
        and top edges, respectively. Any value can be ``None`` to use the
        default for that value. Set to ``False`` to disable trimming. The
        default is sensible values for typical L3 images.
    kwargs
        Any remaining arguments are passed to ``reproject.reproject_adaptive``.
    
    Returns
    -------
    composite : np.ndarray
        The composite image array
    wcsh : WCS
        The coordinate system for the composite image
    """
    imgs = []
    wcses = []
    with utils.ignore_fits_warnings():
        for fname in fname_i, fname_o:
            if fname is not None:
                if isinstance(fname, tuple):
                    img, fname = fname
                else:
                    img = None
                fname = os.path.expanduser(fname)
                with fits.open(fname) as hdul:
                    hdu = 1 if hdul[0].data is None else 0
                    if img is None:
                        img = hdul[hdu].data
                    hdr = hdul[hdu].header
                    wcs = WCS(hdr, hdul, key=key)
                imgs.append(img)
                wcses.append(wcs)
    
    if wcsh is None:
        wcsh, _, _ = gen_header(
                fname_i, fname_o, proj=proj, level=level, key=key)
    else:
        wcsh = wcsh.deepcopy()
    
    image_trim_default = [[20, 25, 0, 0], [33, 40, 42, 39]]
    if not image_trim:
        image_trim = [[0] * 4] * 2
    if image_trim == 'auto':
        image_trim = image_trim_default
    
    for i in range(len(image_trim)):
        for j in range(len(image_trim[i])):
            if image_trim[i][j] is None:
                image_trim[i][j] = image_trim_default[i][j]
    
    for i, img in enumerate(imgs):
        trim = image_trim[i]
        wcses[i] = wcses[i][trim[2]:img.shape[0]-trim[3],
                            trim[0]:img.shape[1]-trim[1]]
        imgs[i] = imgs[i][trim[2]:img.shape[0]-trim[3],
                          trim[0]:img.shape[1]-trim[1]]
    
    if wcsh.wcs.ctype[0][:4] == 'HPLN':
        wcs_target = censor_wcs(wcsh)
        wcses = [censor_wcs(wcs) for wcs in wcses]
    else:
        wcs_target = wcsh
    
    if bounds is None:
        bounds = find_collective_bounds(wcses, wcs_target, key=key)
    elif not bounds:
        naxis1, naxis2 = wcsh.pixel_shape
        bounds = (0, naxis1, 0, naxis2)
    wcsh = wcsh[bounds[2]:bounds[3], bounds[0]:bounds[1]]
    wcs_target = wcs_target[bounds[2]:bounds[3], bounds[0]:bounds[1]]
    
    with utils.ignore_fits_warnings():
        reproj_args = dict(
            roundtrip_coords=False, boundary_mode='ignore_threshold',
            combine_function='first')
        reproj_args.update(kwargs)
        composite, footprint = reproject.mosaicking.reproject_and_coadd(
            list(zip(imgs, wcses)), wcs_target, wcs_target.array_shape,
            reproject_function=reproject.reproject_adaptive,
            **reproj_args)
        composite[footprint == 0] = np.nan
        return composite, wcsh


def censor_wcs(wcs, obstime=True, observer=True):
    """Removes observer details from a WCS
    
    When input images have slightly different viewpoints, Sunpy will say this
    is an invalid coordinate transformation. Here we censor information from the
    WCS to pacify Sunpy.
    """
    wcs = wcs.deepcopy()
    if observer:
        wcs.wcs.aux.hgln_obs = None
        wcs.wcs.aux.hglt_obs = None
        wcs.wcs.aux.dsun_obs = None
    if obstime:
        wcs.wcs.dateobs = ''
        wcs.wcs.dateavg = ''
        wcs.wcs.datebeg = ''
        wcs.wcs.dateend = ''
    return wcs
