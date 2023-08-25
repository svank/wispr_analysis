from collections.abc import Iterable
import copy
import os

from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import reproject

from . import utils


def find_bounds(hdr, wcs_target, trim=(0, 0, 0, 0), key=' ', wrap_aware=False,
        world_coord_bounds=None):
    """Finds the pixel bounds of a FITS header in an output WCS.
    
    The edges of the input image are transformed to the coordinate system of
    ``wcs_target``, and the extrema of these transformed coordinates are found.
    In other words, this finds the size of the output image that is required to
    bound the reprojected input image.
    
    Optionall, handles the case that the x axis of the output WCS is periodic
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
    """
    Generates a header suitable for a WISPR compositte field of view.
    
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
        If `True`, the output coordinate system is shifted so that the lines of
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
    
    return wcsh, naxis1, naxis2


def gen_composite(fname_i, fname_o, proj='ARC', level=False, key=' ',
        return_both=False, bounds=None, wcsh=None, naxis1=None, naxis2=None,
        blank_i=False, blank_o=False,
        image_trim='auto', **kwargs):
    """
    Generates a composite image from two WISPR images (inner and outer)
    
    Parameters
    ----------
    fname_i, fname_o : str
        The file names of the inner and outer images
    proj : str
        The projection to use for the composite field of view
    level : boolean
        If `True`, the output coordinate system is shifted so that the lines of
        zero degrees latitude and longitude appear straight. Intended for use
        with helioprojective frames only.
    key : str
        The coordinate system associated with this key value will be loaded
        from the headers.
    return_both : booean
        If ``True``, the two individual images are reprojected into the common
        field of view but are not combined. Instead, two separate, reprojected
        images are returned.
    bounds : tuple
        The composite image is cropped at these pixel locations. The ordering
        is (left, right, bottom, top). If None, the bounds are autodetected to
        tightly bound the reprojected image data. If False, cropping is not
        performed.
    wcsh : WCS
        The coordinate system for the composite field of view. If not provided,
        the system is automatically generated.
    naxis1, naxis2 : int
        If ``wcsh`` is provided, the size of the output image must also be
        provided with these two values.
    blank_i, blank_o : boolean
        Set either flag to ``True`` to not render the inner or outer image.
        Files must still be provided for any images not drawn so that the
        proper size of the output image can be determined. This flag is inteded
        for rendering frames in an animation, when some frames should still
        render a full field of view but one imager is temporarily not being
        shown.
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
    img_i, img_o = None, None
    with utils.ignore_fits_warnings():
        if isinstance(fname_i, tuple):
            img_i, fname_i = fname_i
        fname_i = os.path.expanduser(fname_i)
        with fits.open(fname_i) as hdul:
            hdu = 1 if hdul[0].data is None else 0
            if img_i is None:
                img_i = hdul[hdu].data
            hdr_i = hdul[hdu].header
            # The input images have slightly different viewpoints. If
            # sunpy.coordinates has been imported, astropy will know enough
            # about HPC to say this is an invalid coordinate transformation.
            # Here we censor information from the header to pacify Sunpy.
            keys = [k for k in hdr_i if k.endswith("_OBS")
                    or k.startswith('DATE-') or k.startswith('MJD-')]
            for k in keys:
                hdr_i.remove(k)
            wcs_i = WCS(hdr_i, hdul, key=key)
        
        if isinstance(fname_o, tuple):
            img_o, fname_o = fname_o
        fname_o = os.path.expanduser(fname_o)
        with fits.open(fname_o) as hdul:
            hdu = 1 if hdul[0].data is None else 0
            if img_o is None:
                img_o = hdul[hdu].data
            hdr_o = hdul[hdu].header
            keys = [k for k in hdr_o if k.endswith("_OBS")
                    or k.startswith('DATE-') or k.startswith('MJD-')]
            for k in keys:
                hdr_o.remove(k)
            wcs_o = WCS(hdr_o, hdul, key=key)
    
    if wcsh is None:
        wcsh, naxis1, naxis2 = gen_header(
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
    
    imgs = []
    wcses = []
    for img, hdr, wcs, trim in zip(
            (img_o, img_i), (hdr_o, hdr_i), (wcs_o, wcs_i), image_trim[::-1]):
        wcses.append(wcs[trim[2]:img.shape[0]-trim[3],
                         trim[0]:img.shape[1]-trim[1]])
        imgs.append(img[trim[2]:img.shape[0]-trim[3],
                        trim[0]:img.shape[1]-trim[1]])
        hdr['NAXIS1'] -= trim[0] + trim[1]
        hdr['NAXIS2'] -= trim[2] + trim[3]
        hdr['CRPIX1'] -= trim[0]
        hdr['CRPIX2'] -= trim[2]
        hdr['CRPIX1A'] -= trim[0]
        hdr['CRPIX2A'] -= trim[2]
    img_o, img_i = imgs
    wcs_o, wcs_i = wcses
    
    if bounds is None:
        img_i_bounds = find_bounds((hdr_i, wcs_i), wcsh, key=key)
        img_o_bounds = find_bounds((hdr_o, wcs_o), wcsh, key=key)
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
    
    with utils.ignore_fits_warnings():
        if blank_i:
            o1 = np.full((naxis2, naxis1), np.nan)
        else:
            o1 = reproject.reproject_adaptive(
                    (img_i, wcs_i), wcsh, (naxis2, naxis1),
                    roundtrip_coords=False, return_footprint=False,
                    boundary_mode='ignore_threshold', **kwargs)
        if blank_o:
            o2 = np.full((naxis2, naxis1), np.nan)
        else:
            o2 = reproject.reproject_adaptive(
                    (img_o, wcs_o), wcsh, (naxis2, naxis1),
                    roundtrip_coords=False, return_footprint=False,
                    boundary_mode='ignore_threshold', **kwargs)
    if return_both:
        return o1, o2, wcsh
    composite = np.where(np.isnan(o1), o2, o1)
    return composite, wcsh


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
