from itertools import repeat
import os

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import reproject
from tqdm.contrib.concurrent import process_map

from . import utils


def padded_slice(img, sl):
    """
    Returns a chunk of an image, padded with zeros if outside the image bounds
    
    Arguments
    ---------
    img : ``np.ndarray``
        The image to take a chunk out of
    sl : Iterable of ints
        The region to extract, given (start, stop, start, stop) for dimensions
        0 and 1, respectively.
    
    Returns
    -------
    chunk : ``np.ndarray``
        The extracted chunk, with zeros filling any edges that lie outside the
        bounds of ``img``.
    """
    output_shape = np.asarray(img.shape)
    output_shape[0] = sl[1] - sl[0]
    output_shape[1] = sl[3] - sl[2]
    src = [max(sl[0], 0),
           min(sl[1], img.shape[0]),
           max(sl[2], 0),
           min(sl[3], img.shape[1])]
    dst = [src[0] - sl[0], src[1] - sl[0],
           src[2] - sl[2], src[3] - sl[2]]
    output = np.zeros(output_shape, dtype=img.dtype)
    output[dst[0]:dst[1],dst[2]:dst[3]] = img[src[0]:src[1],src[2]:src[3]]
    return output


def _extract_from_frame(data):
    fname, coord, r, target_wcs, wcs_key, trim_amount, clip_as_r = data
    t = utils.to_timestamp(fname)
    with utils.ignore_fits_warnings(), fits.open(fname) as hdul:
        header = hdul[0].header
        wcs = WCS(header, hdul, key=wcs_key)
    
    orig_shape = (header['naxis2'], header['naxis1'])
    crpix = wcs.wcs.crpix
    wcs.wcs.crpix = crpix[0] - trim_amount, crpix[1] - trim_amount

    # Check if the coordinate is in-frame for this image
    xw, yw = wcs.all_world2pix(*coord, 0)
    if np.isnan(xw) or np.isnan(yw):
        return None
    xw, yw = int(np.round(xw)), int(np.round(yw))
    if not (
            clip_as_r <= xw < header['naxis1'] - trim_amount - clip_as_r
            and clip_as_r <= yw < header['naxis2'] - trim_amount - clip_as_r):
        return None
    
    # It is, so extract the region around the coordinate
    src_coords = xw, yw
    with utils.ignore_fits_warnings():
        data = fits.getdata(fname)
        data = data[trim_amount:-trim_amount, trim_amount:-trim_amount]
        if target_wcs is not None:
            # Reproject each cutout to the same frame
            if not isinstance(target_wcs, WCS):
                target_wcs = WCS(target_wcs, key=wcs_key)
            data = reproject.reproject_adaptive(
                    (data, wcs), target_wcs, orig_shape, return_footprint=False,
                    roundtrip_coords=False, conserve_flux=True)
            xw, yw = target_wcs.all_world2pix(*coord, 0)
            if np.isnan(xw) or np.isnan(yw):
                return None
            xw, yw = int(np.round(xw)), int(np.round(yw))
            if not (
                    clip_as_r <= xw < orig_shape[1] - clip_as_r
                    and clip_as_r <= yw < orig_shape[0] - clip_as_r):
                return None
            if np.any(np.isnan(data[
                    yw-clip_as_r:yw+clip_as_r+1, xw-clip_as_r:xw+clip_as_r+1])):
                return None
    chunk = padded_slice(data, (yw-r, yw+r+1, xw-r, xw+r+1))
    return chunk, os.path.basename(fname), t, src_coords


def collect_region_all_frames(coord, data_dir, r=5, target_wcs=None,
        wcs_key=' ', detector=None, trim_amount=40, clip_as_r=None):
    """
    Extracts the region around a celestial coordinate from each image it's in.
    
    Arguments
    ---------
    coord : ``tuple``
        The coordinate around which data should be extracted, as (RA, Dec). Can
        be a tuple of numbers (in degrees), or astropy Coordinate objects.
    data_dir : ``str``
        The directory containing the FITS files from which data should be
        extracted. Will be searched recursively by `utils.collect_files`.
    r : ``int``
        The size of the region to extract. The total size is (2*r + 1).
    target_wcs : WCS or FITS header
        If given, each image is reprojected to this frame before the region is
        extracted.
    wcs_key : str
        If parsing a FITS header as `target_wcs`, use this ``wcs_key``.
    detector : 'i' or 'o'
        If specified, only use files from this WISPR detector.
    trim_amount : int
        Trim this many pixels from the edge of each image (before checking if
        the coordinate lies in this image).
    clip_as_r : int
        If given, for the purposes of determining whether the extraction region
        lies within the bounds of an image, actually use this value rather than
        `r` itself to determine the extraction region size.
    
    Returns
    -------
    data : ``ndarray``
        The collected slices, of shape (nt, ny nx).
    fnames : tuple
        The filename corresponding to each extracted slice.
    ts : tuple
        The timestamp corresponding to each extracted slice.
    px_coords : tuple
        The pixel coordinates corresponding to the celestial coordinate in each
        extracted slice.
    """
    try:
        coord = coord.ra.to(u.deg).value, coord.dec.to(u.deg).value
    except AttributeError:
        pass
    if clip_as_r is None:
        clip_as_r = r
    if detector is None:
        files = utils.collect_files(data_dir, separate_detectors=False)
    elif detector[0].lower() == 'i':
        files, _ = utils.collect_files(data_dir, separate_detectors=True)
    elif detector[0].lower() == 'o':
        _, files = utils.collect_files(data_dir, separate_detectors=True)
    else:
        raise ValueError("Invalid value for 'detector'")
    iterable = zip(
            files, repeat(coord), repeat(r), repeat(target_wcs),
            repeat(wcs_key), repeat(trim_amount), repeat(clip_as_r))
    # data = [x for x in map(_extract_from_frame, iterable) if x is not None]
    data = [x for x in process_map(
        _extract_from_frame, iterable, total=len(files), chunksize=3)
        if x is not None]
    data, fnames, ts, px_coords = zip(*data)
    data = np.stack(data)
    return data, fnames, ts, px_coords
