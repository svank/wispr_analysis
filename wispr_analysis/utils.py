from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from math import ceil
import os
import warnings

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import scipy.signal


def to_timestamp(datestring, as_datetime=False):
    if isinstance(datestring, Iterable) and not isinstance(datestring, str):
        return [to_timestamp(x, as_datetime=as_datetime) for x in datestring]
    # Check if we got a filename
    if datestring.endswith('.fits'):
        # Grab just the filename if it's a full path
        if '/' in datestring:
            datestring = datestring.split('/')[-1]
        # Extract the timestamp part of the standard WISPR filename
        datestring = datestring.split('_')[3]
    try:
        dt = datetime.strptime(
                datestring, "%Y%m%dT%H%M%S")
    except ValueError:
        dt = datetime.strptime(
                datestring, "%Y-%m-%dT%H:%M:%S.%f")
    dt = dt.replace(tzinfo=timezone.utc)
    if as_datetime:
        return dt
    return dt.timestamp()


def get_PSP_path(data_dir):
    data_dir = os.path.expanduser(data_dir)
    files = collect_files(data_dir, separate_detectors=False, order='date-avg',
            include_headers=True)
    return get_PSP_path_from_headers([f[1] for f in files])


def get_PSP_path_from_headers(headers):
    times = []
    positions = []
    vs = []
    for header in headers:
        times.append(to_timestamp(header['DATE-AVG']))
        positions.append((
            header['HCIx_OBS'],
            header['HCIy_OBS'],
            header['HCIz_OBS']))
        vs.append((
            header['HCIx_VOB'],
            header['HCIy_VOB'],
            header['HCIz_VOB']))
    return np.array(times), np.array(positions), np.array(vs)


def collect_files(top_level_dir, separate_detectors=True, order=None,
        include_sortkey=False, include_headers=False, between=(None, None),
        filters=None):
    """Given a directory, returns a sorted list of all WISPR FITS files.
    
    Subdirectories are searched, so this lists all WISPR files for an
    encounter when they are in separate subdirectories by date. By default,
    returns two lists, one for each detector. Set ``separate_detectors`` to
    False to return only a single, sorted list. Returned items are full paths
    relative to the given directory.
    
    Invalid files are omitted. The expected structure is that subdirectories
    have names starting with "20" (i.e. for the year 20XX), and file names
    should be in the standard formated provided by the WISPR team.
    
    Files are ordered by the value of the FITS header key provided as the
    ``order`` argument. Set ``order`` to None to sort by filename instead
    (which is implicitly DATE-BEG, as that is contained in the filenames).
    
    Parameters
    ----------
    top_level_dir : str
        The directory containing the FITS files or directories of FITS files
    separate_detectors : boolean
        If ``True``, two lists are returned---one for the inner imager, and one
        for the outer. If ``False``, a single list of all images is returned.
    order : str or None
        A FITS header keyword, the value of which is used as the sort key for
        sorting the files. If ``None``, the timestamp embedded in the filenames
        is used, which is faster than loading the header from each file.
    include_sortkey : boolean
        Whether to return the value of the sort key along with each file name.
    include_headers : boolean
        Whether to return the FITS header of each file along with the file
        names.
    between : tuple
        Minimum and maximum allowable values of the sort key. Files whose
        sort-key value falls outside this range are excluded. Either value can
        be ``None``, to apply no minimum or maximum value.
    filters : list or tuple
        A list of additional filters to apply. Each element should be a tuple
        containing a FITS header keyword, a minimum value, and a maximum value.
        Either of the latter two values can be ``None`` to apply to minimum or
        maximum. If only one filter is provided, it can be passed directly
        rather than as a one-element list. If the minimum or maximum value is
        provided as a number, the values extracted from the FITS headers will
        be converted to the same type.
        
    Returns
    -------
    A list or two lists, depending on ``separate_detectors``. Each element is a
    filename, or a tuple of up to ``(sort_key, filename, header)`` depending on
    the values of ``include_sortkey`` and ``include_headers``.
    """
    i_files = []
    o_files = []
    subdirs = []
    if filters is None:
        filters = []
    if between is None:
        between = (None, None)
    parse_key_as_timestamp = False
    if order is None or order.startswith("DATE"):
        parse_key_as_timestamp = True
        between = (
            to_timestamp(between[0]) if between[0] is not None else None,
            to_timestamp(between[1]) if between[1] is not None else None,
        )
    if len(filters) == 3 and isinstance(filters[0], str):
        filters = [filters]
    
    top_level_dir = os.path.expanduser(top_level_dir)
    # Find all valid subdirectories.
    for fname in os.listdir(top_level_dir):
        path = f"{top_level_dir}/{fname}"
        if os.path.isdir(path) and fname.startswith('20'):
            subdirs.append(path)
    if len(subdirs) == 0:
        subdirs.append(top_level_dir)

    for dir in subdirs:
        for file in os.listdir(dir):
            if file[0:3] != 'psp' or file[-5:] != '.fits':
                continue
            fname = f"{dir}/{file}"
            with ignore_fits_warnings():
                if order is None:
                    key = file.split('_')[3]
                    if include_headers or len(filters):
                        header = fits.getheader(fname)
                else:
                    header = fits.getheader(fname)
                    key = header[order]
            
            fkey = to_timestamp(key) if parse_key_as_timestamp else key
            if ((between[0] is not None and fkey < between[0])
                    or (between[1] is not None and fkey > between[1])):
                continue
            
            skip = False
            for filter in filters:
                value = header[filter[0]]
                if filter[1] is not None:
                    value = type(filter[1])(value)
                elif filter[2] is not None:
                    value = type(filter[2])(value)
                else:
                    continue
                if ((filter[1] is not None and value < filter[1])
                        or (filter[2] is not None and value > filter[2])):
                    skip = True
                    break
            if skip:
                continue
            
            if include_headers:
                item = (key, fname, header)
            else:
                item = (key, fname)
            
            if fname[-9] == '1':
                i_files.append(item)
            else:
                o_files.append(item)
    
    def cleaner(v):
        if not include_sortkey:
            if include_headers:
                return v[1:]
            else:
                return v[1]
        return v

    if separate_detectors:
        i_files = sorted(i_files)
        o_files = sorted(o_files)
        return [cleaner(v) for v in i_files], [cleaner(v) for v in o_files]
    
    files = sorted(i_files + o_files)
    return [cleaner(v) for v in files]


def ensure_data(input, header=True, wcs=False, wcs_key=' '):
    if isinstance(input, str):
        input = os.path.expanduser(input)
        with ignore_fits_warnings(), fits.open(input) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
            w = WCS(hdr, hdul, key=wcs_key)
    elif isinstance(input, list) or isinstance(input, tuple):
        data = input[0]
        hdr = input[1]
        try:
            w = input[2]
        except IndexError:
            if isinstance(hdr, fits.Header):
                with ignore_fits_warnings():
                    w = WCS(hdr, key=wcs_key)
            else:
                w = None
    else:
        data = input
        hdr = None
        w = None
    
    ret_val = [data]
    if header:
        ret_val.append(hdr)
    if wcs:
        ret_val.append(w)
    if len(ret_val) == 1:
        ret_val = ret_val[0]
    return ret_val


def get_hann_rolloff(shape, rolloff, zeros=0):
    """
    Generates an ND Hann rolloff window
    
    The window is zero at the edges, rises through half a Hann window to one,
    and then drops through half a Hann window to zero. In multiple dimensions,
    the corners are the product of multiple Hann functions.
    
    Parameters
    ----------
    shape : tuple or int
        The shape of the generated window
    rolloff : tuple or int
        The width of the half-Hann window (i.e. the number of pixels it takes
        to rise from 0 to 1). Can be specified as one number, or a tuple of
        numbers, one for each axis.
    zeros : tuple or int
        The number of zeros that should be at the edges before the Hann rolloff
        starts. Can be specified as one number, or a tuple of numbers, one for
        each axis.
    
    Returns
    -------
    rolloff : ``np.ndarray``
        The Hann-rolloff window
    """
    shape = np.atleast_1d(shape)
    rolloff = np.atleast_1d(rolloff)
    zeros = np.atleast_1d(zeros)
    
    if len(rolloff) == 1:
        rolloff = np.concatenate([rolloff] * len(shape))
    elif len(rolloff) != len(shape):
        raise ValueError(
                "`rolloff` must be a scalar or match the length of `shape`")
    hann_widths = rolloff * 2
    if (np.all(hann_widths <= 2)
            or np.any(hann_widths != hann_widths.astype(int))):
        raise ValueError(
                "`rolloff` should be > 1 and an integer or half-integer")
    
    if np.any(zeros < 0):
        raise ValueError("`zeros` must be > 0")
    if len(zeros) == 1:
        zeros = np.concatenate([zeros] * len(shape))
    elif len(zeros) != len(shape):
        raise ValueError(
                "`zeros` must be a scalar or match the length of `shape`")
    
    # We'll create one dimension of the output array, roll it off, duplicate &
    # stack it to create the next dimension, roll it off, etc. This lets us
    # replace lots of multiplication with copies, which is especially helpful
    # for large mask arrays. The order we go through the dimensions seems to
    # be much faster than the other way through, probably for cache reasons.
    mask = np.ones(shape[-1])
    for i, hann_width, nzeros in zip(
            range(len(shape)-1, -1, -1), hann_widths[::-1], zeros[::-1]):
        if i != len(shape) - 1:
            # Duplicate the existing dimensions
            mask = np.stack([mask] * shape[i], axis=0)
        if hann_width / 2 >= shape[i] - 2 * nzeros:
            raise ValueError(f"Rolloff size of {hann_width/2} is too large for "
                             f"dimension {i} with size {shape[i]}")
        if hann_width >= shape[i] - 2 * nzeros:
            warnings.warn(f"Rolloff size of {hann_width/2} doesn't fit for "
                          f"dimension {i} with size {shape[i]}---the two ends "
                           "overlap")
        
        # Create a [:, :, :] type of slice, and then set the index for the
        # current dimension to be just the end so we can multiply it by
        # our window.
        mask_indices = [slice(None)] * (len(shape) - i)
        
        if hann_width:
            window = scipy.signal.windows.hann(hann_width)[:ceil(hann_width/2)]
            mask_indices[0] = slice(nzeros, window.size + nzeros)
            window_indices = [None] * (len(shape) - i)
            window_indices[0] = slice(None)
            mask[tuple(mask_indices)] = (
                    mask[tuple(mask_indices)] * window[tuple(window_indices)])
            
            mask_indices[0] = slice(
                    -window.size-nzeros, -nzeros if nzeros else None)
            window_indices[0] = slice(None, None, -1)
            mask[tuple(mask_indices)] = (
                    mask[tuple(mask_indices)] * window[tuple(window_indices)])
        if nzeros:
            mask_indices[0] = slice(0, nzeros)
            mask[tuple(mask_indices)] = 0
            mask_indices[0] = slice(-nzeros, None)
            mask[tuple(mask_indices)] = 0
    return mask


@contextmanager
def ignore_fits_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
                action='ignore', message=".*'BLANK' keyword.*")
        warnings.filterwarnings(
                action='ignore', message=".*datfix.*")
        yield


def sliding_window_stats(data, window_width, stats=['mean', 'std'],
        trim=None, sliding_window_stride=1):
    """
    Computes stats in a sliding window.
    
    The output is an array that matches the shape of the input array, where
    each pixel is the statistic for a window centered on that pixel. Values are
    duplicated at the edges, where the window does not fit in the input array.
    
    Parameters
    ----------
    data : ``ndarray``
        The input array. Can have any dimensionality.
    window_width : int
        The size of the sliding window. Currently restricted to having the same
        size in all dimensions.
    stats : str or list of str
        The statistics to compute. Multiple stats can be specified. Supported
        options are 'mean', 'std', and 'median'. For each, ``NaN`` values within
        the window are ignored.
    trim : list
        A list of 2*N values, where N is the number of dimensions of the input
        array. Specifies a number of pixels to remove from the beginning and
        end of each dimension before computing stats. The outermost output
        pixels will be replicated that many additional times so the output and
        input arrays have the same shape. The order is
        ``(begin1, end1, begin2, end2, ...)``.
    sliding_window_stride : int
        Set this to a value N > 1 to compute the sliding window only for every
        N pixels along each dimension. The computed values will then be
        replicated into the skipped pixels.
    
    """
    stats_orig = stats
    if isinstance(stats, str):
        stats = [stats]
    if trim is None:
        trim = [0, 0] * len(data.shape)
    
    cut = []
    for i in range(len(data.shape)):
        cut.append(slice(trim[2*i], data.shape[i] - trim[2*i + 1]))
    data_trimmed = data[tuple(cut)]
    
    sliding_window = np.lib.stride_tricks.sliding_window_view(
            data_trimmed, [window_width] * len(data.shape))
    cut = [slice(None, None, sliding_window_stride)] * len(data.shape)
    sliding_window = sliding_window[tuple(cut)]
    
    outputs = []
    name_to_fcn = {
            'mean': np.nanmean,
            'std': np.nanstd,
            'median': np.nanmedian,
        }
    
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore',
                message=".*empty slice.*")
        warnings.filterwarnings(action='ignore',
                message=".*degrees of freedom <= 0.*")
        
        for stat in stats:
            # In the sliding window, the first N dimensions are the dimensions
            # of the input array, and the second N dimensions are the
            # dimensions of the window. E.g. for a 5x5 window over a 25x25
            # array, the sliding window ix 23x23x5x5.
            window_axes = np.arange(len(data.shape), 2*len(data.shape))
            outputs.append(name_to_fcn[stat](sliding_window,
                axis=tuple(window_axes)))
    
    if sliding_window_stride > 1:
        # Duplicate rows/columns to account for the stride
        for i in range(len(outputs)):
            for j in range(len(data.shape)):
                outputs[i] = np.repeat(outputs[i], sliding_window_stride,
                        axis=j)
    # Pad the edges, since the sliding window can't go all the way to the edge.
    padding = []
    for i in range(len(data.shape)):
        pad = data.shape[i] - outputs[0].shape[i] - trim[2*i] - trim[2*i + 1]
        padding.append((pad//2 + trim[2*i], ceil(pad/2) + trim[2*i + 1]))
    
    for i in range(len(outputs)):
        outputs[i] = np.pad(outputs[i], padding, mode='edge')
    
    if isinstance(stats_orig, str):
        return outputs[0]
    return outputs

