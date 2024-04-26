from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from math import ceil
import os
import re
import warnings

from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseLowLevelWCS, HighLevelWCSWrapper
import numba
import numpy as np
import scipy.signal


def to_timestamp(datestring, as_datetime=False, read_headers=False):
    """
    Converts to a standard number-of-seconds POSIX timestamp
    
    Parameters
    ----------
    datestring: ``str``, float, int, or Iterable
        If a ``str`` ending in ``.fits``, a filename from which the timestamp
        should be extracted (but see the ``read_headers`` option). If otherwise
        a ``str``, a timestamp as stored in WISPR headers. If a ``float`` or
        ``int``, a POSIX timestamp. If an iterable, multiple of the preceding
        types.
    as_datetime : ``bool``
        If ``True``, timestamps are returned as ``datetime`` instances, rather
        than numeric timestamps.
    read_headers : ``bool``
        If True, when filenames are provided, read ``DATE-AVG`` from the
        headers, rather than using the filename timestamp (which is
        ``DATE-BEG``).
    """
    if isinstance(datestring, fits.Header):
        datestring = datestring['date-avg']
    if (isinstance(datestring, Iterable)
            and not isinstance(datestring, str)
            and (not isinstance(datestring, u.Quantity)
                 or datestring.size > 1)):
        return [to_timestamp(
                    x, as_datetime=as_datetime, read_headers=read_headers)
                for x in datestring]
    if isinstance(datestring, (float, int)):
        if as_datetime:
            return datetime.fromtimestamp(datestring, timezone.utc)
        return datestring
    if isinstance(datestring, u.Quantity):
        return to_timestamp(datestring.to_value(u.s), as_datetime=as_datetime)
    if datestring == '':
        if as_datetime:
            return None
        return np.nan
    # Check if we got a filename
    if datestring.endswith('.fits') or datestring.endswith('.fits.gz'):
        if read_headers:
            with ignore_fits_warnings():
                datestring = fits.getheader(datestring)['date-avg']
        else:
            # Grab just the filename if it's a full path
            datestring = os.path.basename(datestring)
            # Extract the timestamp part of the standard WISPR filename
            datestring = datestring.split('_')[3]
    datestring = datestring.replace(' ', 'T')
    try:
        dt = datetime.strptime(
                datestring, "%Y%m%dT%H%M%S")
    except ValueError:
        try:
            dt = datetime.strptime(
                    datestring, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            dt = datetime.strptime(
                    datestring, "%Y-%m-%dT%H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)
    if as_datetime:
        return dt
    return dt.timestamp()


def from_timestamp(timestamp, millis=False):
    if isinstance(timestamp, Iterable) and not isinstance(timestamp, str):
        return [from_timestamp(ts) for ts in timestamp]
    if not isinstance(timestamp, (float, int)):
        raise ValueError("Invalid timestamp type")
    datetime = to_timestamp(timestamp, as_datetime=True)
    if not millis or timestamp - int(timestamp) == 0:
        fmtstr = "%Y-%m-%dT%H:%M:%S"
    else:
        fmtstr = "%Y-%m-%dT%H:%M:%S.%f"
    return datetime.strftime(fmtstr)


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
    
    Invalid files are omitted. It is expected that file names are in the
    standard formated provided by the WISPR team.
    
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
    if not os.path.exists(top_level_dir):
        raise FileNotFoundError(f'Could not find directory {top_level_dir}')
    for dirpath, _, fnames in os.walk(top_level_dir):
        for file in fnames:
            if (file[0:3] != 'psp'
                    or (file[-5:] != '.fits' and file[-8:] != '.fits.gz')):
                continue
            fname = os.path.join(dirpath, file)
            with ignore_fits_warnings():
                if order is None:
                    key = file.split('_')[3]
                    if include_headers or len(filters):
                        header = fits.getheader(fname)
                        if (header['naxis'] == 0
                                and header.get('extend', False)):
                            header = fits.getheader(fname, 1)
                else:
                    header = fits.getheader(fname)
                    if header['naxis'] == 0 and header.get('extend', False):
                        header = fits.getheader(fname, 1)
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
            hdu = 1 if hdul[0].data is None else 0
            data = hdul[hdu].data
            hdr = hdul[hdu].header
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
        trim=None, sliding_window_stride=1, where=None, check_nans=True,
        stride_fill='repeat'):
    """
    Computes stats in a sliding window.
    
    The output is an array that matches the shape of the input array, where
    each pixel is the statistic for a window centered on that pixel. Values are
    duplicated at the edges, where the window does not fit in the input array.
    
    Note: When calculating the mean or standard deviation, if the input array
    has <= 3 dimensions, an optimized routine is used that calculates rolling
    statistics, which may introduce some floating-point error (e.g. regions
    with the same exact input values may have different outputs at the
    floating-point-error level). Otherwise, the statistic is computed from
    scratch at each location in the array.
    
    Parameters
    ----------
    data : ``ndarray``
        The input array. Can have any dimensionality.
    window_width : int or tuple
        The size of the sliding window. If an integer, it is applied to all
        axes.
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
    sliding_window_stride : int or tuple
        Set this to a value N > 1 to compute the sliding window only for every
        N pixels along each dimension. The computed values will then be
        interpolated into the skipped pixels. If an integer, it is applied to
        all axes.
    where : ``ndarray``
        An optional binary mask indicating which pixels to include in the
        calculations. Cannot be used when calculating medians.
    check_nans : boolean
        Whether to use the NaN-handling calculation functions, which does slow
        things down. Note that for 'mean' and 'std', infinities are also
        filtered if ``check_nans`` is ``True``.
    stride_fill : str
        Sets the mode for filling in the skipped values when
        sliding_window_stride > 1. Allowed values are 'repeat' and 'interp',
        to select between simply repeat values into the following rows/columns,
        or linearly interpolating them. Note that mean calculations are fast
        enough that, depending on factors such as the window size, calculating
        a mean with a sliding_window_stride > 1 and linearly interpolating the
        skipped values may be slower than just letting
        sliding_window_stride==1.
    """
    if where is not None and 'median' in stats:
        raise ValueError("'where' parameter not supported for medians")
    
    stats_orig = stats
    if isinstance(stats, str):
        stats = [stats]
    
    if trim is None:
        trim = [0, 0] * data.ndim
    
    if type(window_width) is int:
        window_width = [window_width] * data.ndim
    
    if type(sliding_window_stride) is int:
        sliding_window_stride = np.full(
                data.ndim, sliding_window_stride, dtype=int)
    else:
        sliding_window_stride = np.asarray(sliding_window_stride, dtype=int)
    sliding_window_stride[sliding_window_stride < 1] = 1
    
    for i in range(len(data.shape)):
        if trim[2*i] + trim[2*i+1] + window_width[i] > data.shape[i]:
            raise ValueError(f"Trim and window do not fit along dimension {i}")
    
    # Impose the `trim` parameter
    cut = []
    for i in range(len(data.shape)):
        cut.append(slice(trim[2*i], data.shape[i] - trim[2*i + 1]))
    data_trimmed = data[tuple(cut)]
    if where is not None:
        where_trimmed = where[tuple(cut)]
    
    # For an (x, y) input array with window sizes (wx, wy), this produces a
    # view (ox, oy, wx, wy), where for any valid coordinate (ox, oy) in the
    # output array, the last two dimensions will span the portion of the input
    # array that corresponds to the window position for this output pixel.
    sliding_window = np.lib.stride_tricks.sliding_window_view(
            data_trimmed, window_width)
    
    # Impose the window stride, if any. Note that for a stride of N, we want to
    # sample every Nth point starting at N/2---not at zero---so that when we
    # fill in the skipped values, the computed values are centered in their
    # replication region.
    cut = []
    start_coords = []
    for i in range(data.ndim):
        start = min((
            sliding_window_stride[i] - 1) // 2, data_trimmed.shape[i] // 2)
        start_coords.append(start)
        cut.append(slice(start, None, sliding_window_stride[i]))
    sliding_window = sliding_window[tuple(cut)]
    if where is not None:
        sliding_where = np.lib.stride_tricks.sliding_window_view(
                where_trimmed, window_width)
        sliding_where = sliding_where[tuple(cut)]
    else:
        # The actual default value these functions use
        sliding_where = True
    
    name_to_fcn = {
            'mean': np.mean if check_nans else np.mean,
            'std': np.std if check_nans else np.std,
            'median': np.nanmedian if check_nans else np.median,
        }
    
    if ('mean' in stats or 'std' in stats) and data.ndim <= 3:
        # We have an optimized function to compute these stats in a rolling
        # window, though it only supports exactly three dimensions. We'll have
        # to adjust some arguments to make it work.
        d = data_trimmed
        ww = np.asarray(window_width).copy()
        sc = np.array(start_coords)
        st = sliding_window_stride.copy()
        if where is None:
            # The optimized function requires a one-element array when there's
            # no actual 'where' array
            w = np.ones([1] * data.ndim)
        else:
            w = where_trimmed
        while d.ndim < 3:
            # Add dummy dimensions as needed
            d = np.expand_dims(d, 0)
            w = np.expand_dims(w, 0)
            ww = np.insert(ww, 0, 1)
            sc = np.insert(sc, 0, 0)
            st = np.insert(st, 0, 1)
        
        means, stds = _sliding_window_mean_std_optimized(
                d, ww, 'mean' in stats, 'std' in stats,
                w, check_nans, st, sc)
        if 'mean' in stats:
            while means.ndim > data.ndim:
                means = means[0]
            name_to_fcn['mean'] = lambda *args, **kwargs: means
        if 'std' in stats:
            while stds.ndim > data.ndim:
                stds = stds[0]
            name_to_fcn['std'] = lambda *args, **kwargs: stds
    
    outputs = []
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore',
                message=".*empty slice.*")
        warnings.filterwarnings(action='ignore',
                message=".*degrees of freedom <= 0.*")
        
        window_axes = tuple(np.arange(len(data.shape), 2*len(data.shape)))
        for stat in stats:
            # In the sliding window, the first N dimensions are the dimensions
            # of the input array, and the second N dimensions are the
            # dimensions of the window. E.g. for a 5x5 window over a 25x25
            # array, the sliding window ix 21x21x5x5.
            kwargs = {}
            if stat != 'median':
                kwargs['where'] = sliding_where
            outputs.append(name_to_fcn[stat](sliding_window,
                axis=window_axes,
                **kwargs
            ))
    
    if np.any(sliding_window_stride > 1):
        # Fill in the skipped rows/columns
        if stride_fill.lower() == 'repeat':
            for i in range(len(outputs)):
                for j in range(len(data.shape)):
                    if sliding_window_stride[j] == 1:
                        continue
                    repeats = np.full(
                            outputs[i].shape[j], sliding_window_stride[j])
                    # The last few rows/columns that we're supposed to create
                    # may be in the zone of influence centered on a spot that
                    # doesn't exist, so we need to repeat the last computed
                    # element a few more times. Alternatively, we may end up
                    # repeating the last element too many times, and this will
                    # clamp that down.
                    repeats[-1] = (data_trimmed.shape[j] - window_width[j] + 1
                            - (outputs[i].shape[j] - 1)
                                * sliding_window_stride[j])
                    outputs[i] = np.repeat(outputs[i], repeats,
                            axis=j)
        elif 'interp' in stride_fill.lower():
            # Get the coordinates of the computed rows/columns in the input
            # array
            strided_coords = tuple(
                    np.arange(
                        start_coords[i],
                        data_trimmed.shape[i] - window_width[i] + 1,
                        sliding_window_stride[i])
                    for i in range(data.ndim))
            
            # Get the coordinates in the input array of all the pixels that
            # should be computed (including those that we're about to fill in)
            coords = [np.arange(sc[0], sc[-1]+1, dtype=np.uint32)
                    for sc in strided_coords]
            # Pad out each dimension, where we can't interpolate, so
            # the right number of pixels come out.
            for i in range(data.ndim):
                target_size = data_trimmed.shape[i] - window_width[i] + 1
                if coords[i].size < target_size:
                    n_to_add = target_size - coords[i].size - coords[i][0]
                    coords[i] = np.concatenate(
                            [coords[i][:1]] * coords[i][0]
                            + [coords[i]]
                            + [coords[i][-1:]] * n_to_add)
            
            coords = np.meshgrid(*coords, indexing='ij')
            coords = np.stack(coords, axis=-1)
            
            for j, output in enumerate(outputs):
                # Interpolate to get the missing pixels. (The actually-computed
                # pixels appear to be returned unmodified.)
                outputs[j] = scipy.interpolate.interpn(strided_coords, output,
                        coords, method='linear', bounds_error=True)
        else:
            raise ValueError(f"Invalid value {stride_fill} for 'stride_fill'")
    
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


@numba.njit(cache=True)
def _sliding_window_mean_std_optimized(data, window_width, mean, std, where,
        check_nans, stride, start_coords):
    """
    Computes the mean and/or standard deviation in a rolling window.
    
    Uses accumulation variables that are updated with only the values
    added/removed at each step, to avoid re-computing the statistics from
    scratch for each rolling window position.
    
    That's easy for the mean---just store a sum and N.
    
    For incremental updates to the variance:
    
    Variance = Sigma (x_i - mu)^2 / N
    
    Expanding the square:
    Variance = (Sigma x_i^2 + Sigma mu^2 - 2 mu Sigma x_i) / N
             = (Sigma x_i^2 + N mu^2     - 2 mu Sigma x_i) / N
             
    Sigma x_i / N = mu, so
             
    Variance = Sigma x_i^2 / N + mu^2 - 2 mu * mu
             = Sigma x_i^2 / N - mu^2
    
    If we have points
        x1, ..., xl, x_{l+1}, ..., xm, x_{m+1}, ..., xn
        |--remove--| |-----keep-----|  |------add-----|
    and in our step we're removing the A points x1...xl and adding the B points
    x_{m+1}...xn, the difference between the old and new variance is
    
    dvar = Var(x_{l+1}...xm) - Var(x1...xm)
         = 1/A Sigma_{l+1}^n (xi - mu_new)^2  - 1/B Sigma_1^m (xi - mu_old)^2
         = 1/A Sigma_{l+1}^n x_i^2 - mu_new^2 - 1/B Sigma_1^m x_i^2 + mu_old^2
           |--------------------------------|   |----------------------------|
             only involving the new mu and      only involving the old mu and
             the new sum-of-squares, after      the old sum-of-squares, before
             all new points are added           any old points are removed
    
    Thus, if we track the variance and the sum-of-squares in our window, we can
    incrementally update the variance as we move the window.
    """
    # For each dimension, find the number of valid sliding window positions
    out_shape = (
            int(np.ceil((data.shape[0] - window_width[0] + 1 - start_coords[0])
                / stride[0])),
            int(np.ceil((data.shape[1] - window_width[1] + 1 - start_coords[1])
                / stride[1])),
            int(np.ceil((data.shape[2] - window_width[2] + 1 - start_coords[2])
                / stride[2])))
    # If we have a size-one dimension, ensure we keep size 1 in the output
    # (important if we're striding)
    out_shape = (
            max(1, out_shape[0]),
            max(1, out_shape[1]),
            max(1, out_shape[2]))
    output_mean = np.empty(out_shape, dtype=np.float64) if mean else None
    output_std = np.empty(out_shape, dtype=np.float64) if std else None

    directions = stride.copy()
    # This marks the location in the output arary we should store stats
    pos = np.array((0, 0, 0))
    # This marks the corresponding location of the window in the data array
    window_bounds = np.empty((3, 2), dtype=np.int64)
    window_bounds[:, 0] = start_coords
    window_bounds[:, 1] = start_coords + window_width
    
    # Accumulate all the valid points in the starting window.
    # Note this is more than just filling the accumulation variables---we're
    # also collecting all the valid values to compute the starting variance.
    sum, sum_of_squares, N = 0, 0, 0
    start_data = np.empty((window_width[0] * window_width[1] * window_width[2]))
    for i in range(window_bounds[0, 0], window_bounds[0, 1]):
        for j in range(window_bounds[1, 0], window_bounds[1, 1]):
            for k in range(window_bounds[2, 0], window_bounds[2, 1]):
                if check_nans and not np.isfinite(data[i, j, k]):
                    continue
                if where.size > 1 and not where[i, j, k]:
                    continue
                start_data[N] = data[i, j, k]
                sum += data[i, j, k]
                sum_of_squares += data[i, j, k]**2
                N += 1
    
    start_data = start_data[:N]
    # n.b. it sounds like, even though we're updating the variance by hand as
    # we iterate, the initial calculation should be from a dedicated variance
    # function, as those are often designed to avoid numeric pitfalls.
    variance = np.var(start_data) if start_data.size else 0

    while True:
        # Store mean and/or std for the current position
        mean_val = (sum / N) if N > 0 else np.nan
        if mean:
            output_mean[pos[0], pos[1], pos[2]] = mean_val
        if std:
            if N > 0:
                output_std[pos[0], pos[1], pos[2]] = np.sqrt(variance)
                # The increment we apply to the variance next time around
                # includes two terms involving the "old" values.
                delta_variance = (mean_val**2 - sum_of_squares / N)
            else:
                output_std[pos[0], pos[1], pos[2]] = np.nan
                delta_variance = 0
        
        # Check if we've hit the end along dimension -1
        if (pos[-1] + np.sign(directions[-1]) >= out_shape[-1]
            or pos[-1] + np.sign(directions[-1]) < 0
        ):
            # We've hit the end along dimension -1. Are we also at the end of
            # dimension -2?
            if (pos[-2] + np.sign(directions[-2]) >= out_shape[-2]
                or pos[-2] + np.sign(directions[-2]) < 0
            ):
                # We've hit the end along dimension -2. Are we also at the end
                # of dimension -3?
                if (pos[-3] + np.sign(directions[-3]) >= out_shape[-3]
                    or pos[-3] + np.sign(directions[-3]) < 0
                ):
                    # We've reached the end of the whole array!
                    break
                else:
                    # We need to advance along dimension -3
                    dsum, dsum_of_squares, dn = _sliding_window_advance(
                            directions, window_bounds, pos, stride, data,
                            check_nans, where, -3)
                    sum += dsum
                    sum_of_squares += dsum_of_squares
                    N += dn
            else:
                # We need to advance along dimension -2
                dsum, dsum_of_squares, dn = _sliding_window_advance(
                        directions, window_bounds, pos, stride, data,
                        check_nans, where, -2)
                sum += dsum
                sum_of_squares += dsum_of_squares
                N += dn
        else:
            # We need to advance along dimension -1
            dsum, dsum_of_squares, dn = _sliding_window_advance(
                    directions, window_bounds, pos, stride, data,
                    check_nans, where, -1)
            sum += dsum
            sum_of_squares += dsum_of_squares
            N += dn
        
        if std:
            # The increment we apply to the variance includes two terms
            # involving the "new" values.
            if N > 0:
                delta_variance += -(sum / N)**2 + sum_of_squares / N
            variance += delta_variance
    
    return output_mean, output_std


@numba.njit(cache=True)
def _sliding_window_advance(directions, window_bounds, pos, stride, data,
        check_nans, where, axis):
    """
    Utility function to handle advancing our window along a certain axis.
    Returns increments to the accumulation variables due to the step.
    """
    # We need to advance along dimension N
    # First remove old values. Identify the bounds of the removed region.
    if directions[axis] > 0:
        x = window_bounds[axis, 0]
        x2 = x + stride[axis]
    else:
        x2 = window_bounds[axis, 1]
        x = x2 - stride[axis]
    
    i, i2 = window_bounds[-3, 0], window_bounds[-3, 1]
    j, j2 = window_bounds[-2, 0], window_bounds[-2, 1]
    k, k2 = window_bounds[-1, 0], window_bounds[-1, 1]
    
    # Assign the removed-region bounds to the appropriate axis
    if axis == -3:
        i, i2 = x, x2
    elif axis == -2:
        j, j2 = x, x2
    elif axis == -1:
        k, k2 = x, x2
    
    dsum, dsum_of_squares, dn = _sliding_window_sum_range(
            data, check_nans, where, i, i2, j, j2, k, k2)
    
    sum_update = -dsum
    sum_of_squares_update = -dsum_of_squares
    N_update = -dn
    
    # Now advance the window. Advance the window by the full stride. It has
    # already been checked that we can move this far.
    window_bounds[axis] += directions[axis]
    # And advance our position in the output arrays by +/- 1
    pos[axis] += np.sign(directions[axis])
    
    # Flip our movement direction for the other dimensions if appropriate.
    # For the last dimension, this isn't necessary, but for the other
    # dimensions, we only advance when we've reached the end along the lower
    # dimensions.
    if axis != -1:
        directions[axis+1:] *= -1
    
    # Now add the new values. Identify the bounds of the added region.
    if directions[axis] > 0:
        x2 = window_bounds[axis, 1]
        x = x2 - stride[axis]
    else:
        x = window_bounds[axis, 0]
        x2 = x + stride[axis]
    
    i, i2 = window_bounds[-3, 0], window_bounds[-3, 1]
    j, j2 = window_bounds[-2, 0], window_bounds[-2, 1]
    k, k2 = window_bounds[-1, 0], window_bounds[-1, 1]
    
    # Assign the added-region bounds to the appropriate axis
    if axis == -3:
        i, i2 = x, x2
    elif axis == -2:
        j, j2 = x, x2
    elif axis == -1:
        k, k2 = x, x2
    
    dsum, dsum_of_squares, dn = _sliding_window_sum_range(
            data, check_nans, where, i, i2, j, j2, k, k2)
    sum_update += dsum
    sum_of_squares_update += dsum_of_squares
    N_update += dn
    
    return sum_update, sum_of_squares_update, N_update


@numba.njit(cache=True)
def _sliding_window_sum_range(data, check_nans, where, i1, i2, j1, j2, k1, k2):
    """
    Utility function to collect all the values within a certain range, handle
    NaNs and `where`, and returns updates to the accumulation variables.
    """
    sum = 0
    sum_of_squares = 0
    n = 0
    for i in range(i1, i2):
        for j in range(j1, j2):
            for k in range(k1, k2):
                if check_nans and not np.isfinite(data[i, j, k]):
                    continue
                if where.size > 1 and not where[i, j, k]:
                    continue
                sum_of_squares += data[i, j, k] ** 2
                sum += data[i, j, k]
                n += 1
    return sum, sum_of_squares, n


def to_orbital_plane_rtheta(x, y, z):
    """ Converts (x,y,z) coordinates of the s/c to in-orbit-plane (x,y) """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2))
    
    return r, phi


def load_orbit_plane_xy(files):
    """ Returns the in-orbit-plane (x,y) path of the s/c from these files """
    r, theta = load_orbit_plane_rtheta(files)
    return r * np.cos(theta), r * np.sin(theta)


def load_orbit_plane_rtheta(files):
    """
    Returns the in-orbit-plane (r, theta) path of the s/c from these files
    """
    xs, ys, zs = load_sc_xyz(files)
    return to_orbital_plane_rtheta(xs, ys, zs)


def load_sc_xyz(files):
    """
    Returns the Heliocentric-Inertial path of the s/c from these files
    """
    if isinstance(files, (str, fits.Header)):
        files = [files]
    xs = []
    ys = []
    zs = []
    with ignore_fits_warnings():
        for f in files:
            if isinstance(f, str):
                with fits.open(f) as hdul:
                    if len(hdul) == 4:
                        h= fits.getheader(f, 1)
                    else:
                        h = fits.getheader(f)
            else:
                h = f
            xs.append(h['hcix_obs'])
            ys.append(h['hciy_obs'])
            zs.append(h['hciz_obs'])
    return np.array(xs), np.array(ys), np.array(zs)


def find_closest_file(target, files, key=None, headers=None):
    """
    From a list of files, find the one closest to a target value
    
    Parameters
    ----------
    target : ``float`` or ``str``
        The target value that the output file should be closest to. If a
        string, treated as a filename, and its value of ``key`` is used as
        ``target``.
    files : ``list``
        The list of files to choose from.
    key : ``str`` or ``list``
        The header key indicating the value to get closest to. If not provided,
        the timestamps in the file names are used. The header values must be
        convertible to ``float``. If a ``list``, it is taken as the value for
        each file that should be compared to ``target``.
    headers : ``astropy.io.fits.Header``
        A list of Headers corresponding to each file. If not provided, the
        headers are read in.
    """
    if key is None:
        if isinstance(target, str):
            target = to_timestamp(target)
        file_values = np.array(to_timestamp(files))
    elif isinstance(key, str):
        if isinstance(target, str):
            with ignore_fits_warnings():
                target = fits.getheader(target)
            target = float(target[key])
        
        if headers is None:
            with ignore_fits_warnings():
                headers = [fits.getheader(f) for f in files]
        file_values = np.array([float(h[key]) for h in headers])
    else:
        file_values = np.asarray(key)
    i = np.argmin(np.abs(file_values - target))
    return files[i]


@numba.njit(cache=True)
def time_window_savgol_filter(xdata, ydata, window_size, poly_order):
    output = np.zeros_like(ydata)
    # Ainv_by_cadence = dict()
    should_do_beginning = True
    should_do_end = False
    for i in range(0, len(xdata)):
        istart = i
        while xdata[i] - xdata[istart] <= window_size / 2:
            istart -= 1
            if istart < 0:
                istart = None
                break
        if istart is None:
            # Keep stepping out until we can fit the full window
            continue
        if should_do_beginning:
            istart = 0
        else:
            istart += 1
        istop = i
        while xdata[istop-1] - xdata[i] <= window_size / 2:
            istop += 1
            if istop == len(xdata):
                should_do_end = True
                break
        if should_do_end:
            istop = len(xdata)
        else:
            istop -= 1
        window_x = xdata[istart:istop] - xdata[i]
        window_x /= np.max(np.abs(window_x))
        window_y = ydata[istart:istop]
        good = np.isfinite(window_y)
        if not np.any(good):
            output[i] = np.nan
            continue
        window_x = window_x[good]
        window_y = window_y[good]
        # This was an attempt to cache values when the data spacing is the same for
        # several windows, but it didn't seem to actually speed things up much.
        # cadences = np.diff(window_x)
        # mean_cadence = np.mean(cadences)
        # deviation = np.abs(cadences / mean_cadence - 1)
        # cadence_varies = np.any(deviation > .05)
        # cadence_varies = True
        # Ainv_key = (np.round(mean_cadence), istop-istart)
        # Ainv = Ainv_by_cadence.get(Ainv_key, None)
        # if Ainv is None or cadence_varies:
            # The cadence varies enough that we should handle it, or it's pretty
            # constant but we haven't seen it yet.
            # We'll have to compute smoothing coefficients. See
            # https://dsp.stackexchange.com/a/9494
            # for how this works
        A = np.empty((len(window_x), poly_order + 1))
        for j in range(poly_order + 1):
            A[:, j] = window_x ** j
        Ainv = np.linalg.pinv(A)
        # if not cadence_varies:
        #     Ainv_by_cadence[Ainv_key] = Ainv
        if should_do_beginning:
            coeffs = np.dot(Ainv, window_y)
            for j in range(0, i+1):
                for p in range(poly_order + 1):
                    output[j] += coeffs[p] * window_x[j]**p
            should_do_beginning = False
        elif should_do_end:
            coeffs = np.dot(Ainv, window_y)
            for j in range(i, istop):
                for p in range(poly_order + 1):
                    output[j] += coeffs[p] * window_x[j - istart]**p
            break
        else:
            # We just need the interpolated value at our window center
            output[i] = np.dot(Ainv, window_y)[0]
    return output


def extract_encounter_number(path, as_int=False):
    """
    Extracts the encounter number from a data path
    """
    if isinstance(path, Iterable) and not isinstance(path, str):
        return [extract_encounter_number(
                    x, as_int=as_int)
                for x in path]
    m = re.search(r'_ENC(\d{2})_', path)
    if m is None:
        m = re.search(r'/E(\d{2})/', path)
    if m is None:
        if as_int:
            return -1
        return None
    result = m.group(1)
    if as_int:
        result = int(result)
    return result


class FakeWCS(BaseLowLevelWCS):
    """
    Helper for plugging custom coordinates in reproject.
    
    Handles all the boilerplate, so subclasses just need to implement
    world_to_pixel_values and/or pixel_to_world_values. The only catch is that
    a "matching" wcs from the other side of the reprojection must be provided
    to __init__ so some of the boilerplate can be copied from it.
    
    If ``None`` is provided for ``input_wcs``, a "unitless" mode is enabled,
    where input/output coordinates are generic and dimensionless.
    """
    def __init__(self, input_wcs):
        if input_wcs is None:
            input_wcs = WCS(naxis=2)
        self.input_wcs = input_wcs

    # The empty docstrings suppress the docs pulling in the base class
    # docstrings
    def world_to_pixel_values(self, *world_arrays):
        """ """
        raise NotImplementedError()

    def pixel_to_world_values(self, *pixel_arrays):
        """ """
        raise NotImplementedError()

    @property
    def pixel_n_dim(self):
        """ """
        return self.input_wcs.pixel_n_dim

    @property
    def world_n_dim(self):
        """ """
        return self.input_wcs.world_n_dim

    @property
    def world_axis_units(self):
        """ """
        return self.input_wcs.world_axis_units

    @property
    def world_axis_physical_types(self):
        """ """
        return self.input_wcs.world_axis_physical_types

    @property
    def world_axis_object_components(self):
        """ """
        return self.input_wcs.world_axis_object_components

    @property
    def world_axis_object_classes(self):
        """ """
        return self.input_wcs.world_axis_object_classes
    
    def as_high_level(self):
        """ """
        return HighLevelWCSWrapper(self)


def test_data_path(*segments):
    return os.path.join(
        os.path.dirname(__file__), 'tests', 'test_data', *segments)


def data_path(*segments):
    return os.path.join(os.path.dirname(__file__), 'data', *segments)


def angle_between_vectors(x1, y1, z1, x2, y2, z2):
    """Returns a signed angle between two vectors, in radians"""
    # Rotate so v1 is our x axis. We want the angle v2 makes to the x axis.
    # Its components in this rotated frame are its dot and cross products
    # with v1.
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    y1 = np.atleast_1d(y1)
    y2 = np.atleast_1d(y2)
    z1 = np.atleast_1d(z1)
    z2 = np.atleast_1d(z2)
    
    dot_product = x1 * x2 + y1 * y2 + z1 * z2
    cross_x = (y1*z2 - z1*y2)
    cross_y = (z1*x2 - x1*z2)
    cross_z = (x1*y2 - y1*x2)
    det = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
    
    angle = np.arctan2(det, dot_product)
    v1_is_zero = ((x1 == 0) * (y1 == 0) * (z1 == 0))
    v2_is_zero = ((x2 == 0) * (y2 == 0) * (z2 == 0))
    angle[v1_is_zero + v2_is_zero] = np.nan
    return angle