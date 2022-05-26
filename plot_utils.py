from collections.abc import Iterable
import copy
from datetime import datetime, timedelta, timezone
from itertools import chain
import warnings

import astropy.units as u
from astropy.wcs import WCS
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

from . import utils
    

wispr_cmap = copy.copy(plt.cm.Greys_r)
wispr_cmap.set_bad('black')

PRESET_L3_I_MIN = 0
PRESET_L3_O_MIN = 0
PRESET_L2_I_MIN = 2e-12
PRESET_L2_O_MIN = 0

COLORBAR_PRESETS = {
        '2': {
            'i': (2e-12, 3e-10),
            'o': (0, 1e-10),
            },
        '3': {
            'i': (0, 1.545e-11),
            'o': (0, .5e-11),
            }
        }


def parse_level_preset(level_preset, header=None):
    if level_preset is None:
        if (header is not None
                and header.get('level', 'L3') in ('L1', 'L2', 'L3')):
            level_preset = header.get('level', 'L3')[1]
        else:
            level_preset = '3'
    else:
        if isinstance(level_preset, str) and level_preset[0] == 'L':
            level_preset = level_preset[1]
    
    level_preset = str(level_preset)
    if level_preset not in ('1', '2', '3'):
        raise ValueError("Level preset not recognized")
    
    return level_preset


def full_size_plot(img, **kwargs):
    """
    Renders a WISPR image at full resolution, without borders or axes.
    
    All keyword arguments are passed to plot_WISPR.
    """
    fig = plt.figure(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis('off')

    plot_WISPR(img, ax, **kwargs)
    return fig, ax


def plot_WISPR(data, ax=None, cmap=None, wcs=None,
        vmin='auto', vmax='auto', wcs_key=' ',
        detector_preset=None, level_preset=None,
        grid=False, **kwargs):
    """
    Does the Right Thing to plot a WISPR image.
    
    Accepts a filename or a data array (optionally with a corresponding WCS
    object). Plots the image with sensible defaults for the color bar, and in
    world coordinates if possible. The colorbar is scaled by the square root of
    the data.
    
    Arguments
    ---------
    data
        A data array to plot, or the name of a FITS file from which to load the
        data and WCS information.
    ax
        A matplotlib ``Axes`` object to use for plotting. Optional.
    cmap
        A matplotlib colormap. Optional.
    wcs
        A ``WCS`` object to use to plot in world coordinates. Optional, and
        only needed if a data array is provided, rather than a FITS file.
    vmin, vmax
        Colorbar limits. If not provided, sensible defaults are chosen.
    wcs_key
        Which WCS to select from the FITS header. For WISPR files, a value of
        ``' '`` loads helioprojective coordinates, and ``'A'`` loads RA/Dec
        coordinates.
    detector_preset
        Colorbar limits are adjusted based on which imager the data is from.
        The imager is auto-detected from the FITS file, but can be overridden
        here with values of 'i' or 'o'. When only a data array is provided,
        the 'i' preset is chosen.
    level_preset
        Colorbar limits are adjusted based on the data level. The level is
        auto-detected from the FITS file, but can be overridden here with
        values of 1, 2, or 3. When only a data array is provided, the '3'
        preset is chosen.
    grid : boolean or float
        Whether to plot a coordinate grid over the image. If a ``float``,
        specifies the opacity of the grid lines.
    kwargs
        Extra arguments passed to ``imshow``.
    """
    data, header, w = utils.ensure_data(data, header=True, wcs=True, wcs_key=wcs_key)
    if wcs is None:
        wcs = w
    
    level_preset = parse_level_preset(level_preset, header)
    
    if detector_preset is None:
        if header is not None and header.get('detector', 1) == 2:
            detector_preset = 'o'
        else:
            detector_preset = 'i'
    else:
        detector_preset = detector_preset[0].lower()
        if detector_preset not in ('i', 'o'):
            raise ValueError("Detector preset not recognized")
    
    if vmin == 'auto':
        vmin = COLORBAR_PRESETS[level_preset][detector_preset][0]
    if vmax == 'auto':
        vmax = COLORBAR_PRESETS[level_preset][detector_preset][1]
    
    if ax is None:
        if wcs is None or wcs is False:
            ax = plt.gca()
        else:
            ax = plt.subplot(111, projection=wcs)
            lon, lat = ax.coords
            lat.set_ticks(np.arange(-90, 90, 10) * u.degree)
            lon.set_ticks(np.arange(-180, 180, 15) * u.degree)
            lat.set_major_formatter('dd')
            lon.set_major_formatter('dd')
            if grid:
                if isinstance(grid, bool):
                    grid = 0.2
                ax.coords.grid(color='white', alpha=grid)
    if cmap is None:
        cmap = wispr_cmap
    
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'
    im = ax.imshow(data, cmap=cmap,
            norm=matplotlib.colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax),
            **kwargs)
    return im


def blink(*imgs, vmins=None, vmaxs=None, cmaps=None, interval=500, show=True,
        labels=None, label_kw={}, setup_fcn=None):
    """
    Renders a movie that blinks between several images.
    
    Arguments
    ---------
    imgs
        Any number of data arrays to be plotted via `plot_WISPR`, or a list of
        functions, each of which is fully responsible for plotting a single
        image.
    vmins, vmaxs, cmaps
        A single value, or a sequence of values---one for each data array
    interval
        The time between frames
    show
        If ``True``, the matplotlib animation is displayed in the notebook
        environment. Otherwise, the animation object is returned.
    labels
        A sequence of text labels to show in the corner of each image, one for
        each input array.
    label_kw
        A dictionary of keyword arguments to pass to ``plt.text`` when
        rendering labels
    setup_fcn
        Used only when ``imgs`` is a sequence of callable functions. If not
        None, this function is called to create the ``Figure`` and ``Axes`` for
        plotting. Otherwise, ``imgs[0]`` is called for this purpose (as well as
        for plotting the first frame).
    """
    if not isinstance(vmins, Iterable):
        vmins = [vmins] * len(imgs)
    if not isinstance(vmaxs, Iterable):
        vmaxs = [vmaxs] * len(imgs)
    if not isinstance(cmaps, Iterable) and not isinstance(cmaps, str):
        cmaps = [cmaps] * len(imgs)
    
    if callable(imgs[0]):
        if setup_fcn:
            setup_fcn()
        else:
            imgs[0]()
        fig = plt.gcf()
        ax = plt.gca()
    else:
        fig, ax = full_size_plot(imgs[0])
    
    if 'x' not in label_kw:
        label_kw['x'] = .02
        label_kw['y'] = .97
        label_kw['transform'] = ax.transAxes
    if 'color' not in label_kw:
        label_kw['color'] = 'white'
    
    def update(i):
        ax.clear()
        if callable(imgs[i]):
            imgs[i]()
        else:
            plot_WISPR(imgs[i], ax=ax, vmin=vmins[i], vmax=vmaxs[i],
                    cmap=cmaps[i])
        ax.axis('off')
        if labels is not None:
            plt.text(s=labels[i], zorder=99, **label_kw)

    ani = FuncAnimation(fig, update, frames=range(len(imgs)), interval=interval)
    if show:
        display(HTML(ani.to_html5_video()))
        plt.close()
    else:
        return ani


def plot_orbit(data_dir):
    """
    Plots PSP's orbital path and marks the locations of WISPR exposures.
    
    Only the plane-of-the-ecliptic components of the spacecraft position are
    plotted.
    
    Arguments
    ---------
    data_dir
        A directory containing WISPR images. It is scanned recursively by
        `utils.collect_files`, and exposure times and spacecraft locations are
        extracted from the headers.
    """
    ifiles, ofiles = utils.collect_files(data_dir, include_headers=True,
            include_sortkey=True, order='date-avg')
    headers = [f[-1] for f in sorted(chain(ifiles, ofiles))]
    itimes = [utils.to_timestamp(f[0]) for f in ifiles]
    otimes = [utils.to_timestamp(f[0]) for f in ofiles]
    times, positions, _ = utils.get_PSP_path_from_headers(headers)
    
    distance_scale = max(positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min())
    
    # Plot the s/c trajectory
    plt.plot(positions[:, 0], positions[:, 1])
    
    # Plot the Sun
    plt.scatter(0, 0, marker='*', color='k', s=80)
    
    ixs = np.interp(itimes, times, positions[:, 0])
    oxs = np.interp(otimes, times, positions[:, 0])
    iys = np.interp(itimes, times, positions[:, 1])
    oys = np.interp(otimes, times, positions[:, 1])
    
    ithetas = np.arctan2(np.gradient(iys), np.gradient(ixs))
    ithetas += np.pi/2
    othetas = np.arctan2(np.gradient(oys), np.gradient(oxs))
    othetas += np.pi/2
    
    ixs += 0.05 * distance_scale * np.cos(ithetas)
    iys += 0.05 * distance_scale * np.sin(ithetas)
    oxs += 0.025 * distance_scale * np.cos(othetas)
    oys += 0.025 * distance_scale * np.sin(othetas)
    
    # Plot the exposure markers with transparency, and plot dummy points
    # without transparency for the plot legend.
    plt.scatter(ixs, iys, color='C1', s=5, alpha=0.2)
    plt.scatter(None, None, color='C1', s=5, label='WISPR-I')
    plt.scatter(oxs, oys, color='C2', s=5, alpha=0.2)
    plt.scatter(None, None, color='C2', s=5, label='WISPR-O')
    plt.gcf().legend(loc='upper right', frameon=False, title='Exposures')
    
    # Collect information for marking dates
    one_day = timedelta(days=1)
    start_date = datetime.fromtimestamp(times[0], timezone.utc)
    end_date = datetime.fromtimestamp(times[-1], timezone.utc) + one_day
    end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    xs = []
    ys = []
    strings = []
    date += one_day
    while date < end_date:
        t = date.timestamp()
        
        x = np.interp(t, times, positions[:, 0])
        y = np.interp(t, times, positions[:, 1])
        
        xs.append(x)
        ys.append(y)
        datestring = date.date().isoformat()
        strings.append(datestring)
        date += one_day
    
    xs = np.array(xs)
    ys = np.array(ys)
    
    thetas = np.arctan2(np.gradient(ys), np.gradient(xs))
    thetas -= np.pi/2
    
    ticks_x = np.vstack((
        xs + 0.01 * distance_scale * np.cos(thetas),
        xs - 0.01 * distance_scale * np.cos(thetas),
        xs * np.nan)).T.flatten()
    ticks_y = np.vstack((
        ys + 0.01 * distance_scale * np.sin(thetas),
        ys - 0.01 * distance_scale * np.sin(thetas),
        ys * np.nan)).T.flatten()
    plt.plot(ticks_x, ticks_y, color='C0')
    
    xs += 0.025 * distance_scale * np.cos(thetas)
    ys += 0.025 * distance_scale * np.sin(thetas)
    
    for i, (x, y, theta, datestring) in enumerate(zip(xs, ys, thetas, strings)):
        # Check if the tick marks are too close together. If so, skip labeling
        # every other date.
        if (i > 0 and
                np.sqrt((x - xs[i-1])**2 + (y - ys[i-1])**2)
                < 0.04 * distance_scale
                and i % 2):
            continue
            
        alignment = 'left'
        if theta > np.pi/2 or theta < -np.pi/2:
            alignment = 'right'
            theta += np.pi
        
        plt.text(x, y, datestring,
                rotation=theta * 180 / np.pi,
                rotation_mode='anchor',
                horizontalalignment=alignment,
                verticalalignment='center')
    
    plt.axis('off')
    plt.gca().set_aspect('equal')


def x_axis_dates(*args, **kwargs):
    """
    Helper to format the x axis as dates and convert dates to matplotlib's format.
    
    Matplotlib expects dates to be given as the number of days since 0001-01-01
    UTC, plus 1, and can be configured to display those timestamps as
    YYYY-MM-DD dates. This function converts a list of timestamps to
    matplotlib's format while also configuring the x axis to display as dates.
    
    Arguments
    ---------
    dates
        A list of dates to be converted to a matplotlib-friendly format. Should
        be either a format understood by utils.to_timestamp, or POSIX timestamps.
    ax
        An Axes instance or an iterable of Axes instances. Optional, defaults
        to plt.gca()
    fig
        The Figure instance containing the Axes or list of Axes. Optional, defaults
        to plt.gcf()
    
    Returns
    -------
    dates
        A list of dates that have been converted to matplotlib's date format,
        ready for use as the x-axis quantity when plotting.
    """
    return axis_dates(*args, **kwargs, axis='x')


def y_axis_dates(*args, **kwargs):
    """
    Helper to format the y axis as dates and convert dates to matplotlib's format.
    
    Same as `x_axis_dates`, but applies to the y axis instead.
    """
    
    return axis_dates(*args, **kwargs, axis='y')


def axis_dates(dates, axis='x', ax=None, fig=None):
    if isinstance(dates[0], str):
        dates = [utils.to_timestamp(d) for d in dates]
    dates = [datetime.fromtimestamp(d, tz=timezone.utc)
             if np.isfinite(d) else None
             for d in dates]
    dates = [mdates.date2num(d) if d is not None else np.nan for d in dates]
    
    if ax is None:
        ax = [plt.gca()]
    else:
        try:
            len(ax)
        except TypeError:
            ax = [ax]
    
    if fig is None:
        fig = plt.gcf()
    
    for a in ax:
        # Fresh locators/formatters are needed for each instance
        loc = mdates.AutoDateLocator()
        fmt = mdates.AutoDateFormatter(loc)
        if axis == 'x':
            axis_obj = a.xaxis
        else:
            axis_obj = a.yaxis
        
        axis_obj.set_major_locator(loc)
        axis_obj.set_major_formatter(fmt)
        
        if axis == 'x':
            for label in a.get_xticklabels(which='major'):
                label.set_ha('right')
                label.set_rotation(30)
    
    if axis == 'x':
        fig.subplots_adjust(bottom=.2)
    else:
        fig.subplots_adjust(left=.2)
    
    return dates


def date_to_mdate(date):
    """
    Converts a single date to matplotlib's format. See `x_axis_dates` for details.
    
    Arguments
    ---------
    date
        A date to be converted to a matplotlib-friendly format. Should be
        either a format understood by utils.to_timestamp, or a POSIX timestamp.
    Returns
    -------
    date
        A date that have been converted to matplotlib's date format, ready for
        use as the x-axis quantity when plotting.
    """
    if isinstance(date, str):
        date = utils.to_timestamp(date)
    date = datetime.fromtimestamp(date, tz=timezone.utc)
    return mdates.date2num(date)

