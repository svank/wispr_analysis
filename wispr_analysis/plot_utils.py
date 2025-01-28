from collections.abc import Iterable
import copy
from datetime import datetime, timedelta, timezone
import os

import astropy.units as u
from astropy.visualization import quantity_support
from astropy.visualization.wcsaxes import WCSAxes
from ipywidgets import interact
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

from . import constellations, planets, utils
from .orbital_frame import PSPOrbitalFrame


quantity_support()
    

wispr_cmap = copy.copy(plt.cm.Greys_r)
wispr_cmap.set_bad('black')

PRESET_L3_I_MIN = 0
PRESET_L3_O_MIN = 0
PRESET_L2_I_MIN = 2e-12
PRESET_L2_O_MIN = 0

COLORBAR_PRESETS = {
        '2': {
            'i': (PRESET_L2_I_MIN, 3e-10),
            'o': (PRESET_L2_O_MIN, 1e-10),
            },
        '3': {
            'i': (PRESET_L3_I_MIN, 1.545e-11),
            'o': (PRESET_L3_O_MIN, .5e-11),
            },
        'W': {
            'i': (.997, 1.03),
            'o': (.997, 1.03)},
        }


def parse_level_preset(level_preset, header=None):
    if level_preset is None:
        if (header is not None
                and header.get('level', 'L3') in ('L1', 'L2', 'L3', 'LW')):
            level_preset = header.get('level', 'L3')[1]
        else:
            level_preset = '3'
    else:
        if isinstance(level_preset, str) and level_preset[0] == 'L':
            level_preset = level_preset[1]
        elif (isinstance(level_preset, str) and level_preset.endswith('.fits')):
            level_preset = os.path.basename(level_preset)
            if level_preset.startswith('psp_L'):
                level_preset = level_preset[5]
    
    level_preset = str(level_preset)
    if level_preset not in ('1', '2', '3', 'W'):
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


def plot_WISPR(data, ax=None, wcs=None,
        vmin='auto', vmax='auto', wcs_key=' ',
        detector_preset=None, level_preset=None,
        grid=False, lat_spacing=10, lon_spacing=10,
        relative_vmin=1, relative_vmax=1, gamma=1/2.2,
        draw_constellations=False, mark_planets=False,
        **kwargs):
    """
    Does the Right Thing to plot a WISPR image.
    
    Accepts a filename or a data array (optionally with a corresponding WCS
    object). Plots the image with sensible defaults for the color bar, and in
    world coordinates if possible. The colorbar is scaled by the square root of
    the data.
    
    Parameters
    ----------
    data
        A data array to plot, or the name of a FITS file from which to load the
        data and WCS information.
    ax
        A matplotlib ``Axes`` object to use for plotting. Optional.
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
    relative_vmin, relative_vmax : float
        These values multiply the preset values for vmin and vmax
    gamma : float
        The gamma factor to use for scaling the color bar. Defaults to 1/2.2.
    kwargs
        Extra arguments passed to ``imshow``.
    """
    data, header, w = utils.ensure_data(
            data, header=True, wcs=True, wcs_key=wcs_key)
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
        vmin = (COLORBAR_PRESETS[level_preset][detector_preset][0]
                * relative_vmin)
    if vmax == 'auto':
        vmax = (COLORBAR_PRESETS[level_preset][detector_preset][1]
                * relative_vmax)
    
    return _plot_internal(
        data, wcs, ax=ax, draw_constellations=draw_constellations,
                   mark_planets=mark_planets, grid=grid,
                   lat_spacing=lat_spacing, lon_spacing=lon_spacing, vmin=vmin,
                   vmax=vmax, gamma=gamma, imshow_kwargs=kwargs)
    
def _plot_internal(data, wcs, ax=None, draw_constellations=False,
                   mark_planets=False, grid=False, lat_spacing=10,
                   lon_spacing=10, vmin=None, vmax=None, gamma=1/2.2,
                   imshow_kwargs={}):
    if ax is None:
        # No axes object was supplied, so we grab the current axes. (If axes
        # are explicitly provided, we take them as-is.)
        ax = plt.gca()
        if wcs and not isinstance(ax, WCSAxes):
            # We can't apply a WCS projection to existing axes. Instead, we
            # have to destroy and recreate the current axes. We skip that if
            # the axes already are WCSAxes, suggesting that this has been
            # handled already.
            position = ax.get_position().bounds
            ax.remove()
            ax = WCSAxes(plt.gcf(), position, wcs=wcs)
            plt.gcf().add_axes(ax)
            setup_WCS_axes(ax, grid=grid, lat_spacing=lat_spacing,
                    lon_spacing=lon_spacing)
    
    if imshow_kwargs.get('cmap') is None:
        imshow_kwargs['cmap'] = wispr_cmap
    
    if 'origin' not in imshow_kwargs:
        imshow_kwargs['origin'] = 'lower'
    if 'interpolation_stage' not in imshow_kwargs:
        imshow_kwargs['interpolation_stage'] = 'data'
    im = ax.imshow(data,
            norm=matplotlib.colors.PowerNorm(
                gamma=gamma, vmin=vmin, vmax=vmax),
            **imshow_kwargs)
    # Set this image to be the one found by plt.colorbar, for instance. But if
    # this manager attribute is empty, pyplot won't accept it.
    if ax.figure.canvas.manager:
        plt.sca(ax)
        plt.sci(im)
    
    if draw_constellations:
        if not wcs:
            raise ValueError("Cannot draw constellations without a WCS")
        constellations.plot_constellations(wcs)
    
    if mark_planets:
        if not wcs:
            raise ValueError("Cannot mark planets without a WCS")
        if isinstance(mark_planets, bool):
            planet_poses = planets.locate_planets(wcs.wcs.dateavg)
            planet_wcs = wcs
        else:
            planet_wcs, planet_poses = mark_planets
        for planet, pos in zip(planets.planets, planet_poses):
            x, y = planet_wcs.world_to_pixel(pos)
            if (0 < x < data.shape[1] and 0 < y < data.shape[0]
                    and not np.isnan(data[int(y), int(x)])):
                ax.annotate(planet,(x+4, y+1), (2, .3),
                            textcoords='offset fontsize',
                            fontsize='x-small',
                            color='white',
                            arrowprops=dict(
                                edgecolor='white',
                                facecolor='white',
                                arrowstyle='-|>'))
    return im


def setup_WCS_axes(ax, grid=True, lat_spacing=10, lon_spacing=10):
    lon, lat = ax.coords
    lat.set_ticks(np.arange(-90, 90, lat_spacing) * u.degree)
    lon.set_ticks(np.arange(-180, 180, lon_spacing) * u.degree)
    lat.set_major_formatter('dd')
    lon.set_major_formatter('dd')
    if grid:
        if isinstance(grid, bool):
            grid = 0.2
        ax.coords.grid(color='white', alpha=grid)
    if 'helioprojective' in lon.default_label:
        lon.set_axislabel("Helioprojective longitude")
        lat.set_axislabel("Helioprojective latitude")
    elif 'pspframe' in lon.default_label:
        lon.set_axislabel("PSP frame longitude")
        lat.set_axislabel("PSP frame latitude")


def blink(*imgs, vmins=None, vmaxs=None, cmaps=None, interval=500, show=True,
        labels=None, label_kw={}, setup_fcn=None, **kwargs):
    """
    Renders a movie that blinks between several images.
    
    Parameters
    ----------
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
    kwargs
        Passed to `plot_WISPR`
    """
    if not isinstance(vmins, Iterable) or isinstance(vmins, str):
        vmins = [vmins] * len(imgs)
    if not isinstance(vmaxs, Iterable) or isinstance(vmaxs, str):
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
        fig, ax = full_size_plot(imgs[0], **kwargs)
    
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
                    cmap=cmaps[i], **kwargs)
        ax.axis('off')
        if labels is not None:
            plt.text(s=labels[i], zorder=99, **label_kw)

    ani = FuncAnimation(fig, update, frames=range(len(imgs)), interval=interval)
    if show:
        display(HTML(ani.to_html5_video()))
        plt.close()
    else:
        return ani


def plot_orbit(data_dir, between=None, filters=None, clip_to_data=True):
    """
    Plots PSP's orbital path and marks the locations of WISPR exposures.
    
    Only the plane-of-the-ecliptic components of the spacecraft position are
    plotted.
    
    Parameters
    ----------
    data_dir
        A directory containing WISPR images. It is scanned recursively by
        ``utils.collect_files``, and exposure times and spacecraft locations are
        extracted from the headers.
    between, filters
        Parameters passed to ``utils.collect_files`` to select only certain files
        to be plotted.
    clip_to_data : bool
        If ``True``, the portion of the orbit outside of the data collection
        window will not be plotted.
    """
    ifiles_dots, ofiles_dots = utils.collect_files(
            data_dir, between=between, filters=filters)
    
    itimes = [utils.to_timestamp(f) for f in ifiles_dots]
    otimes = [utils.to_timestamp(f) for f in ofiles_dots]
    
    positions, times = planets.trace_psp_orbit(
        utils.extract_encounter_number(ifiles_dots[0]))
    positions = positions.transform_to(PSPOrbitalFrame).cartesian
    
    if clip_to_data:
        tmin = min(itimes[0], otimes[0])
        tmax = min(itimes[-1], otimes[-1])
        # Ensure range we keep fully includes all the images
        dt = times[1] - times[0]
        tmin -= dt
        tmax += dt
        f = (times > tmin) * (times < tmax)
        times = times[f]
        positions = positions[f]
    
    distance_scale = max(positions.x.max() - positions.x.min(),
            positions.y.max() - positions.y.min())
    
    # Plot the s/c trajectory
    plt.plot(positions.x, positions.y)
    
    # Plot the Sun
    plt.scatter(0*u.km, 0*u.km, marker='*', color='k', s=80)
    
    # Calculate locations for the dots
    ixs = np.interp(itimes, times, positions.x)
    oxs = np.interp(otimes, times, positions.x)
    iys = np.interp(itimes, times, positions.y)
    oys = np.interp(otimes, times, positions.y)
    
    if len(iys):
        ithetas = np.arctan2(np.gradient(iys), np.gradient(ixs))
        ithetas += np.pi/2 * u.rad
        ixs += 0.05 * distance_scale * np.cos(ithetas)
        iys += 0.05 * distance_scale * np.sin(ithetas)
    
    if len(oys):
        othetas = np.arctan2(np.gradient(oys), np.gradient(oxs))
        othetas += np.pi/2 * u.rad
        oxs += 0.025 * distance_scale * np.cos(othetas)
        oys += 0.025 * distance_scale * np.sin(othetas)
    
    # Plot the exposure markers with transparency, and plot dummy points
    # without transparency for the plot legend.
    if len(iys):
        plt.scatter(ixs, iys, color='C1', s=5, alpha=0.2)
        plt.scatter(None, None, color='C1', s=5, label='WISPR-I')
    if len(ixs):
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
        
        x = np.interp(t, times, positions.x)
        y = np.interp(t, times, positions.y)
        
        xs.append(x)
        ys.append(y)
        datestring = date.date().isoformat()
        strings.append(datestring)
        date += one_day
    
    xs = u.Quantity(xs)
    ys = u.Quantity(ys)
    
    thetas = np.arctan2(np.gradient(ys), np.gradient(xs))
    thetas -= np.pi/2 * u.rad
    
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
    
    last_xy = np.inf, np.inf
    for x, y, theta, datestring in zip(xs, ys, thetas, strings):
        # Check if the tick marks are too close together. If so, skip labeling
        # this date
        if (np.sqrt((x - last_xy[0])**2 + (y - last_xy[1])**2) 
                < 0.04 * distance_scale):
            continue
        last_xy = x, y
            
        alignment = 'left'
        if theta > np.pi/2 * u.rad or theta < -np.pi/2 * u.rad:
            alignment = 'right'
            theta += np.pi * u.rad
        plt.text(x, y, datestring,
                rotation=theta.to_value(u.deg),
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
    
    Parameters
    ----------
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
    if isinstance(dates, u.Quantity):
        dates = dates.to_value(u.s)
    dates = [datetime.fromtimestamp(d, tz=timezone.utc)
             if np.isfinite(d) else None
             for d in dates]
    dates = np.array(
            [mdates.date2num(d) if d is not None else np.nan for d in dates])
    
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
    
    Parameters
    ----------
    date
        A date to be converted to a matplotlib-friendly format. Should be
        either a format understood by utils.to_timestamp, or a POSIX timestamp.
        Can also be a list of such values.
    
    Returns
    -------
    date
        A date that have been converted to matplotlib's date format, ready for
        use as the x-axis quantity when plotting.
    """
    if isinstance(date, (Iterable, np.ndarray)) and not isinstance(date, str):
        output = [date_to_mdate(x) for x in date]
        if isinstance(date, np.ndarray):
            output = np.array(output)
        return output
    
    if isinstance(date, str):
        date = utils.to_timestamp(date)
    date = datetime.fromtimestamp(date, tz=timezone.utc)
    return mdates.date2num(date)


def browse_frames(frames, *args, **kwargs):
    """
    Allows interactive browsing of a list of frames
    
    Parameters
    ----------
    frames : Iterable
        A list of anything that can be the first argument to `plot_WISPR`
    args, kwargs
        Additional arguments to pass to plot_WISPR
    """
    def plot(f):
        plot_WISPR(frames[f], *args, **kwargs)
    interact(plot, f=(0, len(frames)))
