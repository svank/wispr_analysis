import copy
from datetime import datetime, timezone
import functools
import gc
from itertools import repeat
import itertools
from math import floor
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import warnings

from astropy.io import fits
from astropy import constants
import astropy.units as u
from astropy.wcs import WCS
from IPython.display import display, Video
import matplotlib
import numpy as np
import reproject
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from . import composites
from . import constellations
from . import data_cleaning
from . import planets
from . import plot_utils
from . import projections
from . import utils


cmap = copy.copy(matplotlib.cm.Greys_r)
cmap.set_bad('black')


def make_WISPR_video(data_dir, between=(None, None), filters=None,
        trim_threshold=12*60*60, image_trim=True,
        level_preset=None, vmin=None, vmax=None, gamma=1/2.2,
        duration=15, fps=20, n_procs=os.cpu_count(),
        remove_debris=True, debris_mask_dir=None, overlay_coords=True,
        overlay_celest=False, save_location=None, mark_planets=False,
        align=True, draw_constellations=False):
    """
    Renders a video of a WISPR data sequence, in a composite field of view.
    
    The video plays back at a constant rate in time (as opposed to showing each
    WISPR image for one frame, for instance). This gives the best
    representation of the physical speeds of every feature in the image, but
    can produce jittery videos when the WISPR imaging cadence drops. At each
    point in time, the most recent preceding images are shown for the inner
    and outer imagers (or no image is shown if none is available or the most
    recent image is over a day before the frame's timestamp).
    
    The rendered video is displayed in the Jupyter environment by default, but
    can be saved to a file instead.
    
    Parameters
    ----------
    data_dir : str
        Directory containing WISPR fits files. Will be scanned by
        ``utils.collect_files``
    between : tuple
        A beginning and end timestamp of the format 'YYYYMMDDTHHMMSS'. Only the
        files between these timestamps will be included in the video. Either or
        both timestamp can be `None`, to place no limit on the beginning or end
        of the video.
    filters : tuple
        Additional filters to pass to ``utils.collect_files``.
    trim_threshold : float
        After filtering is done with the ``between`` values, the video start and
        end times are determined by scanning the time gaps between successive
        images. Starting from the midpoint of the data sequence, the time range
        being shown expands out in both directions to the first time gap that
        exceeds ``trim_thresholds`` (a value in seconds). Set to None to
        disable.
    level_preset : str or int
        The colorbar range is set with default values appropriate to L2 or L3
        data. This is autodetected from the files but can be overridden with
        this argument.
    vmin, vmax : float
        Colorbar ranges can be set explicitly to override the defaults
    duration : float
        Duration of the output video in seconds. If None, each (inner-FOV)
        frame is shown for an equal amount of time.
    fps : float
        Number of frames per second in the video
    remove_debris : boolean
        Whether to enable the debris streak removal algorithm
    debris_mask_dir : str
        Path to a directory containing pre-computed debris streak masks. If not
        provided and ``remove_debris`` is ``True``, streak detection will be
        run.
    overlay_coords : boolean
        Whether to show a helioprojective coordinate grid
    overlay_celest : boolean
        Whether to show an RA/Dec grid as well as an HP grid
    save_location : str
        If given, a file path at which to save the video. Otherwise, it is
        displayed in the Jupyter environment.
    mark_planets : boolean
        Whether to label planets in the field of view. Requires SPICE kernels
        to be loadable. Call ``planets.load_kernels`` to provide a location for
        the kernels.
    align : boolean
        Whether to reproject the images into a common frame. If False, it is
        assumed that all images are from the same detector.
    """
    if mark_planets:
        # Is a no-op if kernels are already loaded
        planets.load_kernels()
    files = utils.collect_files(
            data_dir, separate_detectors=align, include_sortkey=True,
            include_headers=True, between=between, filters=filters)
    if align:
        i_files, o_files = files
    else:
        i_files = files
        o_files = []
    i_files = [(utils.to_timestamp(f[0]), f[1], f[2]) for f in i_files]
    o_files = [(utils.to_timestamp(f[0]), f[1], f[2]) for f in o_files]
    i_tstamps = [f[0] for f in i_files]
    o_tstamps = [f[0] for f in o_files]
    tstamps = sorted(itertools.chain(i_tstamps, o_tstamps))
    
    # Do this before setting the time range, so the full s/c trajectory is
    # visible in the inset plot
    try:
        path_times, path_positions, _ = utils.get_PSP_path_from_headers(
                [v[-1] for v in sorted(itertools.chain(i_files, o_files))])
    except KeyError as e:
        if "Keyword 'DATE-AVG' not found" in str(e):
            warnings.warn("Could not load orbital path")
            path_times, path_positions = None, None
        else:
            raise
    
    if trim_threshold is not None:
        # Set our time range by computing delta-ts and keeping the middle
        # segment, trimming from the start and end any periods with more than
        # some threshold between frames.
        t_deltas = np.diff(tstamps)
        midpoint = len(t_deltas) // 2
        i = np.nonzero(t_deltas[:midpoint] > trim_threshold)[0]
        if len(i):
            i = i[-1]
        else:
            i = 0
        t_start = tstamps[i+1]
        
        j = np.nonzero(t_deltas[midpoint:] > trim_threshold)[0]
        if len(j):
            j = j[0] + midpoint
        else:
            j = len(tstamps) - 1
        t_end = tstamps[j]
    else:
        t_start = tstamps[0]
        t_end = tstamps[-1]
    
    i_files = [v for v in i_files if t_start <= v[0] <= t_end]
    o_files = [v for v in o_files if t_start <= v[0] <= t_end]
    i_tstamps = [f[0] for f in i_files]
    o_tstamps = [f[0] for f in o_files]
    # Remove the stale `tstamps` data
    del tstamps
    
    if align:
        wcsh, naxis1, naxis2 = composites.gen_header(
                i_files[len(i_files)//2][1], o_files[len(o_files)//2][1])
        bounds = composites.find_collective_bounds(
                ([v[-2] for v in i_files[::3]], [v[-2] for v in o_files[::3]]),
                wcsh, ((33, 40, 42, 39), (20, 25, 26, 31)))
    else:
        wcsh, naxis1, naxis2, bounds = None, None, None, None
    
    if duration is None:
        frames = np.asarray(i_tstamps)
    else:
        frames = np.linspace(t_start, t_end, fps*duration)
    
    images = []
    # Determine the correct pair of images for each timestep
    for t in frames:
        i_cut = np.nonzero(
                (i_tstamps <= t) * (np.abs(i_tstamps - t) < 24 * 60 * 60))[0]
        if len(i_cut):
            i = i_cut[-1]
        else:
            i = None
        
        j_cut = np.nonzero(
                (o_tstamps <= t) * (np.abs(o_tstamps - t) < 24 * 60 * 60))[0]
        if len(j_cut):
            j = j_cut[-1]
        else:
            j = None
        images.append((i, j, t))

    # Sometimes the same image pair is chosen for multiple time steps---the
    # output will be the same except for the overlaid time stamp, so we can
    # render multiple frames from one reprojection call and save time. Convert
    # our list to a list of (unique_image_pair, list_of_timesteps)
    images = [(pair, [d[-1] for d in data])
              for pair, data in itertools.groupby(images, lambda x: x[0:2])]
    
    level_preset = plot_utils.parse_level_preset(level_preset, i_files[0][2])
    
    colorbar_data = plot_utils.COLORBAR_PRESETS[level_preset]
    
    if vmin is None:
        vmin = min(*[d[0] for d in colorbar_data.values()])
    if vmax is None:
        vmax = max(*[d[1] for d in colorbar_data.values()])
    
    if save_location is not None:
        save_location = os.path.expanduser(save_location)
    
    if remove_debris and debris_mask_dir is not None:
        masks_i = data_cleaning.find_mask(
                debris_mask_dir, [f[1] for f in i_files])
        masks_o = data_cleaning.find_mask(
                debris_mask_dir, [f[1] for f in o_files])
    else:
        masks_i = [None] * len(i_files)
        masks_o = [None] * len(o_files)
    
    if draw_constellations:
        # Do this once before we start multiprocessing---the result is cached
        constellations.load_constellation_data()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        def arguments():
            for (i, j), timesteps in images:
                args = dict(
                    timesteps=timesteps, remove_debris=remove_debris,
                    bounds=bounds, wcsh=wcsh, naxis1=naxis1, naxis2=naxis2,
                    overlay_coords=overlay_coords, align=align,
                    overlay_celest=overlay_celest, save_location=save_location,
                    tmpdir=tmpdir, vmin=vmin, vmax=vmax, gamma=gamma,
                    path_positions=path_positions,
                    path_times=path_times, ifile=None, next_ifile=None,
                    prev_ifile=None, ihdr=None, ofile=None, next_ofile=None,
                    prev_ofile=None, ohdr=None, imask=None, omask=None,
                    fallback_ifile=i_files[0][1] if align else None,
                    fallback_ofile=o_files[0][1] if align else None,
                    draw_constellations=draw_constellations,
                    mark_planets=mark_planets, image_trim=image_trim)
                if i is not None:
                    args['ifile'] = i_files[i][1]
                    if 0 < i < len(i_files) - 1:
                        args['next_ifile'] = i_files[i+1][1]
                        args['prev_ifile'] = i_files[i-1][1]
                    args['ihdr'] = i_files[i][2]
                    args['imask'] = masks_i[i]
                if j is not None:
                    args['ofile'] = o_files[j][1]
                    if 0 < j < len(o_files) - 1:
                        args['next_ofile'] = o_files[j+1][1]
                        args['prev_ofile'] = o_files[j-1][1]
                    args['ohdr'] = o_files[j][2]
                    args['omask'] = masks_o[j]
                for k in ['ihdr', 'ohdr']:
                    # CompImageHeaders currently don't unpickle, so convert
                    # them to normal Headers
                    if isinstance(
                            args[k], fits.hdu.compressed.CompImageHeader):
                        args[k] = fits.Header.fromstring(
                                args[k].tostring())
                yield args
        process_map(_draw_WISPR_video_frame,
                    arguments(), total=len(images), max_workers=n_procs)
        video_file = os.path.join(tmpdir, 'out.mp4')
        subprocess.call(
                f"ffmpeg -loglevel error -r {fps} "
                f"-pattern_type glob -i '{tmpdir}/*.png' -c:v libx264 "
                 "-pix_fmt yuv420p -x264-params keyint=30 "
                f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {video_file}",
                shell=True)
        if save_location is not None:
            shutil.move(video_file, save_location)
        else:
            display(Video(video_file, embed=True,
                html_attributes="controls loop"))


def wrap_with_gc(function):
    """
    Wraps a function and manually runs garbage collection after every execution

    We seem to get slowly-increasing memory usage when making a long-ish video,
    which becomes significant with 20 worker processes, and running the GC
    after every frame seems to help keep that in check.
    """
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        ret = function(*args, **kwargs)
        del args, kwargs
        gc.collect()
        return ret
    return wrapped


@wrap_with_gc
def _draw_WISPR_video_frame(data):
    if data['ifile'] is None:
        input_i = None
    elif data['remove_debris'] and data['next_ifile']:
        input_i = data_cleaning.dust_streak_filter(
                data['prev_ifile'], data['ifile'], data['next_ifile'],
                sliding_window_stride=3, precomputed_mask=data['imask'],
                radec=False)
        input_i = (input_i, data['ifile'])
    else:
        input_i = data['ifile']
       
    if data['ofile'] is None:
        input_o = None
    elif data['remove_debris'] and data['next_ofile']:
        input_o = data_cleaning.dust_streak_filter(
                data['prev_ofile'], data['ofile'], data['next_ofile'],
                sliding_window_stride=3, precomputed_mask=data['omask'],
                radec=False)
        input_o = (input_o, data['ofile'])
    else:
        input_o = data['ofile']
    
    if data['align']:
        if data['image_trim']:
            image_trim = [[20, 25, 1, 1], [33, 40, 42, 39]]
        else:
            image_trim = None
        c, wcs_plot = composites.gen_composite(
            # Even if we're blanking one of the images, a header is still
            # needed (for now...)
            input_i if input_i is not None else data['fallback_ifile'],
            input_o if input_o is not None else data['fallback_ofile'],
            bounds=data['bounds'],
            wcsh=data['wcsh'], naxis1=data['naxis1'], naxis2=data['naxis2'],
            blank_i=(input_i is None), blank_o=(input_o is None),
            image_trim=image_trim)
    else:
        image_trim = None
        if isinstance(input_i, tuple):
            c = input_i[0]
            wcs_plot = WCS(input_i[1])
        else:
            with utils.ignore_fits_warnings(), fits.open(input_i) as hdul:
                hdu = 1 if hdul[0].data is None else 0
                c = hdul[hdu].data
                wcs_plot = WCS(hdul[hdu].header, hdul)
    
    if data['overlay_celest']:
        # Determine which input image is closest in time
        if data['ifile'] is None:
            dt_i = np.inf
        else:
            dt_i = np.abs(data['timesteps'][0] - utils.to_timestamp(data['ifile']))
        if data['ofile'] is None:
            dt_o = np.inf
        else:
            dt_o = np.abs(data['timesteps'][0] - utils.to_timestamp(data['ofile']))
        if dt_i < dt_o:
            hdr = data['ihdr']
        else:
            hdr = data['ohdr']
        wcs_ra = projections.produce_radec_for_hp_wcs(wcs_plot, ref_hdr=hdr)
    
    if data['mark_planets']:
        with utils.ignore_fits_warnings():
            if data['ihdr'] is not None:
                with (utils.ignore_fits_warnings(),
                        fits.open(data['ifile']) as hdul):
                    hdu = 1 if hdul[0].data is None else 0
                    planet_poses_i = planets.locate_planets(hdul[hdu].header)
                    wcs_i = WCS(hdul[hdu].header, hdul)
                if image_trim is not None:
                    wcs_i = wcs_i[
                            image_trim[0][2]:-image_trim[0][3],
                            image_trim[0][0]:-image_trim[0][1]]
            else:
                planet_poses_i = [None] * 8
                wcs_i = None
            if data['ohdr'] is not None:
                with (utils.ignore_fits_warnings(),
                        fits.open(data['ofile']) as hdul):
                    hdu = 1 if hdul[0].data is None else 0
                    planet_poses_o = planets.locate_planets(hdul[hdu].header)
                    wcs_o = WCS(hdul[hdu].header, hdul)
                if image_trim is not None:
                    wcs_o = wcs_o[
                            image_trim[1][2]:-image_trim[1][3],
                            image_trim[1][0]:-image_trim[1][1]]
            else:
                planet_poses_o = [None] * 8
                wcs_o = None
    
    composites.set_wcs_observer_details(wcs_plot, wcs_i, wcs_o)
    
    with matplotlib.style.context('dark_background'):
        for t in data['timesteps']:
            # Determine which input image is closest in time
            if data['ifile'] is None:
                dt_i = np.inf
            else:
                dt_i = np.abs(t - utils.to_timestamp(data['ifile']))
            if data['ofile'] is None:
                dt_o = np.inf
            else:
                dt_o = np.abs(t - utils.to_timestamp(data['ofile']))
            if dt_i < dt_o:
                hdr = data['ihdr']
            else:
                hdr = data['ohdr']
            r, theta = utils.load_orbit_plane_rtheta(hdr)
            r = r[0]; theta = theta[0]
            
            fig = matplotlib.figure.Figure(
                figsize=(10, 7.5),
                dpi=250 if data['save_location'] else 150)
            if data['overlay_coords']:
                ax = fig.add_subplot(111, projection=wcs_plot)
                plot_utils.setup_WCS_axes(ax)
                ax.set_xlabel("Helioprojective Longitude")
                ax.set_ylabel("Helioprojective Latitude")
            else:
                ax = fig.add_subplot(111)

            ax.imshow(c, cmap=cmap, origin='lower',
                      norm=matplotlib.colors.PowerNorm(
                          gamma=data['gamma'],
                          vmin=data['vmin'], vmax=data['vmax']))
            timestamp = datetime.fromtimestamp(t, tz=timezone.utc)
            ax.text(20, 20,
                    timestamp.strftime("%Y-%m-%d, %H:%M"),
                    color='white')

            fig.subplots_adjust(
                top=0.96, bottom=0.10,
                left=0.05, right=0.98)
            
            if data['overlay_celest']:
                ax.coords[0].set_ticks_position('b')
                ax.coords[1].set_ticks_position('l')
                
                overlay = ax.get_coords_overlay(wcs_ra)
                overlay.grid(color='gold', alpha=.2)
                overlay[0].set_ticks(spacing=360/24*u.deg)
                overlay[1].set_ticks(spacing=10*u.deg)
                overlay[0].set_ticks_position('t')
                overlay[1].set_ticks_position('r')
                overlay[0].set_axislabel("Right Ascension")
                overlay[1].set_axislabel("Declination")
                
                fig.subplots_adjust(
                    top=0.90, bottom=0.10,
                    left=0.05, right=0.95)
            
            if data['draw_constellations']:
                constellations.plot_constellations(wcs_plot, ax=ax)
            
            main_ax_pos = ax.get_position()
            if data['path_positions'] is not None:
                ax_orbit = fig.add_axes(
                        (.01 + main_ax_pos.xmin,
                         .04 + main_ax_pos.ymin, .12, .12))
                draw_overhead_map(ax_orbit, t, data['path_positions'],
                        data['path_times'])
            
            fig.text(main_ax_pos.xmin + .13,
                     main_ax_pos.ymin + .065,
                     f"r = {(r*u.m / constants.R_sun).si.value:.1f}",
                     color='white')
            fig.text(main_ax_pos.xmin + .19,
                     main_ax_pos.ymin + .065,
                     f"R$_\odot$",
                     color='white')
            fig.text(main_ax_pos.xmin + .13,
                     main_ax_pos.ymin + .04,
                     f"      {(r * u.m).to(u.AU).value:.2f}",
                     color='white')
            fig.text(main_ax_pos.xmin + .19,
                     main_ax_pos.ymin + .04,
                     f"AU",
                     color='white')
            fig.text(main_ax_pos.xmin + .13,
                     main_ax_pos.ymin + .095,
                     f"$\\theta$ = {(theta * 180 / np.pi) % 360:.1f}",
                     color='white')
            fig.text(main_ax_pos.xmin + .196,
                     main_ax_pos.ymin + .095,
                     f"$^\circ$",
                     color='white')
            
            if data['mark_planets']:
                for planet_name, pos_i, pos_o in zip(
                        planets.planets, planet_poses_i, planet_poses_o):
                    if pos_i is None:
                        xi, yi = -1, -1
                    else:
                        xi, yi = wcs_i.world_to_pixel(pos_i)
                    if pos_o is None:
                        xo, yo = -1, -1
                    else:
                        xo, yo = wcs_o.world_to_pixel(pos_o)
                    pos = None
                    if (0 < xi < wcs_i.pixel_shape[0]
                            and 0 < yi < wcs_i.pixel_shape[1]):
                        pos = pos_i
                    elif (0 < xo < wcs_o.pixel_shape[0]
                            and 0 < yo < wcs_o.pixel_shape[1]):
                        pos = pos_o
                    if pos is not None:
                        x, y = wcs_plot.world_to_pixel(pos)
                        # Just in case the composite is being cut off
                        if 0 < x < c.shape[1] and 0 < y < c.shape[0]:
                            ax.annotate(planet_name, (x+7, y),
                                    (x+50, y+10), color='.9',
                                    fontsize='x-small',
                                    arrowprops=dict(
                                        edgecolor='.7',
                                        arrowstyle='->',
                                        mutation_scale=5))

            fig.savefig(f"{data['tmpdir']}/{t:035.20f}.png")


def animate_pointing(data_dir, between=(None, None), show=True, fps=30,
        file_load_interval=1, plot_interval=3):
    """
    Renders a full-sky map showing the WISPR FOV over an encounter as a video.
    
    At each time step, the current FOV for the two imagers are shown, and
    previous FOVs are shown with an opacity decreasing with time. The Sun's
    location is also shown.
    
    This function attempts to use all CPU cores, and splits them between
    separate queues for reprojecting images into the all-sky frame and
    rendering video frames. This is because each video frame draws on at least
    the last few exposures, to avoid flickering. These separate queues are not
    carefully balanced, so it's possible to see incomplete usage of all cores
    or increasing RAM usage if the second queue can't keep up.
    
    Parameters
    ----------
    data_dir : str
        Directory containing WISPR fits files. Will be scanned by
        ``utils.collect_files``
    between : tuple
        A beginning and end timestamp of the format 'YYYYMMDDTHHMMSS'. Only the
        files between these timestamps will be included in the video. Either or
        both timestamp can be ``None``, to place no limit on the beginning or end
        of the video.
    show : boolean
        When ``True``, the rendered video is shown in the Jupyter environment.
        Otherwise, an IPython ``Video`` object is returned.
    fps : float
        Number of frames per second in the video
    file_load_interval : int
        Only the first out of every ``file_load_interval`` files is considered.
        This option is mostly intended to speed up rendering during
        development, and no care is taken to ensure this cooperates well with
        the interleaving of images from the two cameras.
    plot_interval : int
        Only the first out of every ``plot_interval`` considered files is
        rendered in the video.
    """
    files = utils.collect_files(
            data_dir, separate_detectors=False, include_sortkey=True,
            order='DATE-AVG', include_headers=True, between=between)
    path_times, path_positions, _ = utils.get_PSP_path_from_headers(
        [v[-1] for v in files])
    tstamps = [f[0] for f in files]
    files = [f[1] for f in files]
    
    cdelt = 0.15
    shape = [int(floor(180/cdelt)), int(floor(360/cdelt))]
    starfield_wcs = WCS(naxis=2)
    crpix = [shape[1]/2+1, shape[0]/2+1]
    starfield_wcs.wcs.crpix = crpix
    starfield_wcs.wcs.crval = 180, 0
    starfield_wcs.wcs.cdelt = cdelt, cdelt
    starfield_wcs.wcs.ctype = 'RA---MOL', 'DEC--MOL'
    starfield_wcs.wcs.cunit = 'deg', 'deg'
    
    min_val = 0.0
    imap = np.full(shape, 0.0)
    omap = np.full(shape, 0.0)
    emap = np.full(shape, min_val)
    last_few_i = []
    last_few_o = []
    zeros = np.zeros_like(imap, dtype=bool)
    
    last_t_i = np.nan
    last_t_o = np.nan
    cadence_i = np.nan
    cadence_o = np.nan
    last_suns = [np.array((np.nan, np.nan))] * 2
    last_sun_ts = [np.nan] * 2
    
    files = files[::file_load_interval]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        n_cpus = os.cpu_count()
        with (warnings.catch_warnings(),
                multiprocessing.Pool(n_cpus // 2) as p_reproject,
                multiprocessing.Pool(n_cpus // 2) as p_plot,
                tqdm(total=len(files)) as pbar):
            warnings.filterwarnings(action='ignore',
                    message='.*has more axes.*')
            for i, (t, fname, output) in enumerate(
                    zip(tstamps,
                        files,
                        p_reproject.imap(process_one_file_for_pointing,
                            zip(files,
                                repeat(starfield_wcs),
                                repeat(imap.shape))))):
                output, wcs, wcs_celest = output
                pbar.update(1)
                
                t = utils.to_timestamp(t, as_datetime=True)
                ts = t.timestamp()
                # Decay the currently-lit pixels in the FOV maps
                imap -= .04
                omap -= .04
                imap[imap < min_val] = min_val
                omap[omap < min_val] = min_val
                if fname.split('_')[-1][0] == '1':
                    ioutput = output
                    ooutput = zeros
                    cadence_i = ts - last_t_i
                    last_t_i = ts
                else:
                    ioutput = zeros
                    ooutput = output
                    cadence_o = ts - last_t_o
                    last_t_o = ts
                
                last_few_i.append(ioutput)
                if len(last_few_i) > 8:
                    last_few_i.pop(0)
                last_few_o.append(ooutput)
                if len(last_few_o) > 8:
                    last_few_o.pop(0)
                
                imap[np.sum(last_few_i, axis=0, dtype=bool)] = 1
                omap[np.sum(last_few_o, axis=0, dtype=bool)] = 1
                
                sun_coords = wcs.world_to_pixel(0*u.deg, 0*u.deg)
                sun_coords = wcs_celest.pixel_to_world(*sun_coords)
                sun_coords = starfield_wcs.world_to_pixel(sun_coords)
                if np.any(np.isnan(sun_coords)):
                    # Interpolate a Sun location from the last two locations
                    # and the current time.
                    sun_coords = (last_suns[1] +
                            (last_suns[1] - last_suns[0])
                            * (ts - last_sun_ts[1])
                            / (last_sun_ts[1] - last_sun_ts[0]))
                else:
                    last_suns.pop(0)
                    last_sun_ts.pop(0)
                    last_suns.append(np.array(sun_coords))
                    last_sun_ts.append(ts)
                
                if i % plot_interval:
                    continue
                p_plot.apply_async(plot_one_frame_for_pointing,
                        args=(imap.copy(), omap.copy(), starfield_wcs, t, ts,
                            fname, sun_coords, path_positions, path_times,
                            tmpdir, cadence_i, cadence_o))
            p_plot.close()
            p_plot.join()
        
        subprocess.call(
                f"ffmpeg -loglevel error -r {fps} -pattern_type glob"
                f" -i '{tmpdir}/*.png' -c:v libx264 -pix_fmt yuv420p"
                 " -x264-params keyint=30"
                f" -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {tmpdir}/out.mp4",
                shell=True)
        video = Video(f"{tmpdir}/out.mp4",
                embed=True, html_attributes="controls loop")
        if show:
            display(video)
        else:
            return video


def process_one_file_for_pointing(inputs):
    fname, starfield_wcs, shape = inputs
    with utils.ignore_fits_warnings(), fits.open(fname) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header, hdul)
        wcs_celest = WCS(hdul[0].header, hdul, key='A')
    
    data = np.ones_like(data)
    
    return (np.isfinite(
            reproject.reproject_adaptive((data, wcs_celest), starfield_wcs,
                shape, return_footprint=False, roundtrip_coords=False)),
            wcs, wcs_celest)


def plot_one_frame_for_pointing(imap, omap, starfield_wcs, t, ts, fname,
        sun_coords, path_positions, path_times, tmpdir, cadence_i, cadence_o):
    fig = matplotlib.figure.Figure(figsize=(10, 6), dpi=140)
    ax = fig.add_subplot(111, projection=starfield_wcs)
    
    rcolor = np.array((.88, .08, .08))[None, None, :]
    bcolor = np.array((0.001, .48, 1))[None, None, :]
    map = rcolor * imap[:, :, None] + bcolor * omap[:, :, None]
    map = np.where(map > 1, 1, map)
    map = np.where(map == 0, .05, map)
    im = ax.imshow(map, origin='lower')
    text = t.strftime("%Y-%m-%d, %H:%M")
    text += f", Cadence: I {cadence_i:.0f} s / O {cadence_o:.0f} s"
    ax.text(20, 20, text, color='white')
    
    ax.text(1200, 20, os.path.basename(fname), color='white')
    
    ax.scatter(*sun_coords, s=50, color='yellow')
    
    fig.subplots_adjust(top=0.96, bottom=0.10,
            left=0.09, right=0.98)
    lon, lat = ax.coords
    ax.set_xlabel("Right Ascension")
    ax.set_ylabel("Declication")
    ax.coords.grid(color='white', alpha=0.2)
    
    with matplotlib.style.context('dark_background'):
        ax_orbit = fig.add_axes((.85, .73, .12, .12))
        draw_overhead_map(ax_orbit, ts, path_positions, path_times)

    fig.savefig(f"{tmpdir}/{ts:035.20f}.png")


def draw_overhead_map(ax_orbit, t, path_positions, path_times):
    """
    Draws a small plot with an overhead view of PSP's position and trajectory
    """
    ax_orbit.margins(.1, .1)
    ax_orbit.set_aspect('equal')
    ax_orbit.scatter(
            path_positions[:, 0],
            path_positions[:, 1],
            s=1, color='.4', marker='.')
    sc_x = np.interp(t, path_times, path_positions[:, 0])
    sc_y = np.interp(t, path_times, path_positions[:, 1])
    ax_orbit.scatter(sc_x, sc_y, s=6, color='1')
    ax_orbit.scatter(0, 0, color='yellow')
    ax_orbit.xaxis.set_visible(False)
    ax_orbit.yaxis.set_visible(False)
    ax_orbit.set_title("S/C position", fontdict={'fontsize': 9})
    for spine in ax_orbit.spines.values():
        spine.set_color('.4')


def generic_make_video(frame_renderer, *arg_list, parallel=True, fps=20,
        save_to=None):
    """
    Helper function for generating a video
    
    Calls a function repeatedly to render each frame, then uses ffmpeg to
    combine the frames into a video, which is displayed in the current Jupyter
    notebook or saved to disk.
    
    Note: In some cases (particularly when using the parallel mode with many
    cores), it may be helpful to apply the `wrap_with_gc` decorator to the
    frame-rendering function.

    Parameters
    ----------
    frame_renderer : function
        Function than draws a single frame. Must accept as arguments the file
        name to which the frame should be saved as a PNG file, and then all
        other arguments provided in ``arg_list``.
    arg_list
        Arguments to be passed to ``frame_renderer``. Each provided value can
        be a single non-iterable item (including strings), in which case that
        value will be repeated for all function calls, or an iterable of
        arguments, one per function call. The required output filename will be
        prepended to each tuple of arguments.
    parallel : boolean
        If ``True``, frames will be rendered in parallel. If an integer, sets
        the maximum number of worker processes.
    fps : int
        The frames-per-second to use for the final video.
    save_to : str
        An output path where the video should be saved. If None, the video is
        displayed in Jupyter.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        def output_names():
            i = 0
            while True:
                yield f"{tmpdir}/{i:0>10d}.png"
                i += 1
        
        ready_arg_list = [output_names()]
        n = np.inf
        have_iterable = False
        for arg in arg_list:
            if hasattr(arg, "__iter__") and not isinstance(arg, str):
                have_iterable = True
                ready_arg_list.append(arg)
                try:
                    n = min(n, len(arg))
                except TypeError:
                    # Iterable that doesn't expose a length
                    pass
            else:
                ready_arg_list.append(repeat(arg))
        if n == np.inf:
            n = None
        if not have_iterable:
            raise ValueError(
                    "At least one iterable of arguments must be provided")
        
        if parallel:
            if type(parallel) is int:
                max_workers = parallel
            else:
                max_workers = os.cpu_count()
            process_map(
                    frame_renderer, *ready_arg_list,
                    total=n, max_workers=max_workers)
        else:
            for args in tqdm(zip(ready_arg_list), total=n):
                frame_renderer(*args)
        
        video_file = os.path.join(tmpdir, 'out.mp4')
        subprocess.call(
                f"ffmpeg -loglevel error -r {fps} "
                f"-pattern_type glob -i '{tmpdir}/*.png' -c:v libx264 "
                 "-pix_fmt yuv420p -x264-params keyint=30 "
                f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {video_file}",
                shell=True)
        if save_to is None:
            display(Video(video_file, embed=True,
                html_attributes="controls loop"))
        else:
            shutil.move(video_file, os.path.expanduser(save_to))

