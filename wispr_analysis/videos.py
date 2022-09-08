import copy
from datetime import datetime
from itertools import repeat
import itertools
from math import floor
import multiprocessing
import os
import shutil
import tempfile
import warnings

from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from IPython.display import display, Video
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import reproject
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from . import composites
from . import data_cleaning
from . import plot_utils
from . import projections
from . import utils


cmap = copy.copy(plt.cm.Greys_r)
cmap.set_bad('black')

    
def make_WISPR_video(data_dir, between=(None, None), trim_threshold=12*60*60,
        level_preset=None, vmin=None, vmax=None, duration=15, fps=20,
        destreak=True, overlay_celest=False, save_location=None):
    """
    Renders a video of a WISPR data sequence, in a composite field of view.
    
    The video plays back at a constant rate in time (as opposed to showing each
    WISPR image for one frame, for instance). This gives the best
    representation of the physical speeds of every feature in the image, but
    can produce jittery videos when the WISPR imaging cadence drops. At each
    point in time, the most recent preceeding images are shown for the inner
    and outer imagers (or no image is shown if none is available or the most
    recent image is over a day before the frame's timestamp).
    
    The rendered video is displayed in the Jupyter environment by default, but
    can be saved to a file instread.
    
    Arguments
    ---------
    data_dir : str
        Directory containing WISPR fits files. Will be scanned by
        `utils.collect_files`
    between : tuple
        A beginning and end timestamp of the format 'YYYYMMDDTHHMMSS'. Only the
        files between these timestamps will be included in the video. Either or
        both timestamp can be `None`, to place no limit on the beginning or end
        of the video.
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
        Duration of the output video in seconds
    fps : float
        Number of frames per second in the video
    destreak : boolean
        Whether to enable the debris streak removal algorithm
    save_location : str
        If given, a file path at which to save the video. Otherwise, it is
        displayed in the Jupyter environment.
    """
    i_files, o_files = utils.collect_files(
            data_dir, separate_detectors=True, include_sortkey=True,
            include_headers=True, between=between)
    i_files = [(utils.to_timestamp(f[0]), f[1], f[2]) for f in i_files]
    o_files = [(utils.to_timestamp(f[0]), f[1], f[2]) for f in o_files]
    i_tstamps = [f[0] for f in i_files]
    o_tstamps = [f[0] for f in o_files]
    tstamps = sorted(itertools.chain(i_tstamps, o_tstamps))   
    
    # Do this before setting the time range, so the full s/c trajectory is
    # visible in the inset plot
    path_times, path_positions, _ = utils.get_PSP_path_from_headers(
                    [v[-1] for v in sorted(itertools.chain(i_files, o_files))])
    
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

    i_files = [v for v in i_files if t_start <= v[0] <= t_end]
    o_files = [v for v in o_files if t_start <= v[0] <= t_end]
    i_tstamps = [f[0] for f in i_files]
    o_tstamps = [f[0] for f in o_files]
    # Remove the stale `tstamps` data
    del tstamps
    
    wcsh, naxis1, naxis2 = composites.gen_header(
            i_files[len(i_files)//2][2], o_files[len(o_files)//2][2])
    bounds = composites.find_collective_bounds(
            ([v[-1] for v in i_files[::3]], [v[-1] for v in o_files[::3]]),
            wcsh, ((33, 40, 42, 39), (20, 25, 26, 31)))

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
    # output will be the same execpt for the overlaid time stamp, so we can
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
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Rather than explicitly passing ~15 different variables to the
        # per-frame plotting function, let's be hacky and just send the whole
        # dictionary of local variables, which is mostly things we need to send
        # anyway.
        arguments = zip(images, repeat(locals()))
        process_map(draw_WISPR_video_frame, arguments, total=len(images))
        video_file = os.path.join(tmpdir, 'out.mp4')
        os.system(f"ffmpeg -loglevel error -r {fps} -pattern_type glob"
                  f" -i '{tmpdir}/*.png' -c:v libx264 -pix_fmt yuv420p"
                  " -x264-params keyint=30"
                  f" -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {video_file}")
        if save_location is not None:
            shutil.move(video_file, save_location)
        else:
            display(Video(video_file, embed=True,
                html_attributes="controls loop"))


def draw_WISPR_video_frame(data):
    (pair, timesteps), local_vars = data
    # `locals` is not updatable (except at the lop level where locals() is just
    # globals()), so we have to update globals() instead. Since we're
    # multiprocessing, those global variables will be thrown away when this
    # process ends.
    globals().update(local_vars)
    i, j = pair
    
    if i is None:
        input_i = None
    else:
        if destreak and 0 < i < len(i_files) - 1:
            input_i = data_cleaning.dust_streak_filter(
                    i_files[i-1][1], i_files[i][1], i_files[i+1][1],
                    sliding_window_stride=3)
            input_i = (input_i, i_files[i][2])
        else:
            input_i = i_files[i][1]
    
    if j is None:
        input_o = None
    else:
        if destreak and 0 < j < len(o_files) - 1:
            input_o = data_cleaning.dust_streak_filter(
                    o_files[j-1][1], o_files[j][1], o_files[j+1][1],
                    sliding_window_stride=3)
            input_o = (input_o, o_files[j][2])
        else:
            input_o = o_files[j][1]
    
    c, wcs_plot = composites.gen_composite(
        # Even if we're blanking one of the images, a header is still
        # needed (for now...)
        input_i if input_i is not None else i_files[0][1],
        input_o if input_o is not None else o_files[0][1],
        bounds=bounds,
        wcsh=wcsh, naxis1=naxis1, naxis2=naxis2,
        blank_i=(input_i is None), blank_o=(input_o is None))
    
    if overlay_celest:
        # Determine which input image is closest in time
        if i is None:
            dt_i = np.inf
        else:
            dt_i = np.abs(timesteps[0] - i_files[i][0])
        if j is None:
            dt_o = np.inf
        else:
            dt_o = np.abs(timesteps[0] - o_files[j][0])
        if dt_i < dt_o:
            hdr = i_files[i][2]
        else:
            hdr = o_files[j][2]
        wcs_ra = projections.produce_radec_for_hp_wcs(wcs_plot, ref_hdr=hdr)
    
    with plt.style.context('dark_background'):
        for t in timesteps:
            fig = plt.figure(figsize=(10, 7.5),
                    dpi=250 if save_location else 150)
            ax = fig.add_subplot(111, projection=wcs_plot)

            im = ax.imshow(c, cmap=cmap, origin='lower',
                           norm=matplotlib.colors.PowerNorm(
                               gamma=0.5, vmin=0, vmax=vmax))
            text = ax.text(20, 20,
                    datetime.fromtimestamp(t).strftime("%Y-%m-%d, %H:%M"),
                    color='white')

            fig.subplots_adjust(top=0.96, bottom=0.10,
                    left=0.05, right=0.98)
            plot_utils.setup_WCS_axes(ax)
            ax.set_xlabel("Helioprojective Longitude")
            ax.set_ylabel("Helioprojective Latitude")
            
            if overlay_celest:
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
                
                fig.subplots_adjust(top=0.90, bottom=0.10,
                        left=0.05, right=0.95)
            
            ax_orbit = fig.add_axes((.13, .13, .12, .12))
            draw_overhead_map(ax_orbit, t, path_positions, path_times)
            
            fig.savefig(f"{tmpdir}/{t:035.20f}.png")
            plt.close(fig)


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
    
    Arguments
    ---------
    data_dir : str
        Directory containing WISPR fits files. Will be scanned by
        `utils.collect_files`
    between : tuple
        A beginning and end timestamp of the format 'YYYYMMDDTHHMMSS'. Only the
        files between these timestamps will be included in the video. Either or
        both timestamp can be `None`, to place no limit on the beginning or end
        of the video.
    show : boolean
        When `True`, the rendered video is shown in the Jupyter environment.
        Otherwise, an IPython `Video` object is returned.
    fps : float
        Number of frames per second in the video
    file_load_inverval : int
        Only the first out of every `file_load_interval` files is considered.
        This option is mostly intended to speed up rendering during
        development, and no care is taken to ensure this cooperates well with
        the interleaving of images from the two cameras.
    plot_inverval : int
        Only the first out of every `plot_interval` considered files is
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
                
        os.system(f"ffmpeg -loglevel error -r {fps} -pattern_type glob"
                  f" -i '{tmpdir}/*.png' -c:v libx264 -pix_fmt yuv420p"
                  f" -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {tmpdir}/out.mp4")
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
    fig = plt.figure(figsize=(10, 6), dpi=140)
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
    
    plt.scatter(*sun_coords, s=50, color='yellow')
    
    fig.subplots_adjust(top=0.96, bottom=0.10,
            left=0.09, right=0.98)
    lon, lat = ax.coords
    ax.set_xlabel("Right Ascension")
    ax.set_ylabel("Declication")
    ax.coords.grid(color='white', alpha=0.2)
    
    with plt.style.context('dark_background'):
        ax_orbit = fig.add_axes((.85, .73, .12, .12))
        draw_overhead_map(ax_orbit, ts, path_positions, path_times)

    fig.savefig(f"{tmpdir}/{ts:035.20f}.png")
    plt.close('all')


def draw_overhead_map(ax_orbit, t, path_positions, path_times):
    """
    Draws a small plot with an overhead view of PSP's position and trajactory
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
