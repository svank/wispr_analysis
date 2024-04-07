import copy
from datetime import datetime, timezone
import functools
import gc
from itertools import repeat
from math import floor
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import warnings

from astropy.io import fits
import astropy.time
import astropy.units as u
from astropy.wcs import WCS
from IPython.display import display, Video
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import reproject
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from . import (composites, data_cleaning, orbital_frame, planets, plot_utils,
               utils)


cmap = copy.copy(matplotlib.cm.Greys_r)
cmap.set_bad('black')


def make_WISPR_video(data_dir, between=(None, None), filters=None,
                         n_procs=os.cpu_count(), debris_mask_dir=None,
                         save_location=None, timestepper='inner', dt=None,
                         duration=None, fps=30, blank_threshold=30*u.min,
                         **plot_args):
    ifiles, ofiles = utils.collect_files(data_dir, between=between,
                                         filters=filters, include_headers=True)
    ifiles, iheaders = zip(*ifiles)
    ofiles, oheaders = zip(*ofiles)
    itimes = np.array(utils.to_timestamp(iheaders))
    otimes = np.array(utils.to_timestamp(oheaders))
    file2time = dict(zip(ifiles + ofiles, np.concatenate((itimes, otimes))))
    file2next = dict(zip(ifiles[:-1] + ofiles[:-1], ifiles[1:] + ofiles[1:]))
    file2prev = dict(zip(ifiles[1:] + ofiles[1:], ifiles[:-1] + ofiles[:-1]))
    
    if timestepper == 'fixed_duration':
        wispr_time_duration = np.ptp(np.concatenate((itimes, otimes)))
        dt = wispr_time_duration / (duration * fps)
        timestepper = 'fixed_dt'
    
    if timestepper == 'inner':
        timesteps = itimes
        selected_ofiles = []
        for itime in itimes:
            i = np.argmin(np.abs(otimes - itime))
            selected_ofiles.append(ofiles[i])
        ofiles = selected_ofiles
    elif timestepper == 'outer':
        timesteps = otimes
        selected_ifiles = []
        for otime in otimes:
            i = np.argmin(np.abs(itimes - otime))
            selected_ifiles.append(ifiles[i])
        ifiles = selected_ifiles
    elif timestepper == 'fixed_dt':
        # Let's start at the image closest to perihelion, and then go out in
        # dts from there.
        E = utils.extract_encounter_number(data_dir)
        tperi = utils.to_timestamp(planets.get_psp_perihelion_date(E))
        times = itimes
        iperi = np.argmin(np.abs(times - tperi))
        tcore = times[iperi]
        if dt is None:
            dt = times[iperi+1] - times[iperi]
        
        tmin = np.min(times)
        tmax = np.max(times)
        timespan_before = tcore - tmin
        timespan_after = tmax - tcore
        timespan_before = np.ceil(timespan_before / dt) * dt        
        timespan_after = np.ceil(timespan_after / dt) * dt
        
        timesteps = np.arange(
            tcore - timespan_before, tcore + timespan_after + dt, dt)
        selected_ifiles = []
        for time in timesteps:
            i = np.argmin(np.abs(itimes - time))
            selected_ifiles.append(ifiles[i])
        selected_ofiles = []
        for time in timesteps:
            i = np.argmin(np.abs(otimes - time))
            selected_ofiles.append(ofiles[i])
        ifiles = selected_ifiles
        ofiles = selected_ofiles
    
    for i, t in enumerate(timesteps):
        if np.abs(file2time[ifiles[i]] - t) * u.s > blank_threshold:
            ifiles[i] = None
        if np.abs(file2time[ofiles[i]] - t) * u.s > blank_threshold:
            ofiles[i] = None
    
    output_wcs, naxis1, naxis2 = composites.gen_header(iheaders[0], oheaders[0])
    bounds = composites.find_collective_bounds(
        [iheaders[::15], oheaders[::15]], output_wcs)
    crpix = list(output_wcs.wcs.crpix)
    crpix[0] -= bounds[0]
    crpix[1] -= bounds[2]
    output_wcs.wcs.crpix = crpix
    naxis1 = bounds[1] - bounds[0]
    naxis2 = bounds[3] - bounds[2]
    output_wcs.pixel_shape = naxis1, naxis2
    
    psp_poses, psp_times = planets.trace_psp_orbit(
        utils.extract_encounter_number(data_dir), t_start=timesteps[0],
        t_stop=timesteps[-1])
    psp_poses = psp_poses.transform_to(orbital_frame.PSPOrbitalFrame)
    
    output_wcses = []
    planet_poses = []
    with utils.ignore_fits_warnings():
        for t in timesteps:
            psp = planets.locate_psp(date=t)
            psp = psp.transform_to('heliographic_stonyhurst')
            t = datetime.fromtimestamp(t, tz=timezone.utc)
            t = astropy.time.Time(t)
            
            wcs = output_wcs.deepcopy()
            wcs.wcs.dateobs = ''
            wcs.wcs.dateavg = ''
            wcs.wcs.mjdobs = t.mjd
            wcs.wcs.mjdavg = wcs.wcs.mjdobs
            wcs.wcs.aux.hglt_obs = psp.lat.to_value(u.deg)
            wcs.wcs.aux.hgln_obs = psp.lon.to_value(u.deg)
            wcs.wcs.aux.dsun_obs = psp.radius.to_value(u.m)
            wcs.fix()
            
            output_wcses.append(wcs)
            
            # SPICE doesn't always play well with multiprocessing, so we
            # pre-compute the positions here in the main process.
            if plot_args.get('mark_planets', False):
                planet_poses.append(
                    (wcs, planets.locate_planets(wcs.wcs.dateavg)))
            else:
                planet_poses.append(False)
    
    generic_make_video(_draw_WISPR_video_frame, timesteps, ifiles, ofiles,
                       output_wcses, planet_poses, repeat(psp_poses),
                       repeat(psp_times), file2next, file2prev, plot_args,
                       debris_mask_dir, parallel=n_procs, fps=fps,
                       save_to=save_location)


def _draw_WISPR_video_frame(out_file, t, ifile, ofile, wcs, planet_poses,
                            psp_poses, psp_times, file2next, file2prev,
                            plot_args, debris_mask_dir):
    if debris_mask_dir is not None:
        if ifile is not None:
            mask = data_cleaning.find_mask(debris_mask_dir, ifile)
            data = data_cleaning.dust_streak_filter(
                file2prev.get(ifile), ifile, file2next.get(ifile),
                sliding_window_stride=3, precomputed_mask=mask, radec=False)
            ifile = (data, ifile)
        if ofile is not None:
            mask = data_cleaning.find_mask(debris_mask_dir, ofile)
            data = data_cleaning.dust_streak_filter(
                file2prev.get(ofile), ofile, file2next.get(ofile),
                sliding_window_stride=3, precomputed_mask=mask, radec=False)
            ofile = (data, ofile)
    
    composite, _ = composites.gen_composite(
        ifile, ofile, wcsh=wcs, bounds=False, image_trim=[[10]*4]*2)
    
    with matplotlib.style.context('dark_background'):
        fig = plt.figure(figsize=(10, 7.5), dpi=150)
        plot_args = copy.copy(plot_args)
        plot_args['mark_planets'] = planet_poses
        plot_utils.plot_WISPR(composite, wcs=wcs, **plot_args)
        ax = plt.gca()
        timestamp = datetime.fromtimestamp(t, tz=timezone.utc)
        ax.text(40, 30, timestamp.strftime("%Y-%m-%d, %H:%M"), color='white')
        fig.subplots_adjust(top=0.96, bottom=0.10, left=0.05, right=0.98)
        main_ax_pos = ax.get_position()
        
        ax_orbit = fig.add_axes((.01 + main_ax_pos.xmin, .04 + main_ax_pos.ymin,
                                 .12, .12))
        cart = psp_poses.cartesian
        ax_orbit.margins(.1, .1)
        ax_orbit.set_aspect('equal')
        ax_orbit.plot(cart.x, cart.y, color='.4', zorder=-1)
        r = np.interp(t, psp_times, psp_poses.distance)
        theta = np.interp(
            t, psp_times, np.unwrap(psp_poses.lon.to_value(u.deg), period=360))
        theta %= 360
        sc_x = r * np.cos(theta * np.pi/180)
        sc_y = r * np.sin(theta * np.pi/180)
        ax_orbit.scatter(sc_x, sc_y, s=6, color='1')
        ax_orbit.scatter(0, 0, color='yellow')
        ax_orbit.xaxis.set_visible(False)
        ax_orbit.yaxis.set_visible(False)
        ax_orbit.set_title("S/C position", fontdict={'fontsize': 9})
        for spine in ax_orbit.spines.values():
            spine.set_color('.4')
        
        fig.text(main_ax_pos.xmin + .13, main_ax_pos.ymin + .065,
                 f"r = {r.to_value(u.R_sun):.1f}", color='white')
        fig.text(main_ax_pos.xmin + .19, main_ax_pos.ymin + .065,
                 f"R$_\odot$",color='white')
        fig.text(main_ax_pos.xmin + .13, main_ax_pos.ymin + .04,
                 f"      {r.to_value(u.AU):.2f}", color='white')
        fig.text(main_ax_pos.xmin + .19, main_ax_pos.ymin + .04,
                 f"AU", color='white')
        fig.text(main_ax_pos.xmin + .13, main_ax_pos.ymin + .095,
                 f"$\\theta$ = {theta:.1f} $^\ocirc$",
                 color='white')
        
        fig.savefig(out_file)


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


def _function_caller(*args, **kwargs):
    function = args[0]
    args = args[1:]
    function(*args, **kwargs)
    # Make sure matplotlib releases memory
    plt.clf()
    plt.close('all')


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
        
        ready_arg_list = [repeat(frame_renderer), output_names()]
        n = np.inf
        have_iterable = False
        for arg in arg_list:
            if hasattr(arg, "__iter__") and not isinstance(arg, (str, dict)):
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
                    _function_caller, *ready_arg_list,
                    total=n, max_workers=max_workers)
        else:
            for args in tqdm(zip(*ready_arg_list), total=n):
                _function_caller(*args)
        
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

