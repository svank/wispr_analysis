import copy
import dataclasses
import functools
from itertools import repeat

import astropy.constants as c
from astropy.coordinates import SkyCoord
import astropy.visualization
import astropy.units as u
import parkersolarwind as psw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import numba
import scipy.optimize
from sunpy.coordinates.frames import HeliocentricInertial
from tqdm.contrib.concurrent import process_map

from . import orbital_plane_slices as ops
from .. import planets, plot_utils, videos
from ..synthetic_data import synthetic_data

astropy.visualization.quantity_support()


def make_image(t, sc, parcels, output_quantity, synth_kw, output_size_x, output_size_y):
    parcel_width = 1 if output_quantity == 'flux' else 0.25
    if 'parcel_width' not in synth_kw:
        synth_kw['parcel_width'] = parcel_width
    image, wcs = synthetic_data.synthesize_image(sc, parcels, t, output_size_x=output_size_x, output_size_y=output_size_y,
                                                 output_quantity=output_quantity, **synth_kw)
    return image, wcs


def generate_images(simdat, output_quantity='flux', synth_kw={}, disable_pbar=False,
                    output_size_x=200, output_size_y=40):
    images = np.empty((len(simdat.t), output_size_y, output_size_x))
    wcses = []
        
    for i, (image, wcs) in enumerate(process_map(
            make_image, simdat.t, repeat(simdat.sc), repeat(simdat.parcels),
            repeat(output_quantity), repeat(synth_kw), repeat(output_size_x),
            repeat(output_size_y), chunksize=10, disable=disable_pbar)):
        images[i] = image
        wcses.append(wcs)
    return ops.InputDataBundle(
        images=images,
        wcses=wcses,
        times=simdat.t,
        is_inner='both',
        level='3',
        quantity=output_quantity,
        encounter=simdat.encounter)


@numba.njit(cache=True)
def integrate_velocity_profile(rgrid, vgrid, dt):
    t_start = 0. # s
    
    rs = [rgrid[0]]
    ts = [t_start]
    vs = [vgrid[0]]
    
    r = rs[0]
    t = ts[0]
    while r < rgrid[-1]:
        v = np.interp(r, rgrid, vgrid)
        # Adaptive timestep---make sure we go at least 1000 km
        dt = 1000_000 / v
        dt = max(dt, 10*60)
        r = r + v * dt
        t = t + dt
        rs.append(r)
        ts.append(t)
        vs.append(v)
    return rs, ts, vs


@functools.lru_cache(50000)
def gen_parcel_path_physical(Riso=11.5*u.R_sun, Tiso=1.64*u.MK, gamma=1.38,
                             end_point=250*u.R_sun, extras=True):
    # Log space gives more points in the inner region, where the critical point
    # and the iso--poly boundary are, and less points far out where nothing
    # much is happening.
    rs = np.logspace(0, np.log10(end_point.to_value(u.R_sun)), 750)
    rs <<= u.R_sun
    if Riso >= rs[-1]:
        rgrid, rho_iso, vgrid, Tiso, mu = psw.solve_parker_isothermal(rs, Tiso)
    elif Riso <= rs[0]:
        rgrid, rho_poly, vgrid, T_poly, r_crit, uc_crit, gamma, T0, mu = \
            psw.solve_parker_polytropic(rs, Tiso, gamma)
    else:
        riso, rhoiso, uiso, Tiso, rpoly, rhopoly, upoly, Tpoly, gamma, mu = \
            psw.solve_isothermal_layer(
                rs, # Radial grid points
                Riso, # Isothermal layer outer radius
                Tiso, # Isothermal layer temperature
                gamma, # Polytropic layer polytropic index
            )
        rgrid = np.concatenate((riso, rpoly))
        vgrid = np.concatenate((uiso, upoly))

    dt = 10. * 60 * u.s
    rs, ts, vs = integrate_velocity_profile(
        rgrid.to(u.m).value, vgrid.to(u.m/u.s).value, dt.value)
    
    rs = u.Quantity(rs, u.m, copy=False)
    ts = u.Quantity(ts, u.s, copy=False)
    vs = u.Quantity(vs, u.m/u.s, copy=False)

    if not extras:
        return rs, ts, vs
    
    dr = np.gradient(rs)
    dv = np.gradient(vs)
    grad_v = dv / dr
    drho_over_rho = 1 / (1 + dt * grad_v) - 1
    rho = 1
    rhos = [rho]
    for d in drho_over_rho[:-1]:
        rhos.append(rhos[-1] + d * rhos[-1])

    rhos = np.array(rhos)
    rpar = 1 / rhos
    rpar /= rpar[0]

    rperp = rs.value**2
    rperp /= rperp[0]

    rhos *= 1 / rs.value**2
    
    return rs, ts, vs, rhos, rperp, rpar


def gen_parcels_in_plane_physical(n_parcels=500, nt=400, E='E13',
                                  Riso=11.5*u.R_sun, Tiso=1.64*u.MK,
                                  gamma=1.38):
    sc, t = load_sc_spice(E, nt)
    
    parcels = []
    np.random.seed(1)
    t_min = 120 * u.R_sun.to(u.m) / 200_000

    rs, ts, vs, rhos, rperp, rpar = gen_parcel_path_physical(Riso, Tiso, gamma)

    orbital_frame = ops.OrbitalSliceWCS(None, None, None).orbital_frame
    for _ in range(n_parcels):
        this_theta = np.random.uniform(0, 2*np.pi, 1)
        t_start = np.random.uniform(t[0] - t_min, t[-1], 1)
        r = rs.to(u.m).value
        x = r*np.cos(this_theta)
        y = r*np.sin(this_theta)
        z = 0
        c = SkyCoord(x, y, z, unit='m', representation_type='cartesian',
                     frame=orbital_frame).transform_to(HeliocentricInertial)
        cc = c.cartesian
        tvals = t_start + ts.to(u.s).value
        parcels.append(synthetic_data.ArrayThing(
            tvals,
            cc.x.value,
            cc.y.value,
            cc.z.value,
            t=t_start,
            rholist=rhos,
            rperplist=rperp,
            rparlist=rpar,
            t_min=tvals[0],
            t_max=tvals[-1]))

    out = SimulationData(
        sc=sc, parcels=parcels, t=t, plasma_v=None, encounter=E)
    out.Riso = Riso
    out.Tiso = Tiso
    out.gamma = gamma
    return out


@dataclasses.dataclass
class SimulationData:
    sc: synthetic_data.Thing
    parcels: list[synthetic_data.Thing]
    t: list
    plasma_v: float
    encounter: int


def load_sc_spice(E, nt=400):
    coords, times = planets.get_orbital_plane(
        'psp', E, npts=10000, return_times=True)
    f = coords.represent_as('spherical').distance < 0.25 * u.au
    coords = coords[f]
    times = times[f]
    x, y, z = coords.x.to(u.m).value, coords.y.to(u.m).value, coords.z.to(u.m).value
    sc = synthetic_data.ArrayThing(times, x, y, z)
    t = np.linspace(times.min(), times.max(), nt)
    return sc, t


def _plot_overhead(ax_overhead, sc, t, t0, parcels, mark_epsilon, mark_FOV_pos,
                   mark_FOV, mark_derot_ax, mark_bins, detail, fovdat, 
                   override_limits):
    scale_factor = u.R_sun.to(u.m)
    ax_overhead.scatter(0, 0, c='yellow', s=100, zorder=18, edgecolors='black')
    ax_overhead.scatter(sc.x, sc.y, zorder=10, s=100, edgecolors='white')
    ax_overhead.plot(sc.at(t).x, sc.at(t).y, zorder=9, lw=5)

    pxs = []
    pys = []
    for parcel in parcels:
        with parcel.at_temp(t) as p:
            ax_overhead.plot(p.x, p.y, 'C1', alpha=.3)
        with parcel.at_temp(t0) as p:
            pxs.append(p.x)
            pys.append(p.y)
    ax_overhead.scatter(pxs, pys, color='C1', s=36 if detail else 24)

    if mark_epsilon:
        x = np.mean(pxs)
        y = np.mean(pys)
        ax_overhead.plot([0, sc.x, x],
                         [0, sc.y, y],
                         color='gray')
    if mark_FOV_pos:
        x = np.mean(pxs)
        y = np.mean(pys)
        dx = sc.at(t0+.5).x - sc.x
        dy = sc.at(t0+.5).y - sc.y
        ax_overhead.plot([x, sc.x, sc.x - dy],
                         [y, sc.y, sc.y + dx],
                         color='gray')
    if mark_FOV:
        x1, x2 = 0, fovdat[0].shape[1]
        y1 = fovdat[0].shape[0] / 2
        y2 = y1
        lon1, lat1 = fovdat[1].pixel_to_world_values(x1, y1)
        lon2, lat2 = fovdat[1].pixel_to_world_values(x2, y2)
        
        to_sun_x = -sc.x
        to_sun_y = -sc.y
        to_sun_theta = np.arctan2(to_sun_y, to_sun_x)
        t1 = to_sun_theta - lon1 * np.pi/180
        t2 = to_sun_theta - lon2 * np.pi/180
        
        size = (60 if mark_bins else 15) * scale_factor
        x1 = size * np.cos(t1) + sc.x
        x2 = size * np.cos(t2) + sc.x
        y1 = size * np.sin(t1) + sc.y
        y2 = size * np.sin(t2) + sc.y

        ax_overhead.plot([x1, sc.x, x2],
                         [y1, sc.y, y2],
                         color='w', alpha=0.75,
                         lw=2.5, zorder=18)
        if mark_bins:
            rs = np.array([20, 40, 60]) * scale_factor
            ts = np.linspace(t1, t2, 50)
            for r in rs:
                binx = r * np.cos(ts) + sc.x
                biny = r * np.sin(ts) + sc.y
                ax_overhead.plot(
                    binx, biny, color='w', alpha=0.75,
                    lw=2.5, zorder=18)

    if mark_derot_ax:
        length = (10 if detail else 30) * scale_factor
        ax_overhead.arrow(sc.x[0]-length/2, sc.y[0], length, 0, color='.7',
                          zorder=18)
        ax_overhead.arrow(sc.x[0], sc.y[0]-length/2, 0, length, color='.7',
                          zorder=18)

    if detail:
        half_window = 8 * (u.R_sun.to(u.m))
        ax_overhead.set_xlim(sc.x - half_window, sc.x + half_window)
        ax_overhead.set_ylim(sc.y - half_window, sc.y + half_window)
        ax_overhead.plot([0, sc.x[0]], [0, sc.y[0]], color='w', alpha=.5)
        t = np.arctan2(sc.y[0], sc.x[0])
        r = sc.r[0]
        rs = np.arange(0, r, u.R_sun.to(u.m))
        ax_overhead.scatter(rs * np.cos(t), rs * np.sin(t), color='white',
                            s=25, alpha=.5)
    else:
        margin = 1
        xs = list(sc.at(t).x)
        xmin, xmax = np.nanmin(xs), np.nanmax(xs)
        xrange = xmax - xmin
        
        ys = list(sc.at(t).y)
        ymin, ymax = np.nanmin(ys), np.nanmax(ys)
        yrange = ymax - ymin

        if yrange == 0:
            yrange = xrange
        if xrange == 0:
            xrange = yrange
        
        tbounds = np.array([t[0], t[-1]])
        for parcel in parcels:
            with parcel.at_temp(tbounds) as p:
                xs.extend(p.x)
                ys.extend(p.y)
        
        xs = [x for x in xs
              if xmin - margin * xrange <= x <= xmax + margin * xrange]
        ax_overhead.set_xlim(np.nanmin(xs)-1, np.nanmax(xs)+1)

        ys = [y for y in ys
              if ymin - margin * yrange <= y <= ymax + margin * yrange]
        
        xmin, xmax = np.nanmax(xs), np.nanmin(xs)
        if np.nanmax(np.abs(ys)) < 2 * (xmax - xmin) and xmin < 0 < xmax:
            # Include the Sun in the plot y-range if it doesn't warp the aspect
            # ratio too much and if the Sun is already included in the x
            # plotting range
            ys.append(0)
        else:
            # Put "To Sun" arrows?
            pass

        ax_overhead.set_ylim(np.nanmin(ys)-1, np.nanmax(ys) + 1)

    ax_overhead.set_aspect('equal')
    ax_overhead.set_facecolor('black')
    ax_overhead.set_xlabel("X ($R_\odot$)")
    if not detail:
        ax_overhead.set_ylabel("Y ($R_\odot$)")

    # Label axes in R_sun without having to re-scale every coordinate
    formatter = matplotlib.ticker.FuncFormatter(
        lambda x, pos: f"{x / u.R_sun.to(u.m):.0f}")
    ax_overhead.xaxis.set_major_formatter(formatter)
    ax_overhead.yaxis.set_major_formatter(formatter)
    ax_overhead.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(
        20*u.R_sun.to(u.m)))
    ax_overhead.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(
        20*u.R_sun.to(u.m)))

    if override_limits is not None:
        override_limits = np.array(override_limits) * u.R_sun.to(u.m)
        ax_overhead.set_xlim(*override_limits[:2])
        ax_overhead.set_ylim(*override_limits[2:])

def plot_items(sc, parcels, t, t0, include_overhead=True,
               include_overhead_detail=False, mark_epsilon=False,
               mark_FOV_pos=False, mark_FOV=False, mark_bins=False,
               mark_derot_ax=False, synthesize=False, radiants=False, vmin=0,
               vmax=None, parcel_width=1, synth_kw={}, synth_fixed_fov=None,
               synth_celest_wcs=False, output_quantity='flux',
               use_default_figsize=False, synth_colorbar=False,
               overhead_override_limits=None):
    sc = sc.at(t0)
    n_plots = include_overhead + include_overhead_detail + synthesize
    if use_default_figsize:
        figsize = None
    else:
        figsize = (7 * n_plots, 7)
    fig, axs = plt.subplots(1, n_plots,
                            figsize=figsize, squeeze=False,
                            # dpi=200,
                            layout='constrained',
                            # width_ratios=[1] * n_plots if not (synthesize and synth_fixed_fov) else [1] * (n_plots-1) + [3],
                           )
    axs = list(axs.flatten())
    # fig.subplots_adjust(left=.04, right=.99, wspace=.08)
    
    ax_overhead = axs.pop(0) if include_overhead else None
    ax_overhead_detail = axs.pop(0) if include_overhead_detail else None
    ax_syn = axs.pop(0) if synthesize else None

    # Even if we're not showing it, synthesize an image so we can grab the FOV
    image, wcs = synthetic_data.synthesize_image(
        sc, parcels, t0, output_size_x=200, output_size_y=125,
        parcel_width=parcel_width, fixed_fov_range=synth_fixed_fov,
        celestial_wcs=synth_celest_wcs,
        projection='CAR' if synth_fixed_fov else 'ARC',
        output_quantity=output_quantity, **synth_kw)
    if ax_syn is not None:
        cmap = 'Greys_r'
        if output_quantity in ('dsun', 'distance'):
            image /= u.R_sun.to(u.m)
            gamma = 1
        else:
            gamma = 1/2.2
        if output_quantity == 'dsun':
            cmap = 'coolwarm'
        if output_quantity == 'distance':
            cmap = 'viridis'
        
        xs = ([sc.at(t[0]).x] + [p.at(t[0]).x for p in parcels]
              + [sc.at(t[-1]).x] + [p.at(t[-1]).x for p in parcels])
        ys = ([sc.at(t[0]).y] + [p.at(t[0]).y for p in parcels]
              + [sc.at(t[-1]).y] + [p.at(t[-1]).y for p in parcels])
        
        # Recreate axis with WCS projection (I wish there were a better way!)
        ax_syn.remove()
        ax_syn = fig.add_subplot(100 + 10 * n_plots + n_plots, projection=wcs)
        if vmax is None:
            if output_quantity == 'dsun':
                vmax = 2 * sc.at(t0).r / u.R_sun.to(u.m)
            else:
                vmax = image.max()
                if vmax == 0:
                    vmax = 1
        im = ax_syn.imshow(image, origin='lower', aspect='equal', cmap=cmap,
                           norm=matplotlib.colors.PowerNorm(
                               gamma=gamma, vmin=vmin, vmax=vmax))
        
        lon, lat = ax_syn.coords
        lat.set_major_formatter('dd')
        lon.set_major_formatter('dd')
        if synth_celest_wcs:
            ax_syn.set_xlabel("Fixed Longitude")
        else:
            ax_syn.set_xlabel("HP Longitude")
        ax_syn.set_ylabel(" ")
        ax_syn.coords.grid(color='white', alpha=0.1)
        
        if radiants:
            for parcel in parcels:
                radiant = synthetic_data.calculate_radiant(sc, parcel, t0)
                if radiant is None:
                    continue
                x, y = wcs.world_to_pixel(radiant * u.rad, 0 * u.rad)
                ax_syn.scatter(x, y, s=10, marker='x', color='green')
        if synth_colorbar:
            plt.colorbar(im, ax=[
                ax for ax in (ax_syn, ax_overhead, ax_overhead_detail)
                if ax is not None])
    if ax_overhead is not None:
        _plot_overhead(ax_overhead, sc, t, t0, parcels, mark_epsilon,
                       mark_FOV_pos, mark_FOV, mark_derot_ax, mark_bins,
                       detail=False, fovdat=(image, wcs),
                       override_limits=overhead_override_limits)
    
    if ax_overhead_detail is not None:
        _plot_overhead(ax_overhead_detail, sc, t, t0, parcels, mark_epsilon,
                       mark_FOV_pos, mark_FOV, mark_derot_ax, mark_bins=False,
                       detail=True, fovdat=(image, wcs), override_limits=None)
    
    # fig.subplots_adjust(left=.05, right=.95, bottom=.05, top=.95, wspace=.1)


@videos.wrap_with_gc
def draw_frame(output_name, i, setup):
    plot_items(setup.sc, setup.parcels, setup.t, setup.t[i],
               include_overhead_detail=False, synthesize=True,
               mark_FOV=True, include_overhead=True, output_quantity='flux',
               synth_colorbar=False,
               overhead_override_limits=(-100, 100, -100, 100),
              )
    plt.savefig(output_name)
    plt.close()


orbital_frame = ops.OrbitalSliceWCS(None, None, None).orbital_frame


def gen_simdat(Riso, Tiso, gamma, launch_thetas, launch_times, sc, t_sc,
               time_epoch):
    rs, ts, vs, rhos, rperp, rpar = gen_parcel_path_physical(Riso, Tiso, gamma)
    
    parcels = []
    for theta, t0 in zip(launch_thetas, launch_times):
        tvals = ts + t0 + time_epoch
        rvals = rs.to(u.m)
        x = rvals * np.cos(theta)
        y = rvals * np.sin(theta)
        coord = SkyCoord(x, y, 0*u.m, representation_type='cartesian',
                     frame=orbital_frame).transform_to(HeliocentricInertial)
        cc = coord.cartesian
        parcel = synthetic_data.ArrayThing(
            tvals,
            cc.x,
            cc.y,
            cc.z,
            t=t0,
            rholist=rhos,
            rperplist=rperp,
            rparlist=rpar,
            t_min=tvals[0],
            t_max=tvals[-1])
        parcels.append(parcel)
    simdat = SimulationData(sc=sc, parcels=parcels, t=t_sc,
                            plasma_v=None, encounter=12)
    return rs, ts, vs, simdat


def try_out_fitting(Riso, Tiso, gamma, launch_thetas, launch_times,
                    model='both', max_elongation=None,
                    Riso_lock=None, Tiso_lock=None, gamma_lock=None,
                    launch_thetas_lock=None, launch_times_lock=None,
                    init_guess=None, pull_in=0, only_setup=False):
    launch_thetas, launch_times = np.broadcast_arrays(
        launch_thetas, launch_times, subok=True)
    
    sc, t_sc = load_sc_spice(12, 500)
    
    t_sc = t_sc << u.s
    
    time_epoch = t_sc[0]
    
    sc.xlist <<= u.m
    sc.ylist <<= u.m
    sc.zlist <<= u.m
    sc.tlist <<= u.s
    
    rs_actual, ts_actual, vs_actual, simdat = gen_simdat(
        Riso, Tiso, gamma, launch_thetas, launch_times, sc, t_sc, time_epoch)
    
    all_times = [t_sc] * len(launch_times)
    sc = simdat.sc.at(np.concatenate(all_times))
    
    c_sc = SkyCoord(sc.x, sc.y, sc.z,
                representation_type='cartesian',
                frame=HeliocentricInertial).transform_to(orbital_frame)
    c_sc_cart = c_sc.cartesian
    
    all_elongations = compute_observed_elongations(
        Riso, Tiso, gamma, launch_thetas, launch_times,
        all_times, time_epoch, (c_sc_cart.x, c_sc_cart.y))
    
    all_target_elongations = []
    all_times_for_fit = []
    while len(all_elongations):
        elongations = all_elongations[:len(t_sc)]
        all_elongations = all_elongations[len(t_sc):]
        max = max_elongation if max_elongation is not None else 108.5*u.deg
        f = (elongations > 13.5*u.deg) * (elongations < max)
        if np.any(f):
            all_target_elongations.append(elongations[f])
            all_times_for_fit.append(t_sc[f])
    
    all_times_as_one = np.concatenate(all_times_for_fit)
    all_elongations_as_one = np.concatenate(all_target_elongations)
    
    sc = simdat.sc.at(all_times_as_one)
    c_sc = SkyCoord(sc.x, sc.y, sc.z,
                    representation_type='cartesian',
                    frame=HeliocentricInertial).transform_to(orbital_frame)
    c_sc_cart = c_sc.cartesian
    
    if only_setup:
        fit_result = FittingResult(
        res=None, Riso=Riso, Tiso=Tiso, gamma=gamma,
        fitted_Riso=None, fitted_Tiso=None, fitted_gamma=None,
        fitted_lons=None, fitted_times=None,
        launch_thetas=launch_thetas, launch_times=launch_times, simdat=simdat,
        times_for_fit=all_times_for_fit, sc_coords=(c_sc_cart.x, c_sc_cart.y),
        target_elongations=all_target_elongations,
        time_epoch=time_epoch, init_guess=None,
        rs_actual=rs_actual, vs_actual=vs_actual, ts_actual=ts_actual)
        return fit_result

    if len(all_times_for_fit) > 1:
        start_idx = np.cumsum(list(len(t) for t in all_times_for_fit))
        start_idx = np.concatenate(([0], start_idx[:-1])).astype(int)
    else:
        start_idx = np.array([0])
    
    start_guess_lon = (c_sc.lon[start_idx] + 15*u.deg).to(u.rad).value
    start_times = u.Quantity([t[0] for t in all_times_for_fit])
    start_guess_time_offset = (start_times - time_epoch).to(u.hr).value
    
    initial_guess = []
    bounds = []
    locks = []
    if Riso_lock is not None:
        locks.append(Riso_lock)
    else:
        if model == 'poly':
            initial_guess.append(0.05)
            bounds.append((0, 0.1))
        elif model == 'iso':
            initial_guess.append(1e5 + 0.5)
            bounds.append((1e5, 1e5+1))
        else:
            bounds.append((1.5, 5))
            initial_guess.append(2.5)
        locks.append(None)
    
    if Tiso_lock is not None:
        locks.append(Tiso_lock)
    else:
        initial_guess.append(3)
        bounds.append((.5, 3.5))
        locks.append(None)
    
    if gamma_lock is not None:
        locks.append(gamma_lock)
    else:
        initial_guess.append(1.3)
        bounds.append((1.05, 1.5))
        locks.append(None)
    
    if launch_thetas_lock is not None:
        locks.append(launch_thetas_lock)
    else:
        initial_guess.extend(start_guess_lon)
        bounds.extend(([(None, None)]) * len(start_guess_lon))
        locks.append(None)
    
    if launch_times_lock is not None:
        locks.append(launch_times_lock)
    else:
        initial_guess.extend(start_guess_time_offset)
        bounds.extend(([(None, None)]) * len(start_guess_time_offset))
        locks.append(None)
    
    if init_guess is not None:
        initial_guess = init_guess
    
    def unpack_args(x):
        i = 0
        if locks[0] is None:
            Riso = x[i] * u.R_sun
            i += 1
        else:
            Riso = locks[0]
        
        if locks[1] is None:
            Tiso = x[i] * u.MK
            i += 1
        else:
            Tiso = locks[1]
        
        if locks[2] is None:
            gamma = x[i]
            i += 1
        else:
            gamma = locks[2]
        
        if locks[3] is None:
            launch_angles = x[i:i+len(start_guess_lon)] * u.rad
            i += len(start_guess_lon)
        else:
            launch_angles = locks[3]
        
        if locks[4] is None:
            launch_times = x[i:i+len(start_guess_lon)] * u.hr
        else:
            launch_times = locks[4]
        return Riso, Tiso, gamma, launch_angles, launch_times
    
    C0_Cg2_base = (2 * c.k_B / (c.m_p * c.G * c.M_sun))
    # Constraints as given by Shi, Velli, Bale et al. 2022
    def constraint_upper(x):
        Riso, Tiso, gamma, _, _ = unpack_args(x)
        C0_Cg2 = C0_Cg2_base * gamma * Tiso * Riso
        # We require C0 < Cg
        #            => C0 / Cg < 1
        #            => C0^2 / Cg^2 < 1
        #            => (C0 / Cg)^2 - 1 < 0
        # To put as a "constraint must be positive" form, return -(C0 / Cg - 1)
        term = -(C0_Cg2.si.value - 1)
        return term - pull_in
    
    def constraint_lower(x):
        Riso, Tiso, gamma, _, _ = unpack_args(x)
        C0_Cg2 = C0_Cg2_base * gamma * Tiso * Riso
        # We require (C0 / Cg)^2 > 2 (gamma - 1)
        term = C0_Cg2.si.value - 2 * (gamma - 1)
        return term - pull_in
    
    extra_args = (unpack_args, all_elongations_as_one, all_times_for_fit, time_epoch,
                  (c_sc_cart.x, c_sc_cart.y))
    # print(initial_guess, bounds)
    res = scipy.optimize.minimize(
        elongation_residual_ltsq,
        initial_guess,
        method='trust-constr',
        bounds=bounds,
        constraints=[
            scipy.optimize.NonlinearConstraint(constraint_upper, 0, np.inf),
            scipy.optimize.NonlinearConstraint(constraint_lower, 0, np.inf),
            ],
        # method='SLSQP',
        # constraints=[
        #     {'type': 'ineq', 'fun': constraint_upper},
        #     {'type': 'ineq', 'fun': constraint_lower},
        #     ],
        args=extra_args,
        # jac='2-point',
        # options=dict(verbose=3),
        # callback=lambda intermediate_result: print(intermediate_result)
        )
    
    if not res.success:
        print(res)
    
    Rfit, Tfit, gamma_fit, angles_fit, times_fit = unpack_args(res.x)
    
    fit_result = FittingResult(
        res=res, Riso=Riso, Tiso=Tiso, gamma=gamma,
        fitted_Riso=Rfit, fitted_Tiso=Tfit, fitted_gamma=gamma_fit,
        fitted_lons=angles_fit, fitted_times=times_fit,
        launch_thetas=launch_thetas, launch_times=launch_times, simdat=simdat,
        times_for_fit=all_times_for_fit, sc_coords=(c_sc_cart.x, c_sc_cart.y),
        target_elongations=all_target_elongations,
        time_epoch=time_epoch, init_guess=initial_guess,
        rs_actual=rs_actual, vs_actual=vs_actual, ts_actual=ts_actual)
    return fit_result


def compute_observed_elongations(Riso, Tiso, gamma, lons, time_offsets,
                                 all_times_for_fit, parcel_launch_time_epoch, 
                                 sc_coords):
    rs, ts, vs = gen_parcel_path_physical(Riso, Tiso, gamma, extras=False)

    ts = ts + parcel_launch_time_epoch

    xs = []
    ys = []
    for times_for_fit, offset, lon in zip(
            all_times_for_fit, time_offsets, lons):
        parcel_rs_at_sc_ts = np.interp(times_for_fit, ts + offset, rs)
        
        xs.append(parcel_rs_at_sc_ts * np.cos(lon))
        ys.append(parcel_rs_at_sc_ts * np.sin(lon))
    
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    
    sun_in_sc_frame = -sc_coords[0], -sc_coords[1]
    parcel_in_sc_frame = xs - sc_coords[0], ys - sc_coords[1]
    
    sun_angle = np.arctan2(sun_in_sc_frame[1], sun_in_sc_frame[0])
    parcel_angle = np.arctan2(parcel_in_sc_frame[1], parcel_in_sc_frame[0])
    # This sign convention works for a typical WISPR pointing
    elongations = sun_angle - parcel_angle
    # In-place unit conversion
    elongations <<= u.deg
    return elongations

guesses = []
def elongation_residual(x, unpacker, target_elongations, all_times_for_fit,
                        parcel_launch_time_epoch, sc_coords):
    # print(x)
    global extra_args
    extra_args = (all_times_for_fit, parcel_launch_time_epoch, sc_coords)
    Riso, Tiso, gamma, lons, time_offsets = unpacker(x)
    guesses.append(copy.deepcopy((Riso, Tiso, gamma, lons, time_offsets)))
    computed = compute_observed_elongations(
        Riso, Tiso, gamma, lons, time_offsets, all_times_for_fit,
        parcel_launch_time_epoch, sc_coords)
    resid = computed - target_elongations
    resid <<= u.deg
    return resid.value


def elongation_residual_ltsq(*args, **kwargs):
    resid = elongation_residual(*args, **kwargs)
    resid[np.isnan(resid)] = 90
    return np.sqrt(np.sum(np.square(resid)))


@dataclasses.dataclass
class FittingResult:
    res: scipy.optimize.OptimizeResult
    Riso: float
    Tiso: float
    gamma: float
    launch_thetas: np.ndarray
    launch_times: np.ndarray
    fitted_Riso: u.Quantity
    fitted_Tiso: u.Quantity
    fitted_gamma: float
    fitted_lons: u.Quantity
    fitted_times: u.Quantity
    simdat: SimulationData
    target_elongations: list
    times_for_fit: list
    rs_actual: u.Quantity
    vs_actual: u.Quantity
    ts_actual: u.Quantity
    time_epoch: float
    sc_coords: tuple
    init_guess: tuple
    
    def __post_init__(self):
        if self.res is None:
            self.fitted_elongations = [None] * len(self.launch_times)
            self.fitted_times = self.fitted_elongations
            self.fitted_lons = self.fitted_times
            self.rs_fitted = [np.nan] * u.km
            self.vs_fitted = [np.nan] * u.km/u.s
            self.ts_fitted = [np.nan] * u.s
            return
        
        (self.rs_fitted, self.ts_fitted, self.vs_fitted,
            self.simdat_fitted) = gen_simdat(
            self.fitted_Riso, self.fitted_Tiso, self.fitted_gamma,
            self.fitted_lons, self.fitted_times, self.simdat.sc, self.simdat.t,
            self.time_epoch)
    
        fitted_elongations_as_one = compute_observed_elongations(
            self.fitted_Riso, self.fitted_Tiso, self.fitted_gamma,
            self.fitted_lons, self.fitted_times, self.times_for_fit,
            self.time_epoch, self.sc_coords)
        
        self.fitted_elongations = []
        for targ_elon in self.target_elongations:
            i = len(targ_elon)
            self.fitted_elongations.append(fitted_elongations_as_one[:i])
            fitted_elongations_as_one = fitted_elongations_as_one[i:]
        
        self.resid = self.res.fun
    
    @property
    def n_parcels_intended(self):
        return len(self.launch_thetas)
    
    @property
    def n_parcels_visible(self):
        return len(self.times_for_fit)
    
    def get_target_vs(self):
        vs = []
        for ts, offset in zip(self.times_for_fit, self.launch_times):
            vs.append(np.interp(
                ts - offset - self.time_epoch, self.ts_actual, self.vs_actual))
        return vs
    
    def get_fitted_vs(self):
        vs = []
        for ts, offset in zip(self.times_for_fit, self.fitted_times):
            vs.append(np.interp(
                ts - offset - self.time_epoch, self.ts_fitted, self.vs_fitted))
        return vs

    def _simulate_scenario(self, simdat, elongations):
        simdat_stripped = copy.deepcopy(simdat)
        simdat_stripped.sc = simdat_stripped.sc.strip_units()
        simdat_stripped.parcels = [p.strip_units() for p in simdat_stripped.parcels]
        simdat_stripped.t = simdat_stripped.t.si.value
        
        videos.generic_make_video(
            draw_frame, range(0, len(simdat_stripped.t), 12),
            simdat_stripped, fps=10, parallel=12)
        
        bundle = generate_images(
            simdat_stripped, output_size_x=500, output_size_y=175)
        composite_jmap = ops.extract_slices(
            bundle, 500, disable_pbar=True)
        
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        composite_jmap.plot(pmax=100)
        plt.subplot(122)
        for times, elongations in zip(
                self.times_for_fit, elongations):
            x = plot_utils.x_axis_dates(times)
            plt.plot(x, elongations)
        plt.ylabel("Observed parcel elongation")
        plt.suptitle("Observations we're trying to fit")
        plt.show()
    
    def simulate_target(self):
        self._simulate_scenario(self.simdat, self.target_elongations)
    
    def simulate_fitted(self):
        self._simulate_scenario(self.simdat_fitted, self.fitted_elongations)
    
    def generate_target_jmap(self, quantity='flux'):
        simdat_stripped = copy.deepcopy(self.simdat)
        simdat_stripped.sc = simdat_stripped.sc.strip_units()
        simdat_stripped.parcels = [p.strip_units() for p in simdat_stripped.parcels]
        simdat_stripped.t = simdat_stripped.t.si.value
        bundle = generate_images(
            simdat_stripped, output_size_x=500, output_size_y=175,
            output_quantity=quantity)
        jmap = ops.extract_slices(
            bundle, 500, disable_pbar=True)
        return jmap
    
    def generate_fitted_jmap(self, quantity='flux'):
        simdat_stripped = copy.deepcopy(self.simdat_fitted)
        simdat_stripped.sc = simdat_stripped.sc.strip_units()
        simdat_stripped.parcels = [p.strip_units() for p in simdat_stripped.parcels]
        simdat_stripped.t = simdat_stripped.t.si.value
        bundle = generate_images(
            simdat_stripped, output_size_x=500, output_size_y=175,
            output_quantity=quantity)
        jmap = ops.extract_slices(
            bundle, 500, disable_pbar=True)
        return jmap


    def print_fit_output(self):
        print(self.res)
        

    def print_fit_summary(self):
        print("                Actual    |    Fitted      |     Delta")
        print(f"Riso:      {self.Riso:7.3f} | {self.fitted_Riso:7.3f} |{self.Riso - self.fitted_Riso:7.3f}")
        print(f"Tiso:      {self.Tiso:7.3f}     | {self.fitted_Tiso:7.3f}     |{self.Tiso - self.fitted_Tiso:7.3f}")
        print(f"Gamma:     {self.gamma:7.3f}        | {self.fitted_gamma:7.3f}        |{self.gamma - self.fitted_gamma:7.3f}")
        
        launch_thetas = self.launch_thetas.to(u.deg)
        fitted_thetas = self.fitted_lons.to(u.deg)
        print(f"Longitude: {launch_thetas[0]:7.3f}    | "
              f"{fitted_thetas[0]:7.3f}    |"
              f"{launch_thetas[0] - fitted_thetas[0]:7.3f}")
        for i in range(1, len(launch_thetas)):
            print(f"           {launch_thetas[i]:7.3f}    | "
                  f"{fitted_thetas[i]:7.3f}    |"
                  f"{launch_thetas[i] - fitted_thetas[i]:7.3f}")
        
        launch_times = self.launch_times.to(u.hr)
        fitted_times = self.fitted_times.to(u.hr)
        print(f"Launch t:  {launch_times[0]:7.3f}      | "
              f"{fitted_times[0]:7.3f}      |"
              f"{launch_times[0] - fitted_times[0]:7.3f}")
        for i in range(1, len(launch_times)):
            print(f"           {launch_times[i]:7.3f}      | "
                  f"{fitted_times[i]:7.3f}      |"
                  f"{launch_times[i] - fitted_times[i]:7.3f}")

    def fit_summary(self):
        seen_rs = []
        for i, (times_for_fit, launch_time) in enumerate(zip(
                self.times_for_fit, self.launch_times)):
            seen_rs.append(np.interp(
                times_for_fit,
                self.ts_actual + self.time_epoch + launch_time,
                self.rs_actual))

        plt.figure(figsize=(10,10))
        plt.suptitle("Fit results")
        
        plt.subplot(221)
        for i, (times, elongations, fitted_elongations) in enumerate(zip(
                self.times_for_fit,
                self.target_elongations,
                self.fitted_elongations)):
            x = plot_utils.x_axis_dates(times)
            plt.plot(x, elongations, color=f"C{i}", ls='--')
            if fitted_elongations is not None:
                plt.plot(x, fitted_elongations, color=f"C{i}", lw=1)
        legend_elements = [
            matplotlib.lines.Line2D([0], [0], color='k', ls='--'),
            matplotlib.lines.Line2D([0], [0], color='k', lw=1)
        ]
        plt.legend(legend_elements, ['Target', 'Fitted'])
        plt.ylabel("Elongation")
        
        plt.subplot(222)
        plt.plot(self.rs_actual.to(u.R_sun), self.vs_actual.to(u.km/u.s), label="Actual")
        plt.plot(self.rs_fitted.to(u.R_sun), self.vs_fitted.to(u.km/u.s), label="From fit")
        for i, seen_r in enumerate(seen_rs):
            plt.axvline(seen_r[0], ls=':', color=f"C{i}")
            plt.axvline(seen_r[-1], ls=':', color=f"C{i}")
        plt.xlabel("R$_\odot$")
        plt.ylabel("Plasma velocity (km/s)")
        plt.legend()
    
        plt.subplot(223)
        plt.plot(self.ts_actual.to(u.hr), self.rs_actual.to(u.R_sun),
                 label='Actual', color='k')
        plt.axvline(self.ts_actual.to(u.hr)[0], color='k', ls=':')
        for i, (t, ft) in enumerate(zip(self.launch_times, self.fitted_times)):
            if ft is None:
                continue
            tvals = (self.ts_fitted + (ft - t)).to(u.hr)
            plt.plot(tvals,
                     self.rs_fitted.to(u.R_sun), label='From fit',
                     color=f"C{i}", ls='--')
            plt.axvline(tvals[0], color=f"C{i}", ls=':')
        for i, seen_r in enumerate(seen_rs):
            plt.axhline(seen_r[0], ls=':', color=f"C{i}")
            plt.axhline(seen_r[-1], ls=':', color=f"C{i}")
        plt.ylabel("R$_\odot$")
        plt.xlabel("Time (hr)")
        plt.legend()
        
        all_times = np.unique(np.concatenate(self.times_for_fit))
        sc = self.simdat.sc.at(all_times)
        c_sc = SkyCoord(sc.x, sc.y, sc.z, unit='m',
                        representation_type='cartesian',
                        frame=HeliocentricInertial).transform_to(orbital_frame)
        c_sc_cart = c_sc.cartesian
        
        plt.subplot(224)
        plt.scatter(0, 0, c='yellow', s=100, zorder=18, edgecolors='black')
        plt.plot(
            c_sc_cart.x.to(u.R_sun),
            c_sc_cart.y.to(u.R_sun),
            zorder=9, lw=5)
    
        for i, (lon_actual, lon_fit) in enumerate(zip(
                self.launch_thetas, self.fitted_lons)):
            x = self.rs_actual.to(u.R_sun) * np.cos(lon_actual)
            y = self.rs_actual.to(u.R_sun) * np.sin(lon_actual)
            plt.plot(x, y, ls='--', color=f"C{i}")
            if lon_fit is not None:
                x = self.rs_fitted.to(u.R_sun) * np.cos(lon_fit)
                y = self.rs_fitted.to(u.R_sun) * np.sin(lon_fit)
                plt.plot(x, y, lw=1, color=f"C{i}")
        
        legend_elements = [
            matplotlib.lines.Line2D([0], [0], color='k', ls='--'),
            matplotlib.lines.Line2D([0], [0], color='k', lw=1)
        ]
        plt.legend(legend_elements, ['Actual', 'Fitted'])
        plt.gca().set_aspect('equal')
        plt.gca().set_facecolor('black')
        plt.xlabel("X ($R_\odot$)")
        plt.ylabel("Y ($R_\odot$)")
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)