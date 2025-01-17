import copy
from dataclasses import dataclass

from astropy.coordinates import SkyCoord
import astropy.units as u
from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from tqdm.auto import tqdm

from .. import orbital_frame, planets, plot_utils, utils


@dataclass
class StationaryPointState:
    epsilon: u.Quantity = np.nan
    delta_phi: u.Quantity = np.nan
    beta: u.Quantity = np.nan
    v_sc: u.Quantity = np.nan
    v_pxy: u.Quantity = np.nan
    r_sc: u.Quantity = np.nan
    alpha: u.Quantity = np.nan

    @property
    def v_pxy_constr1(self):
        return (self.v_sc * np.sin(self.beta)
                / np.sin(self.epsilon + self.delta_phi))

    @property
    def v_a(self):
        return self._v_a(1)
    
    def _v_a(self, factor):
        return (factor * np.sqrt(self.v_sc**2 + self.v_pxy**2
                       - 2 * self.v_sc * self.v_pxy
                         * np.cos(self.delta))
                )
    
    @property
    def v_p(self):
        return self.v_pxy / np.cos(self.theta)

    @property
    def d_xy(self):
        return self.r_sc * np.sin(self.delta_phi) / np.sin(self.gamma_prime)
    
    @property
    def d_p_sc(self):
        return np.sqrt(self.d_xy**2 + self.d_z**2)

    @property
    def r_pxy(self):
        return self.r_sc * np.sin(self.epsilon) / np.sin(self.gamma_prime)
    
    @property
    def r_p(self):
        return self.r_pxy / np.cos(self.theta)

    @property
    def d_z(self):
        return self.d_xy * np.tan(self.alpha)

    @property
    def gamma_prime(self):
        return 180*u.deg - self.epsilon - self.delta_phi
    
    @property
    def gamma(self):
        return 180*u.deg - self.gamma_prime

    @property
    def theta(self):
        return np.arctan(self.d_z / self.r_pxy).to(u.deg)

    @property
    def v_z(self):
        return self.v_pxy * np.tan(self.theta)

    @property
    def dalpha_dt(self):
        return ((self.v_a * np.tan(self.alpha) + self.v_pxy * np.tan(self.theta))
                / (self.d_xy * (1 + np.tan(self.alpha)**2)) * u.rad)
    
    @property
    def kappa(self):
        return 180 * u.deg - self.beta - self.epsilon
    
    @property
    def delta(self):
        return 180 * u.deg - self.beta - self.gamma

    def copy(self):
        return copy.deepcopy(self)


class DivergingStationaryPointState(StationaryPointState):
    @property
    def v_pxy_constr1(self):
        return (self.v_sc * np.sin(self.beta_prime)
                / np.sin(self.epsilon + self.delta_phi))

    @property
    def d_xy(self):
        return self.r_sc * np.sin(self.delta_phi) / np.sin(self.gamma)
    
    @property
    def v_a(self):
        return self._v_a(-1)
    
    @property
    def r_pxy(self):
        return self.r_sc * np.sin(self.epsilon) / np.sin(self.gamma)
    
    @property
    def gamma_prime(self):
        raise NotImplementedError()

    @property
    def gamma(self):
        return 180*u.deg - self.epsilon - self.delta_phi
    
    @property
    def beta_prime(self):
        return 180*u.deg - self.beta
    
    @property
    def delta(self):
        return 180 * u.deg - self.beta_prime - self.gamma


@dataclass
class ConstraintsResult:
    delta_phi_c1: u.Quantity
    v_pxy_c1: u.Quantity
    delta_phi_c2: u.Quantity
    v_pxy_c2: u.Quantity
    delta_phi_c3: u.Quantity
    v_pxy_c3: u.Quantity
    vxy2vp: None
    vp2vxy: None
    dphi_grid: u.Quantity
    vpxy_grid: u.Quantity
    dalpha_dt_err: u.Quantity
    con_state: StationaryPointState
    div_state: StationaryPointState
    measured_angles: "MeasuredAngles"
    
    def get_intersect_vxy(self):
        intersects = self._get_intersects_vxy(
            np.concatenate((self.delta_phi_c2, self.delta_phi_c3)),
            np.concatenate((self.v_pxy_c2, self.v_pxy_c3)),
            self.delta_phi_c1, self.v_pxy_c1,
            self.con_state, self.div_state)
        # We'll take the intersection on our grid as a starting point to
        # iteratively find the "true" intersection.
        if len(intersects) == 0:
            return None
        # Sometimes you get duplicates at a single "real" intersection if
        # the line from the numerical grid is a bit jittery, so let's just
        # take the mean as our starting point.
        start_dphi = np.mean(
            u.Quantity([intersect.delta_phi for intersect in intersects]))
        con_state = self.con_state.copy()
        div_state = self.div_state.copy()
        def calc_resid(delta_phi):
            delta_phi = delta_phi * u.deg
            if delta_phi < con_state.kappa:
                state = con_state
            else:
                state = div_state
            state.delta_phi = delta_phi
            state.v_pxy = state.v_pxy_constr1
            return (state.dalpha_dt - self.measured_angles.dalpha_dt).value
        start = start_dphi.to_value(u.deg)
        try:
            root = scipy.optimize.root_scalar(
                calc_resid, bracket=[0, 130],x0=start)
        except ValueError:
            # If there's a discontinuity in the first constraint, minimizing
            # over the whole range will fail
            root = scipy.optimize.root_scalar(
                calc_resid, bracket=[start - 2, start + 2], x0=start)
        
        intersect = intersects[0]
        intersect.delta_phi = root.root * u.deg
        intersect.v_pxy = intersect.v_pxy_constr1
        
        return intersect
    
    @classmethod
    def _get_intersects_vxy(cls, xs1, ys1, xs2, ys2, con_state, div_state):
        intersects = []
        for i in range(len(xs1)-1):
            x1 = xs1[i]
            x2 = xs1[i+1]
            y1 = ys1[i]
            y2 = ys1[i+1]
            x3, x4 = x1, x2
            y3 = np.interp(x3, xs2, ys2, left=np.nan, right=np.nan)
            y4 = np.interp(x4, xs2, ys2, left=np.nan, right=np.nan)
            if np.any(np.isnan(u.Quantity((y1, y2, y3, y4)))):
                continue
            if min(y3, y4) > max(y1, y2) or min(y1, y2) > max(y3, y4):
                continue
            # a1x + b1y = c1
            a1 = y2 - y1
            b1 = x1 - x2
            c1 = a1 * x1 + b1 * y1
            
            # a2x + b2y = c2
            a2 = y4 - y3
            b2 = x3 - x4
            c2 = a2 * x3 + b2 * y3
            
            det = a1 * b2 - a2 * b1
            
            if det == 0:
                continue
            
            xint = ((b2 * c1) - (b1 * c2)) / det
            yint = ((a1 * c2) - (a2 * c1)) / det
            if xint < x1 or xint > x2:
                continue
            state = con_state if xint < con_state.kappa else div_state
            state = state.copy()
            state.delta_phi = xint
            state.v_pxy = yint
            intersects.append(state)
        return intersects
    
    def plot(self, vel_in_plane=True, mark_intersect=True, ax=None,
             show_full_c2=False):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.delta_phi_c1,
                self.v_pxy_c1 if vel_in_plane else self.v_p_c1)
        ax.plot(self.delta_phi_c2,
                self.v_pxy_c2 if vel_in_plane else self.v_p_c2)
        ax.plot(self.delta_phi_c3,
                self.v_pxy_c3 if vel_in_plane else self.v_p_c3)
        ax.set_xlabel("Assumed separation in heliographic longitude")
        if vel_in_plane:
            ax.set_ylabel("Plasma in-plane velocity component (km/s)")
        else:
            ax.set_ylabel("Plasma radial velocity (km/s)")
        if show_full_c2:
            if show_full_c2 not in ('con', 'div', 'both'):
                raise ValueError(
                    "show_full_c2 must be one of 'con', 'div', 'both'")
            if show_full_c2 == 'both':
                states = [self.div_state, self.con_state]
            else:
                states = ([self.con_state] if show_full_c2 == 'con' else
                          [self.div_state])
            for state in states:
                ax.pcolormesh(state.delta_phi,
                              state.v_pxy if vel_in_plane else state.v_p,
                              (state.dalpha_dt -
                               self.measured_angles.dalpha_dt).value,
                              cmap='bwr', vmin=-2e-5, vmax=2e-5, zorder=-20)
        if mark_intersect:
            intersect = self.get_intersect_vxy()
            if intersect:
                ax.scatter(intersect.delta_phi,
                           intersect.v_pxy if vel_in_plane else intersect.v_p,
                           color='C3')
    
    @property
    def v_p_c1(self):
        return self.vxy2vp(self.v_pxy_c1, self.delta_phi_c1)
    
    @property
    def v_p_c2(self):
        return self.vxy2vp(self.v_pxy_c2, self.delta_phi_c2)
    
    @property
    def v_p_c3(self):
        return self.vxy2vp(self.v_pxy_c3, self.delta_phi_c3)


def calc_constraints(measured_angles, cutoff_c2_variants=True, vphi=0*u.km/u.s):
    forward_elongation = planets.get_psp_forward_as_elongation(
        measured_angles.t0)
    forward = forward_elongation.transform_to('pspframe').lon
    sun = SkyCoord(0*u.deg, 0*u.deg, frame='helioprojective',
                   observer=forward_elongation.observer,
                   obstime=forward_elongation.obstime)
    to_sun = sun.transform_to('pspframe').lon
    beta = forward - measured_angles.stationary_point
    epsilon = measured_angles.stationary_point - to_sun
    
    psp = planets.locate_psp(measured_angles.t0)
    r_sc = psp.cartesian.norm()
    v_sc = psp.cartesian.differentials['s'].norm()

    con_state = StationaryPointState(
        epsilon=epsilon, beta=beta, v_sc=v_sc, r_sc=r_sc,
        alpha=measured_angles.alpha)
    div_state = DivergingStationaryPointState(
        epsilon=epsilon, beta=beta, v_sc=v_sc, r_sc=r_sc,
        alpha=measured_angles.alpha)
    
    delta_phis = [np.arange(0, 130, 1) * u.deg]
    con_state.delta_phi = delta_phis[0]
    v_pxys = [con_state.v_pxy_constr1]
    
    for state in (con_state, div_state):
        # TODO: Sometimes, the curve we compute from the grid points will have
        # little jitters that cause one intersect to actually be multiple
        # intersects. Those could be deduplicated somehow---maybe use them as
        # seeds to iteratively find the "true" intersect?
        dphis = np.linspace(2, 130, 300) * u.deg
        if cutoff_c2_variants:
            if state is con_state:
                dphis = dphis[dphis <= state.kappa]
                dphis = np.append(dphis, state.kappa)
            else:
                dphis = dphis[dphis > state.kappa]
                dphis = np.insert(dphis, 0, state.kappa)
        vpxys = np.linspace(0, 450, 600) * u.km/u.s
        dphi_grid, vpxy_grid = np.meshgrid(dphis, vpxys)
        state.delta_phi = dphi_grid
        state.v_pxy = vpxy_grid
        dalpha_dt_grid = state.dalpha_dt
        
        dalpha_dt_err = dalpha_dt_grid - measured_angles.dalpha_dt
        err_range = np.ptp(np.abs(dalpha_dt_err))
        
        # phibest = []
        # for i in range(len(vpxys)):
        #     strip = np.abs(dalpha_dt_err[i])
        #     j = np.argmin(strip)
        #     if j == 0 or j == len(strip) - 1 or strip[j] > 0.01 * err_range:
        #         phibest.append(np.nan * u.deg)
        #     else:
        #         phibest.append(dphis[j])
        
        # phibest = u.Quantity(phibest)
        # delta_phi_c2 = phibest
        # v_pxy_c2 = vpxys
        
        vbest = []
        for i in range(len(dphis)):
            strip = np.abs(dalpha_dt_err[:, i])
            j = np.argmin(strip)
            if j == 0 or j == len(strip) - 1 or strip[j] > 0.01 * err_range:
                vbest.append(np.nan * u.m/u.s)
            else:
                vbest.append(vpxys[j])
        
        vbest = u.Quantity(vbest)
        delta_phis.append(dphis)
        v_pxys.append(vbest)

    def vxy2vp(vxy, dphi):
        # con_state and div_state will produce the same results here
        s = con_state.copy()
        s.delta_phi = dphi
        vp = vxy / np.cos(s.theta)
        return vp
    def vp2vxy(vp, dphi):
        # con_state and div_state will produce the same results here
        s = con_state.copy()
        s.delta_phi = dphi
        vxy = vp * np.cos(s.theta)
        return vxy

    return ConstraintsResult(
        delta_phi_c1=delta_phis[0], v_pxy_c1=v_pxys[0],
        delta_phi_c2=delta_phis[1], v_pxy_c2=v_pxys[1],
        delta_phi_c3=delta_phis[2], v_pxy_c3=v_pxys[2],
        vxy2vp=vxy2vp, vp2vxy=vp2vxy, dphi_grid=dphis, vpxy_grid=vpxys,
        dalpha_dt_err=dalpha_dt_err, con_state=con_state, div_state=div_state, measured_angles=measured_angles)


class InteractiveClicker:
    def __init__(self, frames, wcs, times, plot_opts={}):
        self.frames = frames
        self.wcs = wcs
        self.plot_opts = plot_opts
        self.times = utils.to_timestamp(times) << u.s
        
        self.clicked_alphas = {}
        self.clicked_lons = {}
        self.clicked_times = []
    
    def show(self):
        def draw_frame(i):
            plt.close('all')
            t = self.times[i]
            
            def onclick(event):
                try:
                    x = event.xdata
                    y = event.ydata
                    marker.set_data([x], [y])
                    clicked_coord = self.wcs.pixel_to_world(x, y)
                    self.clicked_alphas[t] = clicked_coord.lat
                    self.clicked_lons[t] = clicked_coord.lon
                    if t not in self.clicked_times:
                        self.clicked_times.append(t)
                except Exception as e:
                    plt.title(e)
            
            fig = plt.figure(figsize=(13, 12))
            plot_utils.plot_WISPR(
                self.frames[i], wcs=self.wcs, **self.plot_opts)
            plt.title(utils.from_timestamp(t))
            marker_x, marker_y = np.nan, np.nan
            if t in self.clicked_times:
                marker_x, marker_y = self.wcs.world_to_pixel_values(
                    self.clicked_lons[t], self.clicked_alphas[t])
            marker, = plt.plot(marker_x, marker_y, '+', color='C1')
            befores = []
            afters = []
            for time in self.clicked_times:
                if time < t:
                    befores.append(u.Quantity(
                        (self.clicked_lons[time], self.clicked_alphas[time])))
                elif time > t:
                    afters.append(u.Quantity(
                        (self.clicked_lons[time], self.clicked_alphas[time])))
            if len(befores):
                x, y = self.wcs.world_to_pixel_values(*u.Quantity(befores).T)
                plt.scatter(x, y, marker='+', color='C0')
            if len(afters):
                x, y = self.wcs.world_to_pixel_values(*u.Quantity(afters).T)
                plt.scatter(x, y, marker='+', color='C2')
            
            fig.canvas.mpl_connect('button_press_event', onclick)
            
            def onpress(event):
                try:
                    if event.key == '[':
                        if widgets.children[0].value > 0:
                            widgets.children[0].value -= 1
                    elif event.key == ']':
                        if widgets.children[0].value < len(self.times) - 1:
                            widgets.children[0].value += 1
                except Exception as e:
                    plt.title(e)
            
            fig.canvas.mpl_connect('key_press_event', onpress)
            plt.show()
        
        widgets = interactive(draw_frame, i=(0, len(self.frames)))
        return widgets
    
    def conclude(self):
        observed_stationary_point = np.mean(
            u.Quantity(list(self.clicked_lons.values())))
        observed_stationary_point_sigma = np.std(
            u.Quantity(list(self.clicked_lons.values())))
        
        times = u.Quantity(self.clicked_times)
        times = np.sort(times)
        alphas = u.Quantity([self.clicked_alphas[t] for t in times])
        t0 = (times[0] + times[-1]) / 2
        times -= t0
        
        (dalpha_dt, alpha), cov = np.polyfit(
            times.value, alphas.value, 1, cov=True)
        dalpha_dt <<= (alphas.unit / times.unit)
        alpha <<= alphas.unit
        
        tstart = times[0]
        tstop = times[-1]
        
        result = MeasuredAngles(
            stationary_point=observed_stationary_point,
            stationary_point_std=observed_stationary_point_sigma,
            alpha=alpha,
            dalpha_dt=dalpha_dt,
            alpha_cov_matrix=cov,
            t0=t0,
            tstart=tstart,
            tstop=tstop)
        
        return result


class AutoClicker(InteractiveClicker):
    """
    A variant of InteractiveClicker that automatically "clicks" the peak pixel.
    
    Results can be collected with ``conclude``.
    """
    
    def __init__(self, frames, wcs, times, plot_opts={}):
        self.frames = frames
        self.wcs = wcs
        self.plot_opts = plot_opts
        self.times = utils.to_timestamp(times) << u.s
        
        self.clicked_alphas = {}
        self.clicked_lons = {}
        self.clicked_times = times
        
        for i in range(len(frames)):
            y, x = np.unravel_index(np.nanargmax(frames[i]), frames[i].shape)
            clicked_coord = wcs.pixel_to_world(x, y)
            self.clicked_alphas[self.times[i]] = clicked_coord.lat
            self.clicked_lons[self.times[i]] = clicked_coord.lon
    
    def show(self):
        raise NotImplementedError(
            "This method does not exist for AutoClicker")


@dataclass
class MeasuredAngles:
    stationary_point: u.Quantity
    stationary_point_std: u.Quantity
    alpha: u.Quantity
    dalpha_dt: u.Quantity
    alpha_cov_matrix: np.ndarray
    t0: u.Quantity
    tstart: u.Quantity
    tstop: u.Quantity
    
    def __str__(self):
        return (f"MeasuredAngles(\n\t"
                f"stationary_point="
                f"{self.stationary_point.value:.3f} "
                f"* u.Unit('{str(self.stationary_point.unit)}'),\n\t"
                f"stationary_point_std="
                f"{self.stationary_point_std.value:.3f} "
                f"* u.Unit('{str(self.stationary_point_std.unit)}'),\n\t"
                f"alpha={self.alpha.value:.3f} "
                f"* u.Unit('{str(self.alpha.unit)}'),\n\t"
                f"dalpha_dt={self.dalpha_dt.value:.4e} "
                f"* u.Unit('{str(self.dalpha_dt.unit)}'),\n\t"
                f"alpha_cov_matrix=np.{repr(self.alpha_cov_matrix)},\n\t"
                f"t0={self.t0.value:.0f} "
                f"* u.Unit('{str(self.t0.unit)}'),\n\t"
                f"tstart={self.tstart.value:.0f} "
                f"* u.Unit('{str(self.tstart.unit)}'),\n\t"
                f"tstop={self.tstop.value:.0f} "
                f"* u.Unit('{str(self.tstop.unit)}'))")


@dataclass
class InferredValues:
    vp: u.Quantity
    vpxy: u.Quantity
    delta_phi: u.Quantity
    rp: u.Quantity
    theta: u.Quantity


def find_uncertainty(measured_angles, n_samples=1000, progress_bar=True):
    vps = []
    vpxys = []
    delta_phis = []
    rps = []
    thetas = []
    iterations = range(n_samples)
    if progress_bar:
        iterations = tqdm(iterations)
    for _ in iterations:
        ma_sample = copy.deepcopy(measured_angles)
        ma_sample.stationary_point = np.random.normal(
            ma_sample.stationary_point.value,
            ma_sample.stationary_point_std.value) << ma_sample.stationary_point.unit
        dalpha_dt, alpha = np.random.multivariate_normal(
            [ma_sample.dalpha_dt.value, ma_sample.alpha.value],
            ma_sample.alpha_cov_matrix)
        ma_sample.dalpha_dt = dalpha_dt << ma_sample.dalpha_dt.unit
        ma_sample.alpha = alpha << ma_sample.alpha.unit
        
        res = calc_constraints(ma_sample)
        intersect = res.get_intersect_vxy()
        
        vps.append(intersect.v_p)
        vpxys.append(intersect.v_pxy)
        delta_phis.append(intersect.delta_phi)
        thetas.append(intersect.theta)
        rps.append(intersect.r_p)
    vps = u.Quantity(vps)
    vpxys = u.Quantity(vpxys)
    delta_phis = u.Quantity(delta_phis)
    rps = u.Quantity(rps)
    thetas = u.Quantity(thetas)
    
    return InferredValues(
        vp=vps, vpxy=vpxys, delta_phi=delta_phis, rp=rps, theta=thetas)