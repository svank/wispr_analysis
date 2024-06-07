import copy
from dataclasses import dataclass

from astropy.coordinates import SkyCoord
import astropy.units as u
from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np

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
        return np.sqrt(self.v_sc**2 + self.v_pxy**2
                       - 2 * self.v_sc * self.v_pxy
                         * np.cos(180*u.deg - self.beta - self.gamma))
    
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
        return (-np.sqrt(self.v_sc**2 + self.v_pxy**2
                         - 2 * self.v_sc* self.v_pxy
                           * np.cos(180*u.deg - self.beta_prime - self.gamma)))
    
    @property
    def r_pxy(self):
        return self.r_sc * np.sin(self.epsilon) / np.sin(self.gamma)
    
    @property
    def gamma_prime(self):
        return None

    @property
    def gamma(self):
        return 180*u.deg - self.epsilon - self.delta_phi
    
    @property
    def beta_prime(self):
        return 180*u.deg - self.beta


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
    
    def get_intersect_vxy(self):
        intersects = self._get_intersects_vxy(
            self.delta_phi_c2, self.v_pxy_c2, self.delta_phi_c1, self.v_pxy_c1,
            self.con_state)
        intersects.extend(self._get_intersects_vxy(
            self.delta_phi_c3, self.v_pxy_c3, self.delta_phi_c1, self.v_pxy_c1,
            self.div_state))
        if len(intersects) == 0:
            return None
        if len(intersects) > 1:
            # Sometimes you get duplicates at a single "real" intersection if
            # the line from the numerical grid is a bit jittery
            delta_phis = [intersect.delta_phi for intersect in intersects]
            vpxys = [intersect.v_pxy for intersect in intersects]
            assert np.all((delta_phis - np.mean(delta_phis)) < 1 * u.deg)
            assert np.all((vpxys - np.mean(vpxys)) < 5 * u.km / u.s)
        return intersects[0]
    
    @classmethod
    def _get_intersects_vxy(cls, xs1, ys1, xs2, ys2, state):
        intersects = []
        for i in range(len(xs1)-1):
            x1 = xs1[i]
            x2 = xs1[i+1]
            y1 = ys1[i]
            y2 = ys1[i+1]
            x3, x4 = x1, x2
            y3 = np.interp(x3, xs2, ys2)
            y4 = np.interp(x4, xs2, ys2)
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
            state = state.copy()
            state.delta_phi = xint
            state.v_pxy = yint
            intersects.append(state)
        return intersects

    @property
    def v_p_c1(self):
        return self.vxy2vp(self.v_pxy_c1, self.delta_phi_c1)
    
    @property
    def v_p_c2(self):
        return self.vxy2vp(self.v_pxy_c2, self.delta_phi_c2)
    
    @property
    def v_p_c3(self):
        return self.vxy2vp(self.v_pxy_c3, self.delta_phi_c3)
    
def calc_constraints(measured_angles, cutoff_c2_variants=True):
    forward_elongation = planets.get_psp_forward_as_elongation(
        measured_angles.t0)
    forward = forward_elongation.transform_to('pspframe').lon
    sun = SkyCoord(0*u.deg, 0*u.deg, frame='helioprojective',
                   observer=forward_elongation.observer,
                   obstime=forward_elongation.obstime)
    to_sun = sun.transform_to('pspframe').lon
    beta = forward - measured_angles.stationary_point
    epsilon = measured_angles.stationary_point - to_sun
    kappa = 180 * u.deg - beta - epsilon
    
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
        dphis = np.linspace(2, 130, 600)*u.deg
        if cutoff_c2_variants:
            if state is con_state:
                dphis = dphis[dphis <= kappa]
                dphis = np.append(dphis, kappa)
            else:
                dphis = dphis[dphis > kappa]
                dphis = np.insert(dphis, 0, kappa)
        vpxys = np.linspace(0, 450, 1000) * u.km/u.s
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
        dalpha_dt_err=dalpha_dt_err, con_state=con_state, div_state=div_state)


class InteractiveClicker:
    def __init__(self, frames, wcs, times, plot_opts={}):
        self.frames = frames
        self.wcs = wcs
        self.plot_opts = plot_opts
        self.times = times
        
        self.clicked_alphas = []
        self.clicked_lons = []
        self.clicked_times = []
    
    def show(self):
        def draw_frame(i):
            plt.close('all')
            def onclick(event):
                try:
                    x = event.xdata
                    y = event.ydata
                    marker.set_data(x, y)
                    t = utils.to_timestamp(self.times[i]) * u.s
                    if t in self.clicked_times:
                        j = self.clicked_times.index(t)
                        self.clicked_alphas.pop(j)
                        self.clicked_lons.pop(j)
                        self.clicked_times.pop(j)
                    clicked_coord = self.wcs.pixel_to_world(x, y)
                    self.clicked_alphas.append(clicked_coord.lat)
                    self.clicked_lons.append(clicked_coord.lon)
                    self.clicked_times.append(t)
                except Exception as e:
                    plt.title(e)
            fig = plt.figure(figsize=(13, 13))
            plot_utils.plot_WISPR(
                self.frames[i], wcs=self.wcs, **self.plot_opts)
            plt.title(utils.to_timestamp(self.times[i], as_datetime=True)
                      .strftime('%Y %b %d, %H:%M'))
            marker_x, marker_y = np.nan, np.nan
            t = utils.to_timestamp(self.times[i]) * u.s
            if t in self.clicked_times:
                j = self.clicked_times.index(t)
                marker_x, marker_y = self.wcs.world_to_pixel_values(
                    self.clicked_lons[j], self.clicked_alphas[j])
            marker, = plt.plot(marker_x, marker_y, '+', color='C1')
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
        return interactive(draw_frame, i=(0, len(self.frames)))
    
    def conclude(self):
        observed_stationary_point = np.mean(u.Quantity(self.clicked_lons))
        
        alphas = u.Quantity(self.clicked_alphas)
        times = u.Quantity(self.clicked_times)
        sort = np.argsort(times)
        times = times[sort]
        alphas = alphas[sort]
        
        fit = np.polyfit(times.value, alphas.value, 1)
        dalpha_dt = fit[0] * alphas.unit / times.unit
        t0 = (times[0] + times[-1]) / 2
        tstart = times[0]
        tstop = times[-1]
        alpha = np.interp(t0, times, alphas)
        
        result = MeasuredAngles(
            stationary_point=observed_stationary_point,
            alpha=alpha,
            dalpha_dt=dalpha_dt,
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
        self.times = times
        
        self.clicked_alphas = []
        self.clicked_lons = []
        self.clicked_times = times * u.s
        
        for i in range(len(frames)):
            y, x = np.unravel_index(np.nanargmax(frames[i]), frames[i].shape)
            clicked_coord = wcs.pixel_to_world(x, y)
            self.clicked_alphas.append(clicked_coord.lat)
            self.clicked_lons.append(clicked_coord.lon)
    
    def show(self):
        raise NotImplementedError(
            "This method does not exist for AutoClicker")


@dataclass
class MeasuredAngles:
    stationary_point: u.Quantity
    alpha: u.Quantity
    dalpha_dt: u.Quantity
    t0: u.Quantity
    tstart: u.Quantity
    tstop: u.Quantity
    
    def __str__(self):
        return (f"MeasuredAngles(\n\tstationary_point="
                f"{self.stationary_point.value:.3f} "
                f"* u.Unit('{str(self.stationary_point.unit)}'),\n\t"
                f"alpha={self.alpha.value:.3f} "
                f"* u.Unit('{str(self.alpha.unit)}'),\n\t"
                f"dalpha_dt={self.dalpha_dt.value:.4e} "
                f"* u.Unit('{str(self.dalpha_dt.unit)}'),\n\t"
                f"t0={self.t0.value:.0f} "
                f"* u.Unit('{str(self.t0.unit)}'),\n\t"
                f"tstart={self.tstart.value:.0f} "
                f"* u.Unit('{str(self.tstart.unit)}'),\n\t"
                f"tstop={self.tstop.value:.0f} "
                f"* u.Unit('{str(self.tstop.unit)}'))")