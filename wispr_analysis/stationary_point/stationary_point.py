import copy
from dataclasses import dataclass

from astropy.coordinates import SkyCoord
import astropy.units as u
from ipywidgets import interactive
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from tqdm.auto import tqdm

from .. import orbital_frame, planets, plot_utils, utils


class StationaryPointState:

    def __init__(self, epsilon=np.nan, delta_phi=np.nan, beta=np.nan,
                 v_sc=np.nan, v_pxy=np.nan, r_sc=np.nan, alpha=np.nan,
                 theta=np.nan, v_pphi=0*u.m/u.s):
        self.epsilon = epsilon
        self.delta_phi = delta_phi
        self.beta = beta
        self.v_sc = v_sc
        self.v_pxy = v_pxy
        self.r_sc = r_sc
        self._alpha = alpha
        self._theta = theta
        self.v_pphi = v_pphi

    @property
    def v_pxy_constr1(self):
        return (np.sin(self.beta) * self.v_sc) / np.sin(self.gamma + self.psi)
    
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
        return np.sqrt(self.v_pr**2 + self.v_pphi**2)
    
    @property
    def v_prxy(self):
        return np.sqrt(self.v_pxy**2 - self.v_pphi**2)
    
    @property
    def v_pr(self):
        return self.v_prxy / np.cos(self.theta)

    @property
    def v_z(self):
        return self.v_pr * np.sin(self.theta)

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
        if self._alpha is np.nan:
            return np.tan(self.theta) * self.r_pxy
        return self.d_xy * np.tan(self.alpha)
    
    @property
    def alpha(self):
        if self._alpha is np.nan:
            return np.arctan2(self.d_z, self.d_xy).to(u.deg)
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        self._theta = np.nan

    @property
    def theta(self):
        if self._theta is np.nan:
            return np.arctan(self.d_z / self.r_pxy).to(u.deg)
        return self._theta
    
    @theta.setter
    def theta(self, value):
        self._theta = value
        self._alpha = np.nan

    @property
    def gamma_prime(self):
        return 180*u.deg - self.epsilon - self.delta_phi
    
    @property
    def gamma(self):
        return 180*u.deg - self.gamma_prime
    
    @property
    def psi(self):
        return np.arcsin(self.v_pphi / self.v_pxy)

    @property
    def dalpha_dt(self):
        return ((self.v_a * np.tan(self.alpha) + self.v_prxy * np.tan(self.theta))
                / (self.d_xy * (1 + np.tan(self.alpha)**2)) * u.rad)
    
    @property
    def kappa(self):
        return 180 * u.deg - self.beta - self.epsilon
    
    @property
    def delta(self):
        return 180 * u.deg - self.beta - self.gamma - self.psi

    def copy(self):
        return copy.deepcopy(self)


class DivergingStationaryPointState(StationaryPointState):
    @property
    def v_pxy_constr1(self):
        return (self.v_sc * np.sin(self.beta_prime)
                / np.sin(self.gamma - self.psi))

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
        return 180 * u.deg - self.beta_prime - self.gamma + self.psi


@njit(cache=True)
def _get_intersects_inner(xs1, ys1, xs2, ys2):
    intersects = []
    for i in range(len(xs1) - 1):
        x1 = xs1[i]
        x2 = xs1[i + 1]
        y1 = ys1[i]
        y2 = ys1[i + 1]
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        if np.any(np.isnan(np.array((x1, x2, y1, y2)))):
            continue
        for x3, x4, y3, y4 in zip(xs2[:-1], xs2[1:], ys2[:-1], ys2[1:]):
            if np.any(np.isnan(np.array((x3, x4, y3, y4)))):
                continue
            if x3 > x4:
                x3, x4 = x4, x3
                y3, y4 = y4, y3
            if x2 < x3 or x1 > x4 or min(y1, y2) > max(y3, y4) or max(y1, y2) < min(y3, y4):
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
            if x1 <= xint < x2:
                intersects.append((xint, yint))
    return intersects


def _get_intersects(xs1, ys1, xs2, ys2):
    x1v = xs1.value if isinstance(xs1, u.Quantity) else xs1
    x2v = xs2.value if isinstance(xs2, u.Quantity) else xs2
    y1v = ys1.value if isinstance(ys1, u.Quantity) else ys1
    y2v = ys2.value if isinstance(ys2, u.Quantity) else ys2
    intersects = _get_intersects_inner(xs1, ys1, xs2, ys2)
    if not len(intersects):
        return intersects
    xint, yint = zip(*intersects)
    if isinstance(xs1, u.Quantity):
        xint <<= xs1.unit
    if isinstance(ys1, u.Quantity):
        yint <<= ys1.unit
    intersects = list(zip(xint, yint))
    return intersects


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
    
    @classmethod
    def _get_intersects_vxy(cls, xs1, ys1, xs2, ys2, con_state, div_state):
        intersects = _get_intersects(xs1, ys1, xs2, ys2)
        states = []
        for dphi_int, vpxy_int in intersects:
            # We just need a value for psi
            state = con_state.copy()
            state.delta_phi = dphi_int
            state.v_pxy = vpxy_int
            psi = state.psi
            # To find which state to copy
            state = con_state if dphi_int + psi < state.kappa else div_state
            state = state.copy()
            state.delta_phi = dphi_int
            state.v_pxy = vpxy_int
            states.append(state)
        return states
    
    def get_intersect_vxy(self, return_all=False):
        intersects = self._get_intersects_vxy(
            np.concatenate((self.delta_phi_c2, self.delta_phi_c3)),
            np.concatenate((self.v_pxy_c2, self.v_pxy_c3)),
            self.delta_phi_c1, self.v_pxy_c1,
            self.con_state, self.div_state)
        if len(intersects) == 0:
            return None
        if return_all:
            return intersects
        # Sometimes you get duplicates at a single "real" intersection if
        # the line from the numerical grid is a bit jittery, so let's just
        # take the mean as our starting point.
        dphi = np.mean(
            u.Quantity([intersect.delta_phi for intersect in intersects]))
        vpxy = np.mean(
            u.Quantity([intersect.v_pxy for intersect in intersects]))
        
        intersect = intersects[0]
        intersect.delta_phi = dphi
        intersect.v_pxy = vpxy
        
        return intersect
    
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


@dataclass
class ThreeConstraintsResult:
    delta_phi_c1: u.Quantity
    theta_c1: u.Quantity
    delta_phi_c2: u.Quantity
    theta_c2: u.Quantity
    dphi_grid: u.Quantity
    theta_grid: u.Quantity
    alpha_grid: u.Quantity
    dalpha_dt_grid: u.Quantity
    con_state: StationaryPointState
    div_state: StationaryPointState
    measured_angles: "MeasuredAngles"
    
    @classmethod
    def _get_intersects_theta(cls, xs1, ys1, xs2, ys2, con_state, div_state):
        intersects = _get_intersects(xs1, ys1, xs2, ys2)
        states = []
        for dphi_int, theta_int in intersects:
            # We just need a value for psi
            state = con_state.copy()
            vgood = np.isfinite(state.v_pxy.ravel())
            v_pxy_int = np.interp(
                dphi_int, state.delta_phi[0, vgood], state.v_pxy[0, vgood])
            state.v_pxy = v_pxy_int
            state.delta_phi = dphi_int
            state.theta = theta_int
            psi = state.psi
            # To find which state to copy
            state = con_state if dphi_int + psi < state.kappa else div_state
            state = state.copy()
            state.delta_phi = dphi_int
            state.theta = theta_int
            state.v_pxy = v_pxy_int
            states.append(state)
        return states
    
    def get_intersect(self, return_all=False):
        intersects = self._get_intersects_theta(
            self.delta_phi_c2,
            self.theta_c2,
            self.delta_phi_c1, self.theta_c1,
            self.con_state, self.div_state)
        if len(intersects) == 0:
            return None
        if return_all:
            return intersects
        # Sometimes you get duplicates at a single "real" intersection if
        # the line from the numerical grid is a bit jittery, so let's just
        # take the mean as our starting point.
        dphi = np.mean(
            u.Quantity([intersect.delta_phi for intersect in intersects]))
        theta = np.mean(
            u.Quantity([intersect.theta for intersect in intersects]))
        
        intersect = intersects[0]
        intersect.delta_phi = dphi
        intersect.theta = theta
        
        return intersect
    
    def plot(self, mark_intersect=True, ax=None, show_alpha_grid=False,
             show_dalpha_dt_grid=False):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.delta_phi_c1, self.theta_c1)
        ax.plot(self.delta_phi_c2, self.theta_c2)
        ax.set_xlabel("Assumed separation in heliographic longitude")
        ax.set_ylabel("Assumed plasma launch latitude")
        if show_alpha_grid:
            ax.pcolormesh(self.dphi_grid,
                          self.theta_grid,
                          (self.alpha_grid -
                           self.measured_angles.alpha).value,
                          cmap='bwr', vmin=-30, vmax=30, zorder=-20)
        if show_dalpha_dt_grid:
            ax.pcolormesh(self.dphi_grid,
                          self.theta_grid,
                          (self.dalpha_dt_grid -
                           self.measured_angles.dalpha_dt).value,
                          cmap='bwr', vmin=-1e-3, vmax=1e-3, zorder=-20)
        if mark_intersect:
            intersect = self.get_intersect()
            if intersect:
                ax.scatter(intersect.delta_phi, intersect.theta, color='C3')
    

def measurements_to_states(measured_angles, vphi=0*u.km/u.s):
    forward_elongation = planets.get_psp_forward_as_elongation(
        measured_angles.t0)
    forward = forward_elongation.transform_to('pspframe').lon
    sun = SkyCoord(0 * u.deg, 0 * u.deg, frame='helioprojective',
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
        alpha=measured_angles.alpha, v_pphi=vphi)
    div_state = DivergingStationaryPointState(
        epsilon=epsilon, beta=beta, v_sc=v_sc, r_sc=r_sc,
        alpha=measured_angles.alpha, v_pphi=vphi)
    
    return con_state, div_state


def calc_constraints(measured_angles, cutoff_c2_variants=True,
                     vphi=0 * u.km / u.s):
    con_state, div_state = measurements_to_states(measured_angles, vphi)
    
    dphis = np.linspace(2, 130, 300) * u.deg
    vpxys = np.linspace(0, 450, 600) * u.km / u.s
    dphi_grid, vpxy_grid = np.meshgrid(dphis, vpxys)
    
    con_state.delta_phi = dphi_grid
    con_state.v_pxy = vpxy_grid
    
    vpxy_c1 = _solve_on_grid(con_state.v_pxy_constr1, vpxy_grid, vpxys)
    
    delta_phis = [dphis]
    v_pxys = [vpxy_c1]
    
    for state in (con_state, div_state):
        # TODO: Sometimes, the curve we compute from the grid points will have
        # little jitters that cause one intersect to actually be multiple
        # intersects. Those could be deduplicated somehow---maybe use them as
        # seeds to iteratively find the "true" intersect?
        state.delta_phi = dphi_grid
        state.v_pxy = vpxy_grid
        
        dalpha_dt_grid = state.dalpha_dt
        
        vbest = _solve_on_grid(dalpha_dt_grid, measured_angles.dalpha_dt, vpxys)
        dphis = dphi_grid[0]
        
        if cutoff_c2_variants:
            s = state.copy()
            s.delta_phi = dphis
            s.v_pxy = vbest
            if state is con_state:
                good = dphis + s.psi <= state.kappa
            else:
                good = dphis + s.psi > state.kappa
            vbest = vbest[good]
            dphis = dphis[good]
        
        delta_phis.append(dphis)
        v_pxys.append(vbest)
    
    def vxy2vp(vxy, dphi):
        # con_state and div_state will produce the same results here
        s = con_state.copy()
        s.delta_phi = dphi
        s.v_pxy = vxy
        return s.v_p
    
    def vp2vxy(vp, dphi):
        # con_state and div_state will produce the same results here
        s = con_state.copy()
        s.delta_phi = dphi
        vr = np.sqrt(vp**2 - s.v_pphi**2)
        vr_xy = vr * np.cos(s.theta)
        vxy = np.sqrt(vr_xy**2 + s.v_pphi**2)
        return vxy
    
    return ConstraintsResult(
        delta_phi_c1=delta_phis[0], v_pxy_c1=v_pxys[0],
        delta_phi_c2=delta_phis[1], v_pxy_c2=v_pxys[1],
        delta_phi_c3=delta_phis[2], v_pxy_c3=v_pxys[2],
        vxy2vp=vxy2vp, vp2vxy=vp2vxy, dphi_grid=dphis, vpxy_grid=vpxys,
        dalpha_dt_err=None, con_state=con_state, div_state=div_state,
        measured_angles=measured_angles)


def calc_three_constraints(measured_angles, vphi=0 * u.km / u.s):
    con_state, div_state = measurements_to_states(
        measured_angles, vphi)
    
    dphis = np.linspace(2, 130, 500) * u.deg
    vpxys = np.linspace(0, 450, 600) * u.km / u.s
    dphi_grid, vpxy_grid = np.meshgrid(dphis, vpxys)
    
    con_state.delta_phi = dphi_grid
    con_state.v_pxy = vpxy_grid
    
    x, vpxy_c1 = _solve_on_grid_flexibly(
        con_state.v_pxy_constr1, vpxy_grid, dphis, vpxys)
    vpxy_c1 = np.concatenate((
        [np.nan] * np.sum(dphis < np.min(x)),
        vpxy_c1,
        [np.nan] * np.sum(dphis > np.max(x))
    ))
    
    thetas = np.linspace(0, 90, 500) * u.deg
    dphi_grid, theta_grid = np.meshgrid(dphis, thetas)
    con_state.delta_phi = dphi_grid
    con_state.v_pxy = vpxy_c1.reshape((1, -1))
    con_state.theta = theta_grid
    div_state.delta_phi = dphi_grid
    div_state.v_pxy = vpxy_c1.reshape((1, -1))
    div_state.theta = theta_grid
    
    is_converging = dphi_grid + con_state.psi <= con_state.kappa
    
    alpha = np.where(is_converging, con_state.alpha, div_state.alpha)
    dalpha_dt = np.where(is_converging, con_state.dalpha_dt,
                         div_state.dalpha_dt)
    
    dphi_c1, theta_c1 = _solve_on_grid_flexibly(alpha, measured_angles.alpha,
                                   dphis, thetas)
    dphi_c2, theta_c2 = _solve_on_grid_flexibly(dalpha_dt, measured_angles.dalpha_dt,
                                     dphis, thetas)
    
    return ThreeConstraintsResult(
        delta_phi_c1=dphi_c1, theta_c1=theta_c1,
        delta_phi_c2=dphi_c2, theta_c2=theta_c2,
        dphi_grid=dphi_grid, theta_grid=theta_grid,
        con_state=con_state, div_state=div_state,
        alpha_grid=alpha, dalpha_dt_grid=dalpha_dt,
        measured_angles=measured_angles)


def _solve_on_grid(grid_of_values, target_value, y_values):
    errors = grid_of_values - target_value
    err_range = np.ptp(np.abs(errors[np.isfinite(errors)]))
    result = []
    for i in range(grid_of_values.shape[1]):
        strip = np.abs(errors[:, i])
        if np.all(np.isnan(strip)):
            result.append(np.nan * y_values.unit)
        else:
            j = np.nanargmin(strip)
            if (j == 0 or j == len(strip) - 1 or np.abs(strip[j]) > 0.01 * err_range
                or any(np.isnan(strip[j-1:j+2]))):
                result.append(np.nan * y_values.unit)
            else:
                result.append(y_values[j])
    return u.Quantity(result)


def _solve_on_grid_flexibly(grid_of_values, target_value, x_values, y_values):
    errors = grid_of_values - target_value
    signs = np.sign(errors)
    points = []
    for i in range(grid_of_values.shape[1]):
        sign_strip = signs[:, i]
        sign_changes = np.nonzero(sign_strip[:-1] != sign_strip[1:])[0]
        ys = []
        for sign_change in sign_changes:
            region = errors[sign_change:sign_change+2, i]
            if np.any(np.isnan(region)):
                continue
            if region[1] > region[0]:
                ys.append(np.interp(0, region, [sign_change, sign_change+1]))
            else:
                ys.append(np.interp(0, region[::-1], [sign_change+1, sign_change]))
        points.append(ys)
    i = 0
    while not len(points[i]):
        i += 1
    j = points[i].pop()
    res_x = [i]
    res_y = [j]
    di = 1
    while True:
        delta_left = delta_right = delta_vert = np.inf
        if i - di >= 0 and len(points[i-di]):
            deltas = np.abs(np.array(points[i-di]) - j)
            closest_left = np.argmin(deltas)
            delta_left = deltas[closest_left]
        if i + di < len(points) and len(points[i+di]):
            deltas = np.abs(np.array(points[i+di]) - j)
            closest_right = np.argmin(deltas)
            delta_right = deltas[closest_right]
        if di == 1 and len(points[i]):
            deltas = np.abs(np.array(points[i]) - j)
            closest_vert = np.argmin(deltas)
            delta_vert = deltas[closest_vert]
        if delta_left == np.inf and delta_right == np.inf and delta_vert == np.inf:
            if i - di < 0 and i + di >= len(points):
                break
            di += 1
            continue
        if delta_left < delta_right and delta_left < delta_vert:
            i = i - di
            j = points[i].pop(closest_left)
        elif delta_right < delta_left and delta_right < delta_vert:
            i = i + di
            j = points[i].pop(closest_right)
        else:
            j = points[i].pop(closest_vert)
        res_x.append(i)
        res_y.append(j)
        di = 1
    if any(len(p) for p in points):
        # raise RuntimeError("Not all points connected")
        print("Not all points connected")
    res_x = u.Quantity([x_values[i] for i in res_x])
    res_y = u.Quantity([_frac_index_to_value(i, y_values) for i in res_y])
    return res_x, res_y

def _frac_index_to_value(index, values):
    if index == int(index):
        return values[int(index)]
    i = int(np.floor(index))
    j = int(np.ceil(index))
    fraction = index - i
    return (1 - fraction) * values[i] + fraction * values[j]


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