import copy
import dataclasses
from itertools import repeat
import warnings

from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
import numba
import numpy as np
import reproject
import scipy
from sunpy.coordinates import (
    HeliocentricInertial, Helioprojective, NorthOffsetFrame)
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from .. import planets, plot_utils, utils


FIXED_ANGLE_TARGET_RANGE = (80, 460)


def _extract_slice(image, wcs, n, is_inner):
    output_wcs = OrbitalSliceWCS(wcs, n, is_inner)
    slice = reproject.reproject_adaptive((image, wcs), output_wcs, (1, n),
                                  center_jacobian=False,
                                  return_footprint=False,
                                  roundtrip_coords=False,
                                  kernel_width=2.5,
                                  sample_region_width=8,
                                  bad_value_mode='ignore',
                                  )
    slice = slice[0]
    # Turn infs into nans
    np.nan_to_num(slice, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    fixed_angles = output_wcs.last_fixed_angles[1, 1:-1]
    hpc = output_wcs.last_hpc[1, 1:-1]
    return (slice, fixed_angles, hpc,
            output_wcs.last_venus_elongation, output_wcs.last_venus_angle)


def extract_slices(bundle: "InputDataBundle", n, title="Orbital plane slices"):
    slices = []
    fixed_angles = []
    venus_elongations = []
    venus_angles = []
    hpcs = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*failed to converge.*')
        warnings.filterwarnings('ignore', message='.*All-NaN slice.*')
        for slice, fixed_angle, hpc, ve, va in process_map(
                    _extract_slice,
                    bundle.images,
                    bundle.wcses,
                    repeat(n),
                    repeat(bundle.is_inner),
                    chunksize=5):
            slices.append(slice)
            fixed_angles.append(fixed_angle)
            hpcs.append(hpc)
            venus_elongations.append(ve)
            venus_angles.append(va)

    slices = np.stack(slices)
    fixed_angles = np.stack(fixed_angles)
    venus_elongations = np.stack(venus_elongations)
    venus_angles = np.stack(venus_angles)

    transformer = OrbitalSliceWCS(
        bundle.wcses[-1], n, bundle.is_inner)

    return PlainJMap(
        slices=slices,
        angles=transformer.pix_to_elongation(np.arange(n)),
        fixed_angles=fixed_angles,
        times=bundle.times,
        transformer=transformer,
        title_stub=title,
        venus_elongations=venus_elongations,
        venus_angles=venus_angles,
        hpcs=hpcs,
        is_inner=bundle.is_inner,
        quantity=bundle.quantity,
        encounter=bundle.encounter,
    )


class OrbitalSliceWCS(utils.FakeWCS):
    orbital_north = None
    
    @classmethod
    def _init_orbital_north(cls):
        # Structuring things this way facilitates testing---the SPICE functions
        # in `planets` can be mocked, the orbital plane can be loaded, and then
        # the functions can be unmocked so the Venus calculations can run
        # normally.
        orbital_plane = planets.get_orbital_plane('psp', '2020-01-01 12:12:12')
        orbital_north = orbital_plane.data[0].cross(orbital_plane.data[20])
        cls.orbital_north = SkyCoord(orbital_north,
                                     representation_type='cartesian',
                                     frame=HeliocentricInertial)

    def __init__(self, input_wcs, n, is_inner, date_override=None):
        super().__init__(input_wcs)
        if is_inner == 'both':
            self.angle_start = 13.5
            self.angle_stop = 103.75
        elif is_inner:
            self.angle_start = 13.5
            self.angle_stop = 51.25
        else:
            self.angle_start = 49
            self.angle_stop = 103.75
        self.n = n
        self.last_hpc = None
        if date_override:
            self.date = date_override
        else:
            try:
                self.date = input_wcs.wcs.dateavg.replace('T', ' ')
            except AttributeError:
                self.date = ''
        if self.date == '':
            self.date = '2020-01-01 12:12:12'
        if self.orbital_north is None:
            self._init_orbital_north()
        self.orbital_frame = NorthOffsetFrame(north=self.orbital_north,
                                              obstime=self.date)

    def pix_to_elongation(self, pixel_num):
        return (pixel_num * (self.angle_stop - self.angle_start)
                / (self.n-1) + self.angle_start)
    
    def pixel_to_world_values(self, *pixel_arrays):
        sc = SkyCoord(self.input_wcs.wcs.aux.hgln_obs * u.deg,
                      self.input_wcs.wcs.aux.hglt_obs * u.deg,
                      self.input_wcs.wcs.aux.dsun_obs * u.m,
                      frame='heliographic_stonyhurst', obstime=self.date)
        sc = sc.transform_to(self.orbital_frame)
        sc_cart = sc.represent_as('cartesian')
        
        angle_to_sun = np.arctan2(-sc_cart.y, -sc_cart.x)
        r = sc.distance

        elongations = self.pix_to_elongation(pixel_arrays[0]) * u.deg
        
        angles = angle_to_sun - elongations

        x = np.cos(angles) * r + sc_cart.x
        y = np.sin(angles) * r + sc_cart.y
        points = SkyCoord(x, y, 0,
                          frame=self.orbital_frame,
                          representation_type='cartesian',
                          observer=sc, unit='m')
        points = points.transform_to(Helioprojective)
        self.last_hpc = points
        Tx = points.Tx.to(u.deg).value
        Ty = points.Ty.to(u.deg).value

        # Negative here just to choose a frame where the fixed-frame angle
        # increases in the same direction as elongation
        self.last_fixed_angles = -angles.to(u.deg).value
        self.last_fixed_angles %= 360
        self.last_fixed_angles[
            self.last_fixed_angles < FIXED_ANGLE_TARGET_RANGE[0]] += 360

        x, y = self.input_wcs.world_to_pixel(points)

        g = ((x > 0) * (x < self.input_wcs.pixel_shape[0]-1)
             * (y > 0) * (y < self.input_wcs.pixel_shape[1]-1))
        Tx[~g] = np.nan
        Ty[~g] = np.nan
        
        venus_pos = planets.locate_planets(
            self.date, only=['venus'], sc_pos=sc)[0]
        venus_pos = self.input_wcs.world_to_pixel(venus_pos)
        if not (0 <= venus_pos[0] < self.input_wcs.pixel_shape[0]
                and 0 <= venus_pos[1] < self.input_wcs.pixel_shape[1]):
            venus_pos = np.nan, np.nan
        self.last_venus_elongation = np.interp(
            venus_pos[0], x[1], elongations[1].value,
            left=np.nan, right=np.nan)
        self.last_venus_angle = np.interp(
            venus_pos[0], x[1], self.last_fixed_angles[1],
            left=np.nan, right=np.nan)
        
        return Tx, Ty


def load_files(files):
    wcses = []
    images = None
    
    with utils.ignore_fits_warnings():
        for i, f in enumerate(tqdm(files)):
            with fits.open(f) as hdul:
                if images is None:
                    images = np.empty((len(files), *hdul[0].data.shape),
                                      dtype='f4')
                images[i] = hdul[0].data
                header = hdul[0].header
                wcses.append(WCS(header, hdul))
    
    times = np.array(utils.to_timestamp(files, read_headers=True))
    
    E = utils.extract_encounter_number(files[0], as_int=True)
    is_inner = header['detector'] == 1
    return InputDataBundle(images=images,
                           wcses=wcses,
                           times=times,
                           level=header['level'],
                           is_inner=is_inner,
                           quantity='flux',
                           encounter=E)


@dataclasses.dataclass
class InputDataBundle:
    images: np.ndarray
    wcses: list[WCS]
    times: np.ndarray
    is_inner: bool
    level: str
    quantity: str
    encounter: int


@dataclasses.dataclass
class BaseJmap:
    slices: np.ndarray
    angles: np.ndarray
    fixed_angles: np.ndarray
    times: np.ndarray
    transformer: OrbitalSliceWCS
    venus_elongations: np.ndarray
    venus_angles: np.ndarray
    is_inner: bool
    quantity: str
    encounter: int
    title_stub: str = 'Orbital-plane slices'

    _title: list[str] = None
    _subtitles: list[list[str]] = dataclasses.field(default_factory=list)

    @property
    def title(self) -> str:
        title = ", ".join(self._title)
        for subtitle in self._subtitles:
            title += '\n'
            title += ', '.join(subtitle)
        return title

    def __post_init__(self):
        self._title = [self.title_stub]

    def make_squarish(self):
        """
        Get closer to a 1:1 aspect ratio (e.g. so unsharp-masking is more
        symmetric)
        """
        if self.slices.shape[0] > self.slices.shape[1]:
            # More rows than columns
            n = int(round(self.slices.shape[0] / self.slices.shape[1]))
            # We'll super-sample each row
            idx = np.linspace(
                0, self.slices.shape[1]-1, n*self.slices.shape[1] - 1)
            new_slices = np.empty((self.slices.shape[0], len(idx)))
            for i in range(self.slices.shape[0]):
                # Interpolate within each row
                new_slices[i] = np.interp(idx, np.arange(
                    self.slices.shape[1]), self.slices[i])
            self.slices = new_slices
            self.angles = np.interp(idx, np.arange(
                len(self.angles)), self.angles)
        else:
            # More columns than rows
            n = int(round(self.slices.shape[1] / self.slices.shape[0]))
            # We'll super-sample each column
            idx = np.linspace(
                0, self.slices.shape[0]-1, n*self.slices.shape[0] - 1)
            new_slices = np.empty((len(idx), self.slices.shape[1]))
            for i in range(self.slices.shape[1]):
                # Interpolate within each column
                new_slices[:, i] = np.interp(idx, np.arange(
                    self.slices.shape[0]), self.slices[:, i])
            self.slices = new_slices
            self.times = np.interp(idx, np.arange(len(self.times)), self.times)
        self._title.append("squarish")

    def trim_nans(self):
        # Trim off all-nan rows
        while np.all(np.isnan(self.slices[0])):
            self.slices = self.slices[1:]
            self.times = self.times[1:]
        while np.all(np.isnan(self.slices[-1])):
            self.slices = self.slices[:-1]
            self.times = self.times[:-1]

    def unsharp_mask(self, radius, amount):
        self._title.append(f"unsharp({radius}, {amount})")
        self.slices = nan_unsharp_mask(
            self.slices, radius=radius, amount=amount)

    def minsmooth(self, radius, percentile):
        self._title.append(f"minsmooth({radius}px, {percentile})")
        self.slices = nan_minsmooth(
            self.slices, radius=radius, percentile=percentile)

    def remove_gaussian_blurred(self, radius):
        self._title.append(f"sub_gaussian({radius}px)")
        self.slices -= nan_gaussian_blur(self.slices, radius)

    def per_row_detrend(self, order):
        for i in range(self.slices.shape[0]):
            y = self.slices[i]
            g = np.isfinite(y)
            y = y[g]
            if y.size == 0:
                continue
            x = np.arange(len(y))
            fit = np.polyfit(x, y, order)
            yf = np.polyval(fit, x)
            self.slices[i][g] -= yf
        self._title.append(f"{order}th-order row detrending")

    def per_col_detrend(self, order):
        for i in range(self.slices.shape[1]):
            y = self.slices[:, i]
            g = np.isfinite(y)
            y = y[g]
            if y.size == 0:
                continue
            x = np.arange(len(y))
            fit = np.polyfit(x, y, order)
            yf = np.polyval(fit, x)
            self.slices[:, i][g] -= yf
        self._title.append(f"{order}th-order col detrending")
    
    def local_col_detrend(self, order=1, window=101):
        if window % 2 != 1:
            window += 1
        for j in range(self.slices.shape[1]):
            data = self.slices[:, j].copy()
            indices = np.arange(data.size, dtype=float)
            fitted = utils.time_window_savgol_filter(
                indices, data, window, order)
            self.slices[:, j] -= fitted
        self._title.append(f"local col detrend({order}, {window}px)")

    def per_row_normalize(self):
        # Do a per-row normalization
        lows, highs = np.nanpercentile(self.slices, [1, 99], axis=1)
        lows = lows[:, None]
        highs = highs[:, None]
        with np.errstate(invalid='ignore'):
            self.slices = (self.slices - lows) / (highs - lows)
        self._title.append("row-normalized")

    def _resample_time_post_hook(self):
        pass
    
    def resample_time(self, new_dt, t_start=None, t_end=None):
        if t_start is None:
            t_start = self.times[0]
        if t_end is None:
            t_end = self.times[-1]
        new_t = np.arange(t_start, t_end+.00001, new_dt)
        
        wcs = ResampleTimeWCS(self.times, new_dt)
        
        self.slices = reproject.reproject_adaptive(
            (self.slices, wcs),
            wcs, (len(new_t), self.slices.shape[-1]),
            boundary_mode='ignore',
            bad_value_mode='ignore',
            center_jacobian=False,
            roundtrip_coords=False,
            return_footprint=False)
        
        self.fixed_angles = reproject.reproject_adaptive(
            (self.fixed_angles, wcs),
            wcs, (len(new_t), self.fixed_angles.shape[-1]),
            boundary_mode='ignore',
            bad_value_mode='ignore',
            center_jacobian=False,
            roundtrip_coords=False,
            return_footprint=False)

        # Clear out little extra bits that show up around imager gaps
        nans = np.isnan(self.slices)
        nans = scipy.ndimage.binary_closing(nans, np.ones((3, 3)))
        self.slices[nans] = np.nan
        
        # Resample the Venus locations
        self.venus_elongations = reproject.reproject_adaptive(
            (self.venus_elongations.reshape((-1, 1)), wcs),
            wcs, (len(new_t), 1),
            boundary_mode='ignore',
            bad_value_mode='ignore',
            center_jacobian=False,
            roundtrip_coords=False,
            return_footprint=False)
        self.venus_elongations = self.venus_elongations[:, 0]
        
        self.venus_angles = reproject.reproject_adaptive(
            (self.venus_angles.reshape((-1, 1)), wcs),
            wcs, (len(new_t), 1),
            boundary_mode='ignore',
            bad_value_mode='ignore',
            center_jacobian=False,
            roundtrip_coords=False,
            return_footprint=False)
        self.venus_angles = self.venus_angles[:, 0]
        
        self.times = new_t
        self._title.append(f"resampled dt={new_dt}")

    def clamp(self, min=-np.inf, max=np.inf):
        self.slices[self.slices < min] = min
        self.slices[self.slices > max] = max
        self._title.append(f"clamp({min}, {max})")
    
    def pclamp(self, pmin=0, pmax=100):
        min, max = np.nanpercentile(self.slices, (pmin, pmax))
        self.slices[self.slices < min] = min
        self.slices[self.slices > max] = max
        self._title.append(f"pclamp({pmin}, {pmax})")

    def percentile_normalize(self, pmin, pmax):
        vmin, vmax = np.nanpercentile(self.slices, [pmin, pmax])
        self.slices = (self.slices - vmin) / (vmax - vmin)
        self._title.append(f"normalized to ({pmin}, {pmax}) percentile")

    def median_filter(self, size):
        self._title.append(f"med_filt({size})")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*All-NaN slice.*')
            self.slices = scipy.ndimage.generic_filter(
                self.slices, np.nanmedian, size)

    def gaussian_filter(self, size, nan_aware=False):
        self._title.append(f"gauss_filt({size})")
        if nan_aware:
            self.slices = nan_gaussian_blur(self.slices, size)
        else:
            self.slices = scipy.ndimage.gaussian_filter(
                self.slices, size, mode='nearest')

    def radial_detrend(self):
        def radial_fcn(x, A, exp, B, C, D, switch):
            toggle = 1 / (1 + np.exp(-x+switch))
            return (toggle * A * x ** exp
                    + (1-toggle) * (B * x ** 2 + C * x + D))
        for i in range(self.slices.shape[0]):
            y = self.slices[i]
            g = np.isfinite(y)
            y = y[g]
            if y.size == 0:
                continue
            x = np.arange(len(y)) + 1
            y *= 1e13
            try:
                popt, _ = scipy.optimize.curve_fit(
                    radial_fcn,
                    x, y,
                    p0=[1, -1, -1, 1, 1, 10],
                    maxfev=30000,
                    bounds=([-np.inf, -6, -np.inf, -np.inf, -np.inf, 0],
                            [np.inf, 0, np.inf, np.inf, np.inf, len(y)]))
            except RuntimeError:
                # Use the last row's fit
                pass
            yf = 1e-13 * radial_fcn(x, *popt)
            self.slices[i][g] -= yf
        self._title.append("radial detrending")
    
    def bg_remove(self, med_size=15, gauss_size=51, nan_aware=False):
        bg = self.deepcopy()
        bg.median_filter(med_size)
        bg.gaussian_filter(gauss_size, nan_aware=nan_aware)
        self.slices -= bg.slices
        self._title.append(f"bg_rem({med_size}, {gauss_size})")
    
    def mask_venus(self, angular_width):
        self._title.append("Venus masked")
        for i in range(self.slices.shape[0]):
            venus_angle = self._get_venus_angles()[i]
            mask_start = venus_angle - angular_width / 2
            mask_stop = venus_angle + angular_width / 2
            xstart, xstop = np.interp(
                (mask_start, mask_stop),
                self.angles,
                np.arange(len(self.angles)))
            if np.isnan(xstart) or np.isnan(xstop):
                continue
            xstart, xstop = int(xstart), int(xstop)
            self.slices[i, xstart:xstop] = np.nan

    def deepcopy(self) -> "BaseJmap":
        return copy.deepcopy(self)

    def plot(self,
             bundle: "InputDataBundle"=None,
             ax=None, label_vr=False, vmin=None, vmax=None,
             pmin=5, pmax=95, gamma=None, interactive=False,
             cmap=None, rel_dates=False, show_full_array=False):
        min, max = np.nanpercentile(self.slices, [pmin, pmax])
        if vmin is None:
            if self.quantity == 'distance':
                # Sensible, also workaround mpl bug #25239
                vmin = 0
            else:
                vmin = min
        if vmax is None:
            vmax = max

        if interactive:
            if bundle is None:
                raise ValueError(
                    "Bundle must be provided for interactive mode")
            if ax is not None:
                raise ValueError("Cannot accept `ax` for interactive mode")
            plt.figure(figsize=(15, 7.5))
            ax = plt.subplot(121)
        elif ax is None:
            ax = plt.gca()
        
        angles = self.angles
        image = self.slices
        if not show_full_array:
            # Trim all-nan angular positions
            while np.all(np.isnan(image[:, 0])):
                image = image[:, 1:]
                angles = angles[1:]
            while np.all(np.isnan(image[:, -1])):
                image = image[:, :-1]
                angles = angles[:-1]
        
        if cmap is None:
            if self.quantity == 'flux':
                cmap = copy.deepcopy(plot_utils.wispr_cmap)
            elif self.quantity == 'distance':
                cmap = copy.deepcopy(plt.get_cmap('viridis_r'))
            else:
                raise ValueError("Invalid quantity")
        else:
            cmap = copy.deepcopy(plt.get_cmap(cmap))
        cmap.set_bad("#4d4540")
        if gamma is None:
            if self.quantity == 'flux':
                gamma = 1/2.2
            elif self.quantity == 'distance':
                gamma = 1
            else:
                raise ValueError("Invalid quantity")
        
        image = image.T
        if rel_dates:
            perihelion = planets.perihelia[self.encounter]
            perihelion = utils.to_timestamp(perihelion.replace(' ', 'T'))
            x = self.times - perihelion
            x /= 3600 * 24
            ax.set_xlabel("Days after perihelion")
        else:
            dates = plot_utils.x_axis_dates(self.times, ax=ax)
            x = dates
        y = angles
        ax.set_ylabel(self.ylabel)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.StrMethodFormatter("{x:.0f}°"))
        im = ax.pcolormesh(x, y, image,
                           cmap=cmap,
                           norm=matplotlib.colors.PowerNorm(gamma=gamma,
                                                            vmin=vmin,
                                                            vmax=vmax),
                           shading='nearest')
        try:
            plt.sci(im)
        except ValueError:
            # May occur if we're plotting onto subplots
            pass

        ax.set_title(self.title)

        if label_vr:
            if bundle is None:
                raise ValueError("Bundle must be provided")
            r = [w.wcs.aux.dsun_obs for w in bundle.wcses]
            vrs = np.gradient(r, self.times)
            vrs /= 1000

            def t2vr(t): return np.interp(t, dates, vrs)
            def vr2t(vr): return np.interp(vr, vrs, dates)
            
            ax2 = plt.gca().secondary_xaxis(
                'top', functions=(t2vr, vr2t))
            ax2.set_xlabel("S/C radial velocity (km/s)")
        
        if interactive:
            if rel_dates:
                raise ValueError(
                    "Interactive mode does not support relative dates")
            if isinstance(bundle, InputDataBundle):
                b = bundle
            else:
                b = bundle[0]
            axi = plt.subplot(122, projection=b.wcses[0])
            plot_utils.plot_WISPR(b.images[0], ax=axi, level_preset=b.level)
            plot_utils.setup_WCS_axes(axi)
            class EventHandler():
                self.last_t = None
                
                def __init__(self, map):
                    self.axmarker, = ax.plot(
                        [None], [None], marker='x', color='C1')
                    self.map = map
                
                def onclick(self, event):
                    try:
                        if event.inaxes is not ax:
                            return
                        t, angle = event.xdata, event.ydata
                        if event.button == 3:
                            # Right click
                            if self.last_t is None:
                                return
                            t = self.last_t
                            event.xdata = self.axmarker.get_data()[0][0]
                        else:
                            self.last_t = t
                        t = np.interp(t, dates, self.map.times)
                        tindex, hpc, hpcs, bb = self.map._get_hpc(
                            t, angle, bundle)
                        if hpc is None:
                            return
                        nonlocal axi
                        axi.remove()
                        wcs = bb.wcses[tindex]
                        axi = plt.subplot(122, projection=wcs)
                        plot_utils.setup_WCS_axes(axi)
                        min, max = np.nanpercentile(bb.images[tindex], [pmin, pmax])
                        axi.imshow(bb.images[tindex], cmap=cmap,
                                norm=matplotlib.colors.PowerNorm(gamma=gamma,
                                                                    vmin=min,
                                                                    vmax=max),
                                origin='lower')
                        x, y = wcs.world_to_pixel(hpc)
                        axi.scatter(x, y-10, marker=3, color='C1')
                        axi.scatter(x, y+10, marker=2, color='C1')
                        x, y = wcs.world_to_pixel(hpcs)
                        axi.plot(x, y-10, color='C0', ls='--')
                        axi.plot(x, y+10, color='C0', ls='--')
                        axi.set_xlim(-.5, bb.images[tindex].shape[1]-.5)
                        axi.set_ylim(-.5, bb.images[tindex].shape[0]-.5)
                        
                        self.axmarker.set_data([event.xdata], [event.ydata])
                    except Exception as e:
                        ax.set_title(e)
            self.handler = EventHandler(self)
            plt.connect('button_press_event', self.handler.onclick)


class ResampleTimeWCS(utils.FakeWCS):
    def __init__(self, times, new_dt):
        super().__init__(None)
        self.times = times
        self.new_dt = new_dt
        
    def pixel_to_world_values(self, *pixel_arrays):
        x, y = pixel_arrays[0], pixel_arrays[1]
        yt = y * self.new_dt + self.times[0]
        yn = np.interp(yt, self.times, np.arange(len(self.times)))
        # If an imager drops out for a while, we'll see a bunch of output
        # timesteps mapping to pretty much the same exact region (the space
        # between two difference slices). For better plotting, let's let
        # the dropout show as a gap and not a big interpolated streak.
        delta = np.gradient(yn, axis=0)
        yn[delta < .2] = np.nan
        return x.copy(), yn
    
    def world_to_pixel_values(self, *world_arrays):
        return world_arrays


class DerotatedFixedAngleWCS(utils.FakeWCS):
    def __init__(self, angle_start, angle_stop, n):
        super().__init__(None)
        self.angle_start = angle_start
        self.angle_stop = angle_stop
        self.n = n
        
    def pixel_to_world_values(self, *pixel_arrays):
        # Convert derotated-Jmap pixel coordinates to fixed-angle coordinates,
        # which is easy and linear
        out_angles = (
            pixel_arrays[0] * (self.angle_stop - self.angle_start)
            / (self.n-1) + self.angle_start)
        return out_angles, pixel_arrays[1].copy()
    
    def world_to_pixel_values(self, *world_arrays):
        # Convert fixed-angle coordinates to derotated-Jmap pixel coordinates,
        # which is easy and linear
        out_pixels = ((world_arrays[0] - self.angle_start)
                      * (self.n-1) / (self.angle_stop - self.angle_start))
        return out_pixels, world_arrays[1].copy()
    

class RotatedFixedAngleWCS(utils.FakeWCS):
    def __init__(self, fixed_angles, angle_start):
        super().__init__(None)
        self.fixed_angles = fixed_angles
        self.angle_start = angle_start
        
    def pixel_to_world_values(self, *pixel_arrays):
        # Convert rotated-Jmap pixel coordinates to fixed-angle
        # coordinates---we know what the FA value is for each pixel center, so
        # we just interpolate our target pixel coordinates against that.
        out = np.interp(
            pixel_arrays[0],
            np.arange(len(self.fixed_angles)),
            self.fixed_angles,
            left=np.nan, right=np.nan)
        return out, pixel_arrays[1].copy()
    
    def world_to_pixel_values(self, *world_arrays):
        # Convert fixed-angle coordinates to rotated-Jmap pixel
        # coordinates---we know what the FA value is for each pixel center, so
        # we just interpolate our target FA coordinates against that.
        out = np.interp(
            world_arrays[0],
            self.fixed_angles,
            np.arange(len(self.fixed_angles)),
            left=np.nan, right=np.nan)
        return out, world_arrays[1].copy()


class PlainJMap(BaseJmap):
    ylabel = "Elongation"

    def __init__(self, *, hpcs, **kwargs):
        super().__init__(**kwargs)
        self.hpcs = hpcs

    def derotate(self, n) -> "DerotatedJMap":
        angle_start, angle_stop = FIXED_ANGLE_TARGET_RANGE
        output = np.empty((self.slices.shape[0], n))
        derotated_wcs = DerotatedFixedAngleWCS(angle_start, angle_stop, n)
        for i, (slice, fixed_angles) in enumerate(
                zip(self.slices, self.fixed_angles)):
            rotated_wcs = RotatedFixedAngleWCS(fixed_angles, angle_start)
            output[i] = reproject.reproject_adaptive(
                (slice.reshape((1, slice.size)), rotated_wcs),
                derotated_wcs, (1, n),
                boundary_mode='ignore',
                bad_value_mode='ignore',
                center_jacobian=False,
                return_footprint=False,
                roundtrip_coords=False)
        outmap = DerotatedJMap(
            slices=output,
            angles=np.linspace(angle_start, angle_stop, n),
            fixed_angles=self.fixed_angles.copy(),
            source_jmap=self,
            times=copy.deepcopy(self.times),
            transformer=copy.deepcopy(self.transformer),
            title_stub=self.title_stub,
            venus_elongations=copy.deepcopy(self.venus_elongations),
            venus_angles=copy.deepcopy(self.venus_angles),
            is_inner=self.is_inner,
            quantity=self.quantity,
            encounter=self.encounter,
            )
        outmap._title = copy.deepcopy(self._title)
        outmap._title.append("derotated")
        outmap._subtitles = copy.deepcopy(self._subtitles)
        return outmap
    
    def _get_venus_angles(self):
        return self.venus_elongations
    
    def _get_hpc(self, t, angle, bundle, angle_is_fixed=False):
        tindex = int(np.round(np.argmin(np.abs(self.times - t))))
        if angle_is_fixed:
            angles = self.fixed_angles[tindex]
        else:
            angles = self.angles
        for offset in [0, -360, -720]:
            if angles[0] <= angle + offset <= angles[-1]:
                angle += offset
                break
        else:
            return None, None, None, None
        aindex = int(np.round(np.argmin(np.abs(angles - angle))))
        return tindex, self.hpcs[tindex][aindex], self.hpcs[tindex], bundle
    
    def _resample_time_post_hook(self):
        # These aren't valid anymore
        self.hpcs = None


class DerotatedJMap(BaseJmap):
    ylabel = "Fixed-frame angular position"
    
    def __init__(self, *, source_jmap: "PlainJMap", **kwargs):
        super().__init__(**kwargs)
        self.source_jmap = source_jmap
        self.other_map = None
    
    def _get_venus_angles(self):
        return self.venus_angles
    
    def _get_hpc(self, t, fixed_angle, bundle):
        return self.source_jmap._get_hpc(
            t, fixed_angle, bundle, angle_is_fixed=True)
    
    def merge(self, other):
        if not np.all(self.times == other.times):
            raise ValueError("Time grids must match")
        if not np.all(self.angles == other.angles):
            raise ValueError("Angle grids must match")
        
        slices = np.where(np.isfinite(self.slices),
                          self.slices, other.slices)
        if self.is_inner:
            source_jmaps = [self, other]
        else:
            source_jmaps = [other, self]
        outmap = MergedDerotatedJMap(
            slices=slices,
            source_jmaps=source_jmaps,
            angles=copy.deepcopy(self.angles),
            fixed_angles=self.fixed_angles.copy(),
            times=copy.deepcopy(self.times),
            transformer=copy.deepcopy(self.transformer),
            title_stub=self.title_stub,
            venus_elongations=copy.deepcopy(self.venus_elongations),
            venus_angles=copy.deepcopy(self.venus_angles),
            quantity=self.quantity,
            encounter=self.encounter,
            )
        outmap._subtitles.append(self._title)
        outmap._subtitles.append(other._title)
        outmap._title = ['Merged']
        return outmap
    
    def rotate(self) -> "PlainJMap":
        angle_start = self.angles[0]
        angle_stop = self.angles[-1]
        n = len(self.angles)
        output = np.empty(
            (self.slices.shape[0], self.source_jmap.slices.shape[1]))
        derotated_wcs = DerotatedFixedAngleWCS(angle_start, angle_stop, n)
        for i, (slice, fixed_angles) in enumerate(
                zip(self.slices, self.fixed_angles)):
            rotated_wcs = RotatedFixedAngleWCS(fixed_angles, angle_start)
            output[i] = reproject.reproject_adaptive(
                (slice.reshape((1, slice.size)), derotated_wcs),
                rotated_wcs, (1, len(fixed_angles)),
                boundary_mode='ignore',
                bad_value_mode='ignore',
                center_jacobian=False,
                return_footprint=False,
                roundtrip_coords=False)
        outmap = PlainJMap(
            slices=output,
            angles=self.source_jmap.angles,
            fixed_angles=self.fixed_angles.copy(),
            times=copy.deepcopy(self.times),
            transformer=copy.deepcopy(self.transformer),
            title_stub=self.title_stub,
            venus_elongations=copy.deepcopy(self.venus_elongations),
            venus_angles=copy.deepcopy(self.venus_angles),
            is_inner=self.is_inner,
            quantity=self.quantity,
            hpcs=None,
            encounter=self.encounter,
            )
        outmap._title = copy.deepcopy(self._title)
        outmap._title.append("Re-rotated")
        outmap._subtitles = copy.deepcopy(self._subtitles)
        return outmap


class MergedDerotatedJMap(DerotatedJMap):
    def __init__(self, *, source_jmaps, **kwargs):
        kwargs['is_inner'] = None
        super().__init__(**kwargs, source_jmap=None)
        self.source_jmaps = source_jmaps    
    
    def _get_hpc(self, t, fixed_angle, bundles):
        for jmap, bundle in zip(self.source_jmaps, bundles):
            ret = jmap._get_hpc(t, fixed_angle, bundle)
            if ret[0] is not None:
                break
        return ret


@numba.njit(parallel=True, cache=True)
def nan_gaussian_blur(data, radius):
    blurred = np.zeros(data.shape)
    for yo in numba.prange(blurred.shape[0]):
        for xo in range(blurred.shape[1]):
            weight_sum = 0
            for dy in range(-4*radius, 4*radius + 1):
                if yo + dy < 0 or yo + dy >= data.shape[0]:
                    continue
                for dx in range(-4*radius, 4*radius + 1):
                    if xo + dx < 0 or xo + dx >= data.shape[1]:
                        continue
                    if np.isnan(data[yo+dy, xo+dx]):
                        continue
                    weight = np.exp(-dx**2 / 2 / radius ** 2
                                    - dy**2 / 2 / radius**2)
                    weight_sum += weight
                    blurred[yo, xo] += weight * data[yo+dy, xo+dx]
            if weight_sum > 0:
                blurred[yo, xo] /= weight_sum
    return blurred


def nan_unsharp_mask(data, radius, amount):
    blurred = nan_gaussian_blur(data, radius)
    return data + amount * (data - blurred)


@numba.njit(parallel=True, cache=True)
def nan_minsmooth(data, radius, percentile):
    output = data.copy()
    for yo in numba.prange(output.shape[0]):
        for xo in range(output.shape[1]):
            ystart = max(0, yo - radius)
            ystop = min(yo + radius + 1, data.shape[0])
            xstart = max(0, xo - radius)
            xstop = min(xo + radius + 1, data.shape[1])
            output[yo, xo] -= np.nanpercentile(
                data[ystart:ystop, xstart:xstop],
                percentile)
    return output
