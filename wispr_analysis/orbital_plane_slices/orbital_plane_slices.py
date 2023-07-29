import copy
import dataclasses
from itertools import repeat
import warnings

from astropy.coordinates import SkyCoord, angular_separation
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.colors
import matplotlib.pyplot as plt
import numba
import numpy as np
import reproject
import scipy
from sunpy.coordinates import (
    HeliocentricInertial, Helioprojective, NorthOffsetFrame)
from tqdm.contrib.concurrent import process_map

from .. import planets, plot_utils


def _extract_slice(image, sc_pos, wcs, angle_start, angle_stop, n):
    transformer = OrbitalSliceTransformer(
        sc_pos, wcs, angle_start, angle_stop, n)
    slice = np.zeros((1, 1, n))
    reproject.adaptive.deforest.map_coordinates(
        image.astype(float).reshape((1, *image.shape)),
        slice,
        transformer,
        out_of_range_nan=True,
        center_jacobian=False)
    slice = slice[0, 0]
    angle = transformer.pix_to_angle(np.arange(slice.size))
    Txs, Tys = transformer.last_hpc.Tx, transformer.last_hpc.Ty
    # Note: at this point the slices could be stacked into a de-rotated J-map,
    # but we first want a plain J map
    while np.isnan(slice[0]):
        slice = slice[1:]
        angle = angle[1:]
        Txs = Txs[1:]
        Tys = Tys[1:]
    while np.isnan(slice[-1]):
        slice = slice[:-1]
        angle = angle[:-1]
        Txs = Txs[:-1]
        Tys = Tys[:-1]
    # Turn infs into nans
    np.nan_to_num(slice, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    return slice, angle, Txs, Tys


def extract_slices(
        bundle: "InputDataBundle", angle_start, angle_stop, n,
        title="Orbital plane slices"):
    slices = []
    angles = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*failed to converge.*')
        warnings.filterwarnings('ignore', message='.*All-NaN slice.*')
        for slice, angle, Txs, Tys in process_map(_extract_slice,
                                                  bundle.images,
                                                  bundle.sc_poses,
                                                  bundle.wcses,
                                                  repeat(angle_start),
                                                  repeat(angle_stop),
                                                  repeat(n),
                                                  chunksize=5):
            slices.append(slice)
            angles.append(angle)

    min_len = min(len(s) for s in slices)
    slices = [s[:min_len] for s in slices]
    angles = [a[:min_len] for a in angles]
    Txs = Txs[:min_len]
    Tys = Tys[:min_len]
    slices = np.stack(slices)
    angles = np.stack(angles)

    elongations = angular_separation(
        0*u.deg, 0*u.deg, Txs, Tys).to(u.deg).value

    transformer = OrbitalSliceTransformer(
        bundle.sc_poses[-1], bundle.wcses[-1], angle_start, angle_stop, n)

    return PlainJMap(
        slices=slices,
        target_angles=angles,
        angles=elongations,
        times=bundle.times,
        transformer=transformer,
        normalized=False,
        title_stub=title,
    )


class OrbitalSliceTransformer:
    def __init__(self, sc_pos, img_wcs, angle_start, angle_stop, n,
                 date_override=None):
        self.sc_pos = sc_pos
        self.img_wcs = img_wcs
        self.angle_start = angle_start
        self.angle_stop = angle_stop
        self.n = n
        self.last_hpc = None

        if date_override:
            self.date = date_override
        else:
            try:
                self.date = img_wcs.wcs.dateavg.replace('T', ' ')
            except AttributeError:
                self.date = ''
        if self.date == '':
            self.date = '2020-01-01 12:12:12'
        orbital_plane = planets.get_orbital_plane('psp', '2020-01-01 12:12:12')
        p1 = orbital_plane.data[0]
        p2 = orbital_plane.data[20]
        orbital_north = p1.cross(p2)
        orbital_north = SkyCoord(orbital_north,
                                 representation_type='cartesian',
                                 frame=HeliocentricInertial, obstime=self.date)
        self.orbital_frame = NorthOffsetFrame(
            north=orbital_north, obstime=self.date)

    def pix_to_angle(self, pixel_num):
        return (pixel_num * (self.angle_stop - self.angle_start)
                / (self.n-1) + self.angle_start)

    def __call__(self, pixel_out):
        sc = SkyCoord(*(self.sc_pos * u.R_sun.to(u.m)),
                      representation_type='cartesian',
                      frame=HeliocentricInertial, obstime=self.date, unit='m')
        sc = sc.transform_to(self.orbital_frame)
        sc_cart = sc.represent_as('cartesian')

        angles = -self.pix_to_angle(pixel_out[..., 0]) * np.pi/180

        dx = np.cos(angles) * u.R_sun.to(u.m)
        dy = np.sin(angles) * u.R_sun.to(u.m)
        points = SkyCoord(sc_cart.x.value + dx,
                          sc_cart.y.value + dy,
                          sc_cart.z.value,
                          frame=self.orbital_frame,
                          representation_type='cartesian',
                          observer=sc, unit='m')
        points = points.transform_to(Helioprojective)
        self.last_hpc = points[1]

        x, y = self.img_wcs.world_to_pixel(points)

        g = ((x > 0) * (x < self.img_wcs.pixel_shape[0]-1)
             * (y > 0) * (y < self.img_wcs.pixel_shape[1]-1))
        x[~g] = np.nan
        y[~g] = np.nan

        pixel_in = np.empty_like(pixel_out)
        pixel_in[..., 0] = x
        pixel_in[..., 1] = y
        return pixel_in


@dataclasses.dataclass
class InputDataBundle:
    images: np.ndarray
    wcses: list[WCS]
    sc_poses: list[tuple]
    times: np.ndarray


@dataclasses.dataclass
class BaseJmap:
    slices: np.ndarray
    angles: np.ndarray
    times: np.ndarray
    transformer: OrbitalSliceTransformer
    normalized: bool
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
        # Trim off all-nan rows from the top
        while np.all(np.isnan(self.slices[-1])):
            self.slices = self.slices[:-1]
            self.times = self.times[:-1]

        # Trim off columns that are all-nan in the top half of the image
        h = self.slices.shape[0]//2
        while np.all(np.isnan(self.slices[h:, -1])):
            self.slices = self.slices[:, :-1]
            self.radii = self.radii[:-1]

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

    def per_row_normalize(self):
        # Do a per-row normalization
        lows, highs = np.nanpercentile(self.slices, [1, 99], axis=1)
        lows = lows[:, None]
        highs = highs[:, None]
        with np.errstate(invalid='ignore'):
            self.slices = (self.slices - lows) / (highs - lows)
        self._title.append("row-normalized")

    def resample_time(self, new_dt, t_start=None, t_end=None):
        if t_start is None:
            t_start = self.times[0]
        if t_end is None:
            t_end = self.times[-1]
        new_t = np.arange(t_start, t_end+.00001, new_dt)

        def transformer(pixel_out):
            x, y = pixel_out[..., 0], pixel_out[..., 1]
            yt = y * new_dt + self.times[0]
            yn = np.interp(yt, self.times, np.arange(len(self.times)))
            # If an imager drops out for a while, we'll see a bunch of output
            # timesteps mapping to pretty much the same exact region (the space
            # between two difference slices). For better plotting, let's let
            # the dropout show as a gap and not a big interpolated streak.
            delta = np.gradient(yn, axis=0)
            yn[delta < .05] = np.nan
            return np.stack((x, yn), axis=-1)

        new_slices = np.zeros((1, len(new_t), self.slices.shape[-1]))
        reproject.adaptive.deforest.map_coordinates(
            self.slices.astype(float).reshape((1, *self.slices.shape)),
            new_slices,
            transformer,
            out_of_range_nan=True,
            boundary_mode='ignore',
            center_jacobian=False)
        self.slices = new_slices[0]
        self.times = new_t
        self._title.append(f"resampled dt={new_dt}")

        # Clear out little extra bits that show up around imager gaps
        nans = np.isnan(self.slices)
        nans = scipy.ndimage.binary_closing(nans, np.ones((3, 3)))
        self.slices[nans] = np.nan

    def clamp(self, min=-np.inf, max=np.inf):
        self.slices[self.slices < min] = min
        self.slices[self.slices > max] = max
        self._title.append(f"clamp({min}, {max})")

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

    def merge(self, other):
        self._subtitles.append(self._title)
        self._subtitles.append(other._title)
        self._title = ['Merged']
        self.slices = np.where(np.isfinite(self.slices),
                               self.slices, other.slices)

    def deepcopy(self) -> "BaseJmap":
        return copy.deepcopy(self)

    def plot(self, bundle=None, ax=None, label_vr=False, vmin=None, vmax=None,
             pmin=5, pmax=95, transpose=False):
        min, max = np.nanpercentile(self.slices, [pmin, pmax])
        if vmin is None:
            vmin = min
        if vmax is None:
            vmax = max

        if ax is None:
            ax = plt.gca()
        if len(np.unique(np.diff(self.times))) > 1:
            # In case of floating-point variation, ensure the unique dts differ
            # significantly
            if np.max(np.diff(np.unique(np.diff(self.times)))) > 1:
                warnings.warn("Sequence cadence varies---plot may be garbled")
        
        cmap = copy.deepcopy(plot_utils.wispr_cmap)
        cmap.set_bad("#4d4540")
        image = self.slices
        if transpose:
            image = image.T
            dates = plot_utils.x_axis_dates(self.times, ax=ax)
            extent = [dates[0], dates[-1],
                      self.angles[0], self.angles[-1]]
            ax.set_ylabel(self.xlabel)
        else:
            dates = plot_utils.y_axis_dates(self.times, ax=ax)
            extent = [self.angles[0], self.angles[-1],
                      dates[0], dates[-1]]
            ax.set_xlabel(self.xlabel)
        im = ax.imshow(image,
                       origin='lower',
                       extent=extent,
                       aspect='auto',
                       cmap=cmap,
                       norm=matplotlib.colors.PowerNorm(gamma=1/2.2,
                                                        vmin=vmin,
                                                        vmax=vmax))
        try:
            plt.sci(im)
        except ValueError:
            # May occur if we're plotting onto subplots
            pass

        ax.set_title(self.title)

        if label_vr:
            if bundle is None:
                raise ValueError("Bundle must be provided")
            sc_poses = bundle.sc_poses
            r = np.sqrt(np.sum(sc_poses**2, axis=1))
            vrs = np.gradient(r, self.times)
            vrs = (vrs * u.R_sun / u.s).to(u.km / u.s).value

            def t2vr(t): return np.interp(t, dates, vrs)
            def vr2t(vr): return np.interp(vr, vrs, dates)
            
            if transpose:
                ax2 = plt.gca().secondary_xaxis(
                    'top', functions=(t2vr, vr2t))
                ax2.set_xlabel("S/C radial velocity (km/s)")
            else:
                ax2 = plt.gca().secondary_yaxis(
                    'right', functions=(t2vr, vr2t))
                ax2.set_ylabel("S/C radial velocity (km/s)")


class PlainJMap(BaseJmap):
    xlabel = "Elongation"

    def __init__(self, *args, target_angles=None, **kwargs):
        self.target_angles = target_angles
        super().__init__(*args, **kwargs)

    def derotate(self) -> "DerotatedJMap":
        output = np.empty((self.slices.shape[0], self.transformer.n))
        for i, (slice, angles) in enumerate(
                zip(self.slices, self.target_angles)):
            def transformer(pixel_out):
                px = self.transformer.pix_to_angle(pixel_out[..., 0])
                px_in = np.interp(
                    px,
                    angles,
                    np.arange(len(angles)),
                    left=np.nan, right=np.nan)
                pixel_in = pixel_out.copy()
                pixel_in[..., 0] = px_in
                return pixel_in

            reproject.adaptive.deforest.map_coordinates(
                slice.reshape((1, 1, slice.size)),
                output[i].reshape((1, 1, output[i].size)),
                transformer,
                out_of_range_nan=True,
                boundary_mode='ignore',
                center_jacobian='false')
        outmap = DerotatedJMap(
            slices=output,
            angles=self.transformer.pix_to_angle(np.arange(output.shape[1])),
            times=copy.deepcopy(self.times),
            transformer=copy.deepcopy(self.transformer),
            normalized=self.normalized,
            title_stub=copy.deepcopy(self.title_stub))
        outmap._title = copy.deepcopy(self._title)
        outmap._title.append("derotated")
        outmap._subtitles = copy.deepcopy(self._subtitles)
        return outmap


class DerotatedJMap(BaseJmap):
    xlabel = "Fixed-frame angular position"


@numba.njit
def nan_gaussian_blur(data, radius):
    blurred = np.zeros(data.shape)
    for yo in range(blurred.shape[0]):
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


@numba.njit
def nan_minsmooth(data, radius, percentile):
    output = data.copy()
    for yo in range(output.shape[0]):
        for xo in range(output.shape[1]):
            ystart = max(0, yo - radius)
            ystop = min(yo + radius + 1, data.shape[0])
            xstart = max(0, xo - radius)
            xstop = min(xo + radius + 1, data.shape[1])
            output[yo, xo] -= np.nanpercentile(
                data[ystart:ystop, xstart:xstop],
                percentile)
    return output
