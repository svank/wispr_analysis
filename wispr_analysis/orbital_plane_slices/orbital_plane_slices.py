import copy
import dataclasses
from itertools import repeat
import warnings

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.colors
import matplotlib.pyplot as plt
import numba
import numpy as np
import reproject
import scipy
from sunpy.coordinates import HeliocentricInertial, Helioprojective, NorthOffsetFrame
from tqdm.contrib.concurrent import process_map

from .. import planets, plot_utils


class OrbitalSliceTransformer:
    def __init__(self, sc_pos, img_wcs, angle_start, angle_stop, n):
        self.sc_pos = sc_pos
        self.img_wcs = img_wcs
        self.angle_start = angle_start
        self.angle_stop = angle_stop
        self.n = n

        try:
            self.date =img_wcs.wcs.dateavg.replace('T', ' ')
        except AttributeError:
            self.date = ''
        if self.date == '':
            self.date = '2020-01-01 12:12:12'
        orbital_plane = planets.get_orbital_plane('psp', '2020-01-01 12:12:12')
        p1 = orbital_plane.data[0]
        p2 = orbital_plane.data[20]
        orbital_north = p1.cross(p2)
        orbital_north = SkyCoord(orbital_north, representation_type='cartesian',
                                 frame=HeliocentricInertial, obstime=self.date)
        self.orbital_frame = NorthOffsetFrame(north=orbital_north, obstime=self.date)

    def pix_to_angle(self, pixel_num):
        return pixel_num * (self.angle_stop - self.angle_start) / self.n + self.angle_start

    def __call__(self, pixel_out):
        sc = SkyCoord(*(self.sc_pos * u.R_sun.to(u.m)), representation_type='cartesian',
                      frame=HeliocentricInertial, obstime=self.date, unit='m')
        sc = sc.transform_to(self.orbital_frame)
        sc_cart = sc.represent_as('cartesian')
        
        angles = -self.pix_to_angle(pixel_out[..., 0]) * np.pi/180
        
        dx = np.cos(angles) * u.R_sun.to(u.m)
        dy = np.sin(angles) * u.R_sun.to(u.m)
        points = SkyCoord(sc_cart.x.value + dx, sc_cart.y.value + dy, sc_cart.z.value,
                          frame=self.orbital_frame, representation_type='cartesian',
                          observer=sc, unit='m')
        points = points.transform_to(Helioprojective)
        
        x, y = self.img_wcs.world_to_pixel(points)
        
        g = (x > 0) * (x < self.img_wcs.pixel_shape[0]-1) * (y > 0) * (y < self.img_wcs.pixel_shape[1]-1)
        x[~g] = np.nan
        y[~g] = np.nan

        pixel_in = np.empty_like(pixel_out)
        pixel_in[..., 0] = x
        pixel_in[..., 1] = y
        return pixel_in


def _extract_slice(img, sc_pos, wcs, tmin, tmax, nt):
    transformer = OrbitalSliceTransformer(sc_pos, wcs, tmin, tmax, nt)
    output = np.zeros((1, 1, nt))
    reproject.adaptive.deforest.map_coordinates(
        img.astype(float).reshape((1, *img.shape)),
        output,
        transformer,
        out_of_range_nan=True,
        center_jacobian=False)
    return output[0, 0], transformer


def extract_slices(data_bundle, normalize=True, tmin=0, tmax=360, nt=360, title=""):
    slices = np.empty((data_bundle.images.shape[0], nt), dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*failed to converge.*')
        warnings.filterwarnings('ignore', message='.*All-NaN slice.*')
        for i, (slice, transformer) in enumerate(process_map(
                _extract_slice, data_bundle.images, data_bundle.sc_poses, data_bundle.wcses,
                repeat(tmin), repeat(tmax), repeat(nt), chunksize=2)):
            slices[i] = slice
    slices[~np.isfinite(slices)] = np.nan
    angles = transformer.pix_to_angle(np.arange(slices.shape[1]))
    # if normalize:
    #     slices *= rs**2
    return OrbitalPlaneSlices(
        slices=slices,
        angles=angles,
        times=data_bundle.times,
        transformer=transformer,
        normalized=normalize,
        title_stub=title,
    )


@dataclasses.dataclass
class InputDataBundle:
    images: np.ndarray
    wcses: list[WCS]
    sc_poses: list[tuple]
    times: np.ndarray


@dataclasses.dataclass
class OrbitalPlaneSlices:
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
        Get closer to a 1:1 aspect ratio (e.g. so unsharp-masking is more symmetric)
        """
        if self.slices.shape[0] > self.slices.shape[1]:
            # More rows than columns
            n = int(round(self.slices.shape[0] / self.slices.shape[1]))
            # We'll super-sample each row
            idx = np.linspace(0, self.slices.shape[1]-1, n*self.slices.shape[1] - 1)
            new_slices = np.empty((self.slices.shape[0], len(idx)))
            for i in range(self.slices.shape[0]):
                # Interpolate within each row
                new_slices[i] = np.interp(idx, np.arange(self.slices.shape[1]), self.slices[i])
            self.slices = new_slices
            self.angles = np.interp(idx, np.arange(len(self.angles)), self.angles)
        else:
            # More columns than rows
            n = int(round(self.slices.shape[1] / self.slices.shape[0]))
            # We'll super-sample each column
            idx = np.linspace(0, self.slices.shape[0]-1, n*self.slices.shape[0] - 1)
            new_slices = np.empty((len(idx), self.slices.shape[1]))
            for i in range(self.slices.shape[1]):
                # Interpolate within each column
                new_slices[:, i] = np.interp(idx, np.arange(self.slices.shape[0]), self.slices[:, i])
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
        self.slices = nan_unsharp_mask(self.slices, radius=radius, amount=amount)

    def minsmooth(self, radius, percentile):
        self._title.append(f"minsmooth({radius}px, {percentile})")
        self.slices = nan_minsmooth(self.slices, radius=radius, percentile=percentile)

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
            # timesteps mapping to pretty much the same exact region (the
            # space between two difference slices). For better plotting,
            # let's let the dropout show as a gap and not a big interpolated streak.
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
        nans = scipy.ndimage.binary_closing(nans, np.ones((3,3)))
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
            return toggle * A * x ** exp + (1-toggle) * (B * x ** 2 + C * x + D)
        for i in range(self.slices.shape[0]):
            y = self.slices[i]
            g = np.isfinite(y)
            y = y[g]
            if y.size == 0:
                continue
            x = np.arange(len(y)) + 1
            y *= 1e13
            try:
                popt, _ = scipy.optimize.curve_fit(radial_fcn, x, y, p0=[1, -1, -1, 1, 1, 10], maxfev=30000,
                                      bounds=([-np.inf, -6, -np.inf, -np.inf, -np.inf, 0],
                                              [np.inf, 0, np.inf, np.inf, np.inf, len(y)]))
            except RuntimeError:
                # Use the last row's fit
                pass
            yf = 1e-13 * radial_fcn(x, *popt)
            self.slices[i][g] -= yf
        self._title.append(f"radial detrending")

    def merge(self, other):
        self._subtitles.append(self._title)
        self._subtitles.append(other._title)
        self._title = ['Merged']
        self.slices = np.where(np.isfinite(self.slices), self.slices, other.slices)

    def deepcopy(self) -> "OrbitalPlaneSlices":
        return copy.deepcopy(self)


def make_plane_map_plot(input, bundle=None, axs=None, label_vr=False, vmin=None, vmax=None, pmin=5, pmax=95):
    min, max = np.nanpercentile(input.slices, [pmin, pmax])
    if vmin is None:
        vmin = min
    if vmax is None:
        vmax = max
    
    if axs is None:
        axs = [plt.gca()]
    if len(np.unique(np.diff(input.times))) > 1:
        # In case of floating-point variation, ensure the unique dts differ significantly
        if np.max(np.diff(np.unique(np.diff(input.times)))) > 1:
            warnings.warn("Sequence cadence varies---plot may be garbled")
    for ax in axs:
        dates = plot_utils.y_axis_dates(input.times, ax=ax)
        im = ax.imshow(input.slices, origin='lower',
                  extent=[input.angles[0], input.angles[-1], dates[0], dates[-1]],
                  aspect='auto', cmap=plot_utils.wispr_cmap,
                  norm=matplotlib.colors.PowerNorm(gamma=1/2.2, vmin=vmin, vmax=vmax))
        cs = ax.contourf(
            input.angles,
            dates,
            np.isfinite(input.slices),
            levels=[0, .5, 1],
            hatches=['.', None],
            colors='none')
        cs.collections[0].set_edgecolor('white')
        cs.collections[0].set_alpha(0.2)
        cs.collections[0].set_linewidth(0)
        cs.collections[1].set_alpha(0)
        plt.sci(im)
        
        ax.set_xlabel("Fixed-frame angular position")
        
        ax.set_title(input.title)

        if label_vr:
            if bundle is None:
                raise ValueError("Bundle must be provided")
            sc_poses = bundle.sc_poses
            r = np.sqrt(np.sum(sc_poses**2, axis=1))
            vrs = np.gradient(r, input.times)
            vrs = (vrs * u.R_sun / u.s).to(u.km / u.s).value

            t2vr = lambda t: np.interp(t, dates, vrs)
            vr2t = lambda vr: np.interp(vr, vrs, dates)
            ax2 = plt.gca().secondary_yaxis('right', functions=(t2vr, vr2t))
            ax2.set_ylabel("S/C radial velocity (km/s)")


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
                    weight = np.exp(-dx**2 / 2 / radius**2 - dy**2 / 2 / radius**2)
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
