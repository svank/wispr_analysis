from contextlib import ExitStack, contextmanager
import copy
from dataclasses import dataclass

import astropy.coordinates
import astropy.time
import astropy.units as u
from astropy.wcs import WCS
import numba
import numpy as np
import scipy
import sunpy.coordinates

from .. import utils


class Thing:
    """Represents a generic object with a position and a velocity in 2-space
    
    Subclasses have ``x``, ``y``, ``vx``, and ``vy`` attributes which provide
    those quantities. Instances can be set to a specific time, which then
    determines how those attributes are computed
    """
    t: float = 0
    t_min: float = None
    t_max: float = None
    
    def get_bad_t(self):
        t = np.atleast_1d(self.t)
        bad_t = np.zeros(len(t), dtype=bool)
        if self.t_min is not None:
            bad_t[t < self.t_min] = True
        if self.t_max is not None:
            bad_t[t > self.t_max] = True
        return bad_t
    
    def process_t_bounds(self, quantity):
        bad_t = self.get_bad_t()
        if isinstance(self.t, (int, float)):
            if bad_t:
                return np.nan
            return quantity
        if np.any(bad_t):
            quantity = np.asarray(quantity, dtype=float)
            if len(quantity.shape) == 0:
                quantity = np.full(bad_t.size, quantity)
            quantity[bad_t] = np.nan
        return quantity
    
    def is_same_time(self, other):
        """ Returns True if the two objects are set to the same time """
        return np.all(self.t == other.t)
    
    def at(self, t):
        """ Returns a copy of the object at time ``t``. """
        t = np.atleast_1d(t)
        out = self.copy()
        out.set_t(t)
        return out
    
    @contextmanager
    def at_temp(self, t):
        if t is None:
            yield self
            return
        old_t = self.t
        self.set_t(t)
        yield self
        self.t = old_t
     
    def set_t(self, t):
        """ Sets the object's time to ``t``. """
        t = np.atleast_1d(t)
        self.t = t
    
    def in_front_of(self, other: "Thing", t=None):
        """Returns whether this object is in front of the given object.
        
        "In front" is defined relative to the forward direction of the other
        object.
        
        Parameters
        ----------
        other : `Thing`
            Another `Thing` instance
        t : float
            Optionally, do the comparison ``t`` seconds into the future.
        
        Returns
        -------
        in_front : boolean
            ``True`` if this object is in front of ``other``.
        """
        if t is None and not self.is_same_time(other):
            raise ValueError(
                    "Objects are not set at same time---must specify `t`")
        with other.at_temp(t) as other, self.at_temp(t) as self:
            separation_vector = self - other
            angle = angle_between_vectors(
                other.vx,
                other.vy,
                other.vz,
                separation_vector.x,
                separation_vector.y,
                separation_vector.z)
            in_front = np.atleast_1d(np.abs(angle) < np.pi/2)
            in_front[separation_vector.r == 0] = False
            return in_front
    
    @property
    def r(self):
        """
        Convenience access to sqrt(x**2 + y**2 + z**2)
        """
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    @property
    def v(self):
        """
        Convenience access to sqrt(vx**2 + vy**2 + vz**2)
        """
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
    
    def copy(self):
        """ Returns a deep copy of this object """
        return copy.deepcopy(self)
    
    def __sub__(self, other):
        if self.is_same_time(other):
            t = self.t
        else:
            t = None
        return DifferenceThing(self, other, t)


@dataclass
class LinearThing(Thing):
    """ Represents an object with constant velocity """
    x_t0: float = 0
    y_t0: float = 0
    z_t0: float = 0
    vx_t0: float = 0
    vy_t0: float = 0
    vz_t0: float = 0
    
    def __init__(self, x=0, y=0, z=0, vx=0, vy=0, vz=0,
            t=0, t_min=None, t_max=None):
        """
        Accepts physical parameters, as well as the corresponding time
        
        That is, ``x``, ``y``, etc. need not be specified for ``t=0`` if the
        appropriate time is provided.
        
        """
        self.vx_t0 = vx
        self.vy_t0 = vy
        self.vz_t0 = vz
        self.t = t
        
        self.x_t0 = x - vx * t
        self.y_t0 = y - vy * t
        self.z_t0 = z - vz * t
        
        self.t_min = t_min
        self.t_max = t_max
    
    @property
    def x(self):
        x = self.x_t0 + self.vx_t0 * self.t
        x = self.process_t_bounds(x)
        return x
    
    @property
    def y(self):
        y = self.y_t0 + self.vy_t0 * self.t
        y = self.process_t_bounds(y)
        return y
    
    @property
    def z(self):
        z = self.z_t0 + self.vz_t0 * self.t
        z = self.process_t_bounds(z)
        return z
    
    @property
    def vx(self):
        vx = self.process_t_bounds(self.vx_t0)
        return vx
    
    @vx.setter
    def vx(self, value):
        self.vx_t0 = value
    
    @property
    def vy(self):
        vy = self.process_t_bounds(self.vy_t0)
        return vy
    
    @vy.setter
    def vy(self, value):
        self.vy_t0 = value
    
    @property
    def vz(self):
        vz = self.process_t_bounds(self.vz_t0)
        return vz
    
    @vz.setter
    def vz(self, value):
        self.vz_t0 = value
    
    def offset_by_time(self, dt):
        out = self.copy()
        out.x_t0 += out.vx_t0 * dt
        out.y_t0 += out.vy_t0 * dt
        out.z_t0 += out.vz_t0 * dt
        return out


@dataclass
class ArrayThing(Thing):
    """
    Represents an object whose position over time is specified numerically
    
    Positions are provided at a number of points in time, and those positions
    are interpolated between as needed.
    """
    
    xlist: np.ndarray = 0
    ylist: np.ndarray = 0
    zlist: np.ndarray = 0
    tlist: np.ndarray = 0
    
    def __init__(self, tlist, xlist=0, ylist=0, zlist=0,
            t=0, t_min=None, t_max=None):
        """
        Parameters
        ----------
        tlist : ``np.ndarray``
            The list of time points at which positions are provided
        xlist, ylist, zlist : ``np.ndarray`` or float
            The specified positions. If either is a single number, that number
            is used at all points in time.
        t : float
            The time the object is currently at.
        """
        xlist = np.atleast_1d(xlist)
        ylist = np.atleast_1d(ylist)
        zlist = np.atleast_1d(zlist)
        tlist = np.atleast_1d(tlist)
        
        # Check that tlist is sorted
        if np.any(np.diff(tlist) < 0):
            raise ValueError("tlist must be sorted")
        
        if len(xlist) == 1:
            xlist = np.repeat(xlist, len(tlist))
        if len(ylist) == 1:
            ylist = np.repeat(ylist, len(tlist))
        if len(zlist) == 1:
            zlist = np.repeat(zlist, len(tlist))
        
        if len(xlist) != len(tlist):
            raise ValueError("Invalid length for xlist")
        if len(ylist) != len(tlist):
            raise ValueError("Invalid length for ylist")
        if len(zlist) != len(tlist):
            raise ValueError("Invalid length for zlist")
        
        self.xlist = xlist
        self.ylist = ylist
        self.zlist = zlist
        self.tlist = tlist
        self.t = t
        
        self.t_min = t_min
        self.t_max = t_max
    
    @property
    def x(self):
        x = scipy.interpolate.interp1d(self.tlist, self.xlist)(self.t)
        x = self.process_t_bounds(x)
        return x
    
    @property
    def y(self):
        y = scipy.interpolate.interp1d(self.tlist, self.ylist)(self.t)
        y = self.process_t_bounds(y)
        return y
    
    @property
    def z(self):
        z = scipy.interpolate.interp1d(self.tlist, self.zlist)(self.t)
        z = self.process_t_bounds(z)
        return z
    
    @property
    def vx(self):
        interpolator = scipy.interpolate.interp1d(self.tlist, self.xlist)
        dt = .0001
        vx = self._finite_difference(interpolator, dt)
        vx = self.process_t_bounds(vx)
        return vx
    
    @property
    def vy(self):
        interpolator = scipy.interpolate.interp1d(self.tlist, self.ylist)
        dt = .0001
        vy = self._finite_difference(interpolator, dt)
        vy = self.process_t_bounds(vy)
        return vy
    
    @property
    def vz(self):
        interpolator = scipy.interpolate.interp1d(self.tlist, self.zlist)
        dt = .0001
        vz = self._finite_difference(interpolator, dt)
        vz = self.process_t_bounds(vz)
        return vz
    
    def _finite_difference(self, interpolator, dt):
        try:
            y1 = interpolator(self.t - dt/2)
            y2 = interpolator(self.t + dt/2)
        except ValueError:
            try:
                y1 = interpolator(self.t)
                y2 = interpolator(self.t + dt)
            except ValueError:
                try:
                    y1 = interpolator(self.t - dt)
                    y2 = interpolator(self.t)
                except:
                    # If out list of times includes both endpoints exactly,
                    # none of these above will work for every time, but one
                    # will work for each time. So compute each time
                    # individually.
                    vals = []
                    for t in self.t:
                        with self.at_temp(t) as s:
                            vals.append(s._finite_difference(interpolator, dt))
                    return np.concatenate(vals)
        return (y2 - y1) / dt
    
    def offset_by_time(self, dt):
        out = self.copy()
        out.tlist += dt
        return out

class DifferenceThing(Thing):
    """ Represents a difference between two Things """
    
    def __init__(self, thing1, thing2, t):
        self.thing1 = thing1.copy()
        self.thing2 = thing2.copy()
        self.t = t
    
    @property
    def x(self):
        with (self.thing1.at_temp(self.t) as thing1,
              self.thing2.at_temp(self.t) as thing2):
            return thing1.x - thing2.x
    
    @property
    def y(self):
        with (self.thing1.at_temp(self.t) as thing1,
              self.thing2.at_temp(self.t) as thing2):
            return thing1.y - thing2.y
    
    @property
    def z(self):
        with (self.thing1.at_temp(self.t) as thing1,
              self.thing2.at_temp(self.t) as thing2):
            return thing1.z - thing2.z
    
    @property
    def vx(self):
        dt = .0001
        with (self.at_temp(self.t - dt/2)):
            before = self.x
        with (self.at_temp(self.t + dt/2)):
            after = self.x
        return (after - before) / dt
    
    @property
    def vy(self):
        dt = .0001
        with (self.at_temp(self.t - dt/2)):
            before = self.y
        with (self.at_temp(self.t + dt/2)):
            after = self.y
        return (after - before) / dt
    
    @property
    def vz(self):
        dt = .0001
        with (self.at_temp(self.t - dt/2)):
            before = self.z
        with (self.at_temp(self.t + dt/2)):
            after = self.z
        return (after - before) / dt


def angle_between_vectors(x1, y1, z1, x2, y2, z2):
    """Returns a signed angle between two vectors"""
    # Rotate so v1 is our x axis. We want the angle v2 makes to the x axis.
    # Its components in this rotated frame are its dot and cross products
    # with v1.
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    y1 = np.atleast_1d(y1)
    y2 = np.atleast_1d(y2)
    z1 = np.atleast_1d(z1)
    z2 = np.atleast_1d(z2)
    
    dot_product = x1 * x2 + y1 * y2 + z1 * z2
    cross_x = (y1*z2 - z1*y2)
    cross_y = (z1*x2 - x1*z2)
    cross_z = (x1*y2 - y1*x2)
    det = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
    
    angle = np.arctan2(det, dot_product)
    v1_is_zero = ((x1 == 0) * (y1 == 0) * (z1 == 0))
    v2_is_zero = ((x2 == 0) * (y2 == 0) * (z2 == 0))
    angle[v1_is_zero + v2_is_zero] = np.nan
    return angle


def calc_hpc(sc: "Thing", parcels: list["Thing"], t=None):
    was_not_list = False
    if isinstance(parcels, Thing):
        parcels = [parcels]
        was_not_list = True
    
    with ExitStack() as stack:
        sc = stack.enter_context(sc.at_temp(t))
        parcels = [stack.enter_context(p.at_temp(t)) for p in parcels]
    
        px = np.array([p.x for p in parcels])
        py = np.array([p.y for p in parcels])
        pz = np.array([p.z for p in parcels])
        
        Tx, Ty = xyz_to_hpc(px, py, pz, sc.x, sc.y, sc.z)
        if was_not_list:
            Tx = Tx[0]
            Ty = Ty[0]
        return Tx, Ty
    

def xyz_to_hpc(xs, ys, zs, scx, scy, scz):
    obstime = '2023/07/05T04:21:00'
    sc_hc = astropy.coordinates.SkyCoord(x=scx, y=scy, z=scz, unit='m',
            representation_type='cartesian', obstime=obstime,
            frame=sunpy.coordinates.frames.HeliocentricInertial)
    p_hc = astropy.coordinates.SkyCoord(x=xs, y=ys, z=zs, unit='m',
            representation_type='cartesian', obstime=obstime,
            frame=sunpy.coordinates.frames.HeliocentricInertial)
    p_hpc = p_hc.transform_to(
            sunpy.coordinates.frames.Helioprojective(
                observer=sc_hc, obstime=obstime))
    
    Tx = p_hpc.Tx.to(u.deg).value
    Ty = p_hpc.Ty.to(u.deg).value
    
    return Tx, Ty


def hpc_to_elpa(Tx, Ty):
    Tx = np.deg2rad(Tx)
    Ty = np.deg2rad(Ty)
    
    elongation = np.arctan2(
            np.sqrt(np.cos(Ty)**2 * np.sin(Tx)**2 + np.sin(Ty)**2),
            np.cos(Ty) * np.cos(Tx)
            )
    pa = np.arctan2(
        -np.cos(Ty) * np.sin(Tx),
        np.sin(Ty)
    )
    
    return np.rad2deg(elongation), np.rad2deg(pa)


def synthesize_image(sc, parcels, t0, fov=95, projection='ARC',
        output_size_x=200, output_size_y=200, parcel_width=1, image_wcs=None,
        celestial_wcs=False, fixed_fov_range=None):
    """Produce a synthetic WISPR image

    Parameters
    ----------
    sc : `Thing`
        Object representing the spacecraft
    parcels : ``List[Thing]``
        Objects representing the plasma blobs
    t0 : float
        The time at which to generate the image
    fov : int, optional
        Camera field-of-view width
    projection : str, optional
        Projection to use
    output_size_x, output_size_y : int, optional
        Size of the generated image
    parcel_width : int, optional
        Width of each plasma blob, in R_sun
    image_wcs : ``WCS``, optional
        The WCS to use for the output image. If not provided, a WISPR-like WCS
        is generated
    celestial_wcs : bool, optional
        Whether to convert the output WCS to RA/Dec

    Returns
    -------
    output_image : ``np.ndarray``
        The output image
    image_wcs : ``WCS``
        The corresponding WCS
    """
    sc = sc.at(t0)
    date = astropy.time.Time(t0, format='unix').fits
    if isinstance(parcel_width, u.Quantity):
        parcel_width = parcel_width.to(u.m).value
    else:
        parcel_width *= u.R_sun.to(u.m)
    
    # Build output image WCS
    if image_wcs is None:
        image_wcs = WCS(naxis=2)
        ## Find elongation of s/c forward direction by computing the angle
        ## between it and the sunward direction
        #forward_elongation = angle_between_vectors(sc.vx, sc.vy, -sc.x, -sc.y)
        ## Set the reference pixel coordinates as the forward-direction
        ## elongation for longitude, and zero latitude (assume s/c is in
        ## ecliptic plane)
        #image_wcs.wcs.crval = forward_elongation[0] * 180 / np.pi, 0
        
        # Set the reference pixel coordinates as 61 degrees HPC, which is the
        # center of WISPR's composite FOV (as detailed in the in-flight
        # calibration paper)
        image_wcs.wcs.crval = 61, 0
        
        # Set the reference pixel to be the central pixel of the image
        image_wcs.wcs.crpix = output_size_x/2 + .5, output_size_y/2 + .5
        # Set the degrees/pixel value so that our specified FOV fits in the
        # image
        cdelt = fov / max(output_size_x, output_size_y)
        image_wcs.wcs.cdelt = cdelt, cdelt
        image_wcs.wcs.ctype = f"HPLN-{projection}", f"HPLT-{projection}"
        image_wcs.wcs.cunit = "deg", "deg"
        image_wcs.array_shape = (output_size_y, output_size_x)
    
    output_image = np.zeros((output_size_y, output_size_x))
    
    x = np.arange(output_image.shape[1])
    y = np.arange(output_image.shape[0])
    
    do_interp = len(x) > 100 and len(y) > 100
    if do_interp:
        slice = np.s_[1::3]
    else:
        slice = np.s_[::1]
    xx, yy = np.meshgrid(x[slice], y[slice])
    xx_full, yy_full = np.meshgrid(x, y)
    # Compute the LOS direction of each pixel (as helioprojective coordinates)
    los = image_wcs.pixel_to_world(xx, yy)
    Tx, Ty = los.Tx, los.Ty
    obstime = date
    sc_coord = astropy.coordinates.SkyCoord(
        sc.x, sc.y, sc.z, frame='heliocentricinertial',
        representation_type='cartesian', unit='m', obstime=obstime)
    # Turn the LOS directions into distant points and transform them to HCI
    # (x,y,z) points
    los = astropy.coordinates.SkyCoord(
        Tx, Ty, 100*u.au, frame='helioprojective', observer=sc_coord)
    los = los.transform_to('heliocentricinertial').cartesian
    los = np.array(
        [los.x.to(u.m).value, los.y.to(u.m).value, los.z.to(u.m).value])
    los = np.transpose(los, axes=(1,2,0))
    sc_pos = np.array([sc.x, sc.y, sc.z]).flatten()
    # Pre-compute x for closest-approach finding
    x_old = los - sc_pos
    x_over_xdotx = x_old / np.sum(x_old**2, axis=-1, keepdims=True)
    
    if do_interp:
        x_new = np.empty((*yy_full.shape, 3))
        x_new[..., 0] = scipy.interpolate.RegularGridInterpolator(
            (y[slice], x[slice]), x_old[..., 0], method='linear',
            bounds_error=False, fill_value=None)((yy_full, xx_full))
        x_new[..., 1] = scipy.interpolate.RegularGridInterpolator(
            (y[slice], x[slice]), x_old[..., 1], method='linear',
            bounds_error=False, fill_value=None)((yy_full, xx_full))
        x_new[..., 2] = scipy.interpolate.RegularGridInterpolator(
            (y[slice], x[slice]), x_old[..., 2], method='linear',
            bounds_error=False, fill_value=None)((yy_full, xx_full))
        x_over_xdotx = scipy.interpolate.RegularGridInterpolator(
            (y[slice], x[slice]), x_over_xdotx, method='linear',
            bounds_error=False, fill_value=None)((yy_full, xx_full))
        x = x_new
    else:
        x = x_old
    
    # Find the center of each parcel as a pixel position
    parcel_poses = np.empty((len(parcels), 3))
    for i, parcel in enumerate(parcels):
        with parcel.at_temp(t0) as p:
            parcel_poses[i] = p.x[0], p.y[0], p.z[0]
    p_coord = astropy.coordinates.SkyCoord(
        parcel_poses[..., 0], parcel_poses[..., 1], parcel_poses[..., 2],
        frame='heliocentricinertial', representation_type='cartesian',
        unit='m', obstime=obstime, observer=sc_coord)
    px, py = image_wcs.world_to_pixel(p_coord)
    np.nan_to_num(px, copy=False)
    np.nan_to_num(py, copy=False)
    px = px.astype(int)
    py = py.astype(int)
    
    for i in range(len(parcels)):
        # Draw each parcel onto the output canvas
        parcel_pos = parcel_poses[i]
        r = np.linalg.norm(parcel_pos)
        _synth_data_one_parcel(sc_pos, parcel_pos, x, x_over_xdotx,
                               parcel_width, output_image,
                               r, px[i], py[i])
    
    if celestial_wcs:
        image_wcs.wcs.ctype = f"RA---{projection}", f"DEC--{projection}"
        to_sun_x = -sc.x[0]
        to_sun_y = -sc.y[0]
        to_sun_theta = np.arctan2(to_sun_y, to_sun_x) * 180 / np.pi
        fov_center = to_sun_theta - 61
        image_wcs.wcs.crval = fov_center, 0
        cdelt = image_wcs.wcs.cdelt[0]
        image_wcs.wcs.cdelt = -cdelt, cdelt
    else:
        image_wcs.wcs.dateobs = date
        image_wcs.wcs.dateavg = date
        c = sc_coord.transform_to('heliographic_stonyhurst')
        image_wcs.wcs.aux.hgln_obs = c.lon.to(u.deg).value
        image_wcs.wcs.aux.hglt_obs = c.lat.to(u.deg).value
        image_wcs.wcs.aux.dsun_obs = c.radius.to(u.m).value
        with utils.ignore_fits_warnings():
            image_wcs.fix()
    
    if fixed_fov_range is not None:
        fixed_fov_start = np.arctan2(-sc.y, -sc.x) * 180 / np.pi - 13.5
        fixed_fov_stop = np.arctan2(-sc.y, -sc.x) * 180 / np.pi - 108
        add_left = ((fixed_fov_range[0] - fixed_fov_start) % 360) / np.abs(image_wcs.wcs.cdelt[0])
        add_right = ((fixed_fov_stop - fixed_fov_range[1]) % 360) / np.abs(image_wcs.wcs.cdelt[0])
        output_image = np.pad(
                output_image,
                ((0, 0), (int(np.round(add_left)), int(np.round(add_right)))))
        crpix = image_wcs.wcs.crpix
        image_wcs.wcs.crpix = crpix[0] + add_left[0], crpix[1]
        
    return output_image, image_wcs


@numba.njit(cache=True)
def _synth_data_one_parcel(sc_pos, parcel_pos, x, x_over_xdotx, parcel_width,
                           output_image, p_r, start_x, start_y):
    # Check if this time point is valid for this parcel
    if np.isnan(parcel_pos[0]):
        return
    
    d_sc2 = np.sum((parcel_pos - sc_pos)**2)
    
    # Clamp the starting point to the image bounds
    start_x = max(0, start_x)
    start_x = min(output_image.shape[1], start_x)
    start_y = max(0, start_y)
    start_y = min(output_image.shape[0], start_y)
    
    # Iterate down from the start point
    for i in range(start_y, output_image.shape[0]):
        n = 0
        # Iterate right from the start point
        for j in range(start_x, output_image.shape[1]):
            if _synth_data_one_pixel(
                    i, j, x, x_over_xdotx, parcel_pos, sc_pos,
                    parcel_width, d_sc2, p_r, output_image):
                # Flux was contributed
                n += 1
                continue
            # Flux was not contributed---we've reached the right-hand edge of
            # the parcel in this row
            break
        
        # Iterate left from the start point
        for j in range(start_x-1, -1, -1):
            if _synth_data_one_pixel(
                    i, j, x, x_over_xdotx, parcel_pos, sc_pos,
                    parcel_width, d_sc2, p_r, output_image):
                # Flux was contributed
                n += 1
                continue
            # Flux was not contributed---we've reached the left-hand edge of
            # the parcel in this row
            break
        if n == 0:
            # No flux was contributed in this row---we're at the bottom edge of
            # this parcel.
            break
    
    # Iterate up from the start point
    for i in range(start_y-1, -1, -1):
        n = 0
        # Iterate right from the start point
        for j in range(start_x, output_image.shape[1]):
            if _synth_data_one_pixel(
                    i, j, x, x_over_xdotx, parcel_pos, sc_pos,
                    parcel_width, d_sc2, p_r, output_image):
                # Flux was contributed
                n += 1
                continue
            # Flux was not contributed---we've reached the right-hand edge of
            # the parcel in this row
            break
        
        for j in range(start_x-1, -1, -1):
            if _synth_data_one_pixel(
                    i, j, x, x_over_xdotx, parcel_pos, sc_pos,
                    parcel_width, d_sc2, p_r, output_image):
                # Flux was contributed
                n += 1
                continue
            # Flux was not contributed---we've reached the left-hand edge of
            # the parcel in this row
            break
        if n == 0:
            # No flux was contributed in this row---we're at the top edge of
            # this parcel.
            break
            
            
@numba.njit(cache=True)
def _synth_data_one_pixel(i, j, x, x_over_xdotx, parcel_pos, sc_pos,
                                 parcel_width, d_sc2, p_r, output_image):
    # Compute the closest-approach distance between each LOS segment
    # and the parcel center, following
    # https://stackoverflow.com/a/50728570. x is the projection of the
    # parcel's position on the line between the s/c and a distant
    # point, and t is the fractional position of the closest approach
    # along that line segment.
    t = (x_over_xdotx[i, j, 0] * (parcel_pos[0] - sc_pos[0])
        + x_over_xdotx[i, j, 1] * (parcel_pos[1] - sc_pos[1])
        + x_over_xdotx[i, j, 2] * (parcel_pos[2] - sc_pos[2]))
    
    # Where the closest approach isn't along that segment (i.e. it's
    # behind the s/c), flag it
    if t < 0 or t > 1:
        return False
    
    # Compute the actual coordinates of the closest-approach point
    # This is how we would do it, but let's not save intermediary values
    # closest_approach = t * x[i, j] + sc_pos
    
    # Compute how close the closest approach is to the parcel center
    d_p = np.sqrt(
              (t * x[i, j, 0] + sc_pos[0] - parcel_pos[0])**2
            + (t * x[i, j, 1] + sc_pos[1] - parcel_pos[1])**2
            + (t * x[i, j, 2] + sc_pos[2] - parcel_pos[2])**2
            )
    
    # Truncate the gaussian
    if d_p > parcel_width:
        return False
    
    flux = np.exp(-d_p**2 / (parcel_width/6)**2 / 2)
    # Scale for the Sun-parcel distance and the parcel-s/c distance
    if d_sc2 < parcel_width**2:
        # Ramp down the flux as the parcel gets really close to the
        # s/c, to avoid flashiness, etc.
        flux *= np.sqrt(d_sc2) / parcel_width
    # Note: don't scale for the s/c-parcel distance---a longer distance
    # means 1/r^2 falloff in flux but an r^2 increase in plasma volume
    # along the LOS 
    rsun = 695700000.0 # m
    flux *= 1 / (p_r / rsun)**2
    
    output_image[i, j] += flux
    
    return True


def calculate_radiant(sc, parcel, t0=0):
    t0 = np.atleast_1d(t0)
    if len(t0) > 1 or t0 != 0:
        sc = sc.at(t0)
        parcel = parcel.at(t0)
    infront = np.atleast_1d(parcel.in_front_of(sc))
    if not np.any(infront):
        return np.full(max(len(t0), len(np.atleast_1d(parcel.x))), np.nan)
    v_sc = sc.v
    e_sc = np.atleast_1d(
            angle_between_vectors(sc.vx, sc.vy, 0, -sc.x, -sc.y, 0))
    v_p = parcel.v
    dphi = np.atleast_1d(
            angle_between_vectors(sc.x, sc.y, 0, parcel.x, parcel.y, 0))
    epsilons = np.linspace(0, np.pi, 300)[None, :]
    with np.errstate(divide='ignore'):
        i = np.argmin(
                np.abs(1 - v_sc / v_p
                    * np.sin(e_sc[:, None] - epsilons)
                    / np.sin(epsilons + dphi[:, None])),
                axis=1)
    epsilon = epsilons[:, i][0]
    epsilon[~infront] = np.nan
    return epsilon


def elongation_to_FOV(sc, elongation):
    """Converts elongations to FOV coordinates.
    
    FOV coordinates are in radians with 0 in the direction of s/c travel,
    and increase to the right in the image plane.
    
    Parameters
    ----------
    sc : `Thing`
        A `Thing` representing the observer. Can be vectorized over time.
    elongation : scalar or ``ndarray``
        Elongation values to be converted. Measured in radians, with zero
        representing the Sunward direction. Unsigned.
    
    Returns
    -------
    fov : ``ndarray``
        Field-of-view position in radians
    """
    sc_direction = angle_between_vectors(
            sc.vx, sc.vy, 0, -sc.x, -sc.y, 0)
    return elongation - sc_direction
