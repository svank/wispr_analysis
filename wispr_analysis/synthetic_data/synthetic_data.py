from contextlib import ExitStack, contextmanager
import copy
import dataclasses

import astropy.constants as c
import astropy.coordinates
import astropy.time
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.ticker
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
import sunpy.coordinates

from .. import orbital_frame, planets, utils


class Thing:
    """Represents a generic object with a position and a velocity in 2-space
    
    Subclasses have ``x``, ``y``, ``vx``, and ``vy`` attributes which provide
    those quantities. Instances can be set to a specific time, which then
    determines how those attributes are computed
    """
    t: float = 0
    t_min: float = None
    t_max: float = None
    
    def strip_units(self):
        out = self.copy()
        for attr in 't', 't_min', 't_max':
            value = getattr(out, attr)
            if isinstance(value, u.Quantity):
                setattr(out, attr, value.si.value)
        return out
    
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
        if (isinstance(self.t, (int, float))
                or (isinstance(self.t, np.ndarray)
                    and self.t.shape == tuple())):
            if bad_t:
                return np.nan
            return quantity
        if np.any(bad_t):
            quantity = np.broadcast_to(quantity, bad_t.shape, subok=True)
            quantity = quantity.copy().astype(float)
            quantity[bad_t] = np.nan
        else:
            if isinstance(quantity, np.ndarray):
                quantity = quantity.astype(float)
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
            angle = utils.angle_between_vectors(
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


@dataclasses.dataclass
class LinearThing(Thing):
    """ Represents an object with constant velocity """
    x_t0: float = 0
    y_t0: float = 0
    z_t0: float = 0
    vx_t0: float = 0
    vy_t0: float = 0
    vz_t0: float = 0
    rperp0: float = 1
    rpar0: float = 1
    rho0: float = 1
    rperp_r2: bool = False
    density_r2: bool = False
    
    def strip_units(self):
        out = super().strip_units()
        for attr in ('x_t0', 'y_t0', 'z_t0', 'vx_t0', 'vy_t0', 'vz_t0',
                     'rperp0', 'rpar0', 'rho0'):
            value = getattr(out, attr)
            if isinstance(value, u.Quantity):
                setattr(out, attr, value.si.value)
        return out
    
    def __init__(self, x=0, y=0, z=0, vx=0, vy=0, vz=0,
                 t=0, t_min=None, t_max=None,
                 rperp=1, rpar=1, rho=1, density_r2=False, rperp_r2=False):
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
        
        self.rperp0 = rperp
        self.rpar0 = rpar
        self.rho0 = rho
        self.rperp_r2 = rperp_r2
        self.density_r2 = density_r2
    
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
    
    @property
    def rperp(self):
        rperp = self.rperp0
        if self.rperp_r2:
            rperp *= 1 / self.r**2
        return rperp
    
    @rperp.setter
    def rperp(self, value):
        self.rperp = value
    
    @property
    def rpar(self):
        rpar = self.rpar0
        return rpar
    
    @rpar.setter
    def rpar(self, value):
        self.rpar = value
    
    @property
    def rho(self):
        rho = self.rho0
        if self.density_r2:
            rho *= 1 / self.r**2
        return rho
    
    @rho.setter
    def rho(self, value):
        self.rho0 = value
    
    def offset_by_time(self, dt):
        out = self.copy()
        out.x_t0 += out.vx_t0 * dt
        out.y_t0 += out.vy_t0 * dt
        out.z_t0 += out.vz_t0 * dt
        return out


@dataclasses.dataclass
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
    rperplist: np.ndarray = 0
    rparlist: np.ndarray = 0
    rholist: np.ndarray = 0
    scale_rho_r2: bool = False
    
    def strip_units(self):
        out = super().strip_units()
        for attr in ('xlist', 'ylist', 'zlist', 'tlist', 'rperplist',
                     'rparlist', 'rholist'):
            value = getattr(out, attr)
            if isinstance(value, u.Quantity):
                setattr(out, attr, value.si.value)
        return out
    
    def __init__(self, tlist, xlist=0, ylist=0, zlist=0,
            t=0, t_min=None, t_max=None, rperplist=1, rparlist=1,
            rholist=1, default_density_r2=False):
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
        if len(tlist) <= 1:
            raise ValueError("tlist must have >= 2 entries")
        if np.any(np.diff(tlist) < 0):
            raise ValueError("tlist must be sorted")
        
        for var in ('x', 'y', 'z', 't', 'rperp', 'rpar', 'rho'):
            name = var + 'list'
            arr = np.atleast_1d(locals()[name])
            if len(arr) == 1:
                arr = np.repeat(arr, len(tlist))
                if var == 'rho':
                    self.scale_rho_r2 = default_density_r2
            if len(arr) != len(tlist):
                raise ValueError(f"Invalid length for {name}")
            setattr(self, name, arr)
        
        self.t = t
        self.t_min = t_min
        self.t_max = t_max
    
    def _access_quantity(self, data_list):
        t_vals = self.process_t_bounds(np.atleast_1d(self.t))
        x = np.interp(t_vals, self.tlist, data_list, left=np.nan, right=np.nan)
        return x
    
    @property
    def x(self):
        return self._access_quantity(self.xlist)
    
    @property
    def y(self):
        return self._access_quantity(self.ylist)
    
    @property
    def z(self):
        return self._access_quantity(self.zlist)
    
    @property
    def vx(self):
        dt = .0001
        if isinstance(self.t, u.Quantity):
            dt *= u.s
        vx = self._finite_difference(self.xlist, dt)
        return vx
    
    @property
    def vy(self):
        dt = .0001
        if isinstance(self.t, u.Quantity):
            dt *= u.s
        vy = self._finite_difference(self.ylist, dt)
        return vy
    
    @property
    def vz(self):
        dt = .0001
        if isinstance(self.t, u.Quantity):
            dt *= u.s
        vz = self._finite_difference(self.zlist, dt)
        return vz
    
    @property
    def rperp(self):
        values = self._access_quantity(self.rperplist)
        return values
    
    @property
    def rpar(self):
        values = self._access_quantity(self.rparlist)
        return values
    
    @property
    def rho(self):
        values = self._access_quantity(self.rholist)
        if self.scale_rho_r2:
            values /= self.r**2
        return values
    
    def _finite_difference(self, spatial_quantity, dt):
        tvals = self.process_t_bounds(np.atleast_1d(self.t))
        interpolator = lambda t: np.interp(t, self.tlist, spatial_quantity,
                                           left=np.nan, right=np.nan)
        try:
            y1 = interpolator(tvals - dt/2)
            y2 = interpolator(tvals + dt/2)
        except ValueError:
            try:
                y1 = interpolator(tvals)
                y2 = interpolator(tvals + dt)
            except ValueError:
                try:
                    y1 = interpolator(tvals - dt)
                    y2 = interpolator(tvals)
                except Exception as e:
                    # If out list of times includes both endpoints exactly,
                    # none of these above will work for every time, but one
                    # will work for each time. So compute each time
                    # individually.
                    if len(tvals) == 1:
                        raise e
                    output = np.empty(tvals.shape)
                    if isinstance(tvals, u.Quantity):
                        output <<= spatial_quantity.unit / tvals.unit
                    for i, t in enumerate(tvals):
                        if np.isfinite(t):
                            with self.at_temp(t) as s:
                                output[i] = (
                                    s._finite_difference(spatial_quantity, dt))
                        else:
                            output[i] = np.nan
                    return output
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
        celestial_wcs=False, fixed_fov_range=None, output_quantity='flux',
        point_forward=False, dmin=None, dmax=None, dsunmin=None, dsunmax=None,
        only_side_of_sun=False, antialias=True,
        thomson=True, use_density=True, expansion=True, scale_sun_dist=True):
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
    parcel_width : int or Quantity, optional
        Width of each plasma blob, in R_sun
    image_wcs : ``WCS``, optional
        The WCS to use for the output image. If not provided, a WISPR-like WCS
        is generated
    celestial_wcs : ``bool``, optional
        Whether to convert the output WCS to RA/Dec
    output_quantity : ``str``
        The quantity to show in the output image. Allowed values are 'flux' and
        'distance'.
    point_forward : ``bool``
        If True, point the camera in the forward direction, instead of real
        WISPR Sun-relative pointing.
    dmin, dmax : ``float``
        If given, only render parcels when their distance to the camera is
        within this range.
    antialias : ``bool``
        If True, a hacky quasi-antialiasing is applied to parcels that appear
        small compared to the pixel size
    thomson : ``bool``
        If True, Thomson scattering is modeled (with more efficient scattering
        in the original photon travel direction).
    use_density : ``bool``
        If True, intensities are scaled with the parcel density.
    expansion : ``bool``
        If True, parcels grow with distance from the Sun. The visible width of
        the parcels is modeled as r^(2/3), so the parcel volume scales
        appropriately with the r^2 growth. Intensities are also scaled by a
        line-of-sight-length through the parcel, which grows with the parcel
        size perpendicular to the parcel--Sun radial line (which grows as r^2)
        and the size parallel to the radial line (which grows when there is a
        radial velocity gradient). These two effects are combined according to
        the viewing angle.

    Returns
    -------
    output_image : ``np.ndarray``
        The output image
    image_wcs : ``WCS``
        The corresponding WCS
    """
    output_quantity = output_quantity.lower()
    sc = sc.at(t0)
    date = astropy.time.Time(t0, format='unix').fits
    if isinstance(parcel_width, u.Quantity):
        parcel_width = parcel_width.to(u.m).value
    else:
        parcel_width *= u.R_sun.to(u.m)
    
    # Build output image WCS
    if image_wcs is None:
        image_wcs = WCS(naxis=2)
        if point_forward:
            x, y, z = sc.x, sc.y, sc.z
            sc_soon = sc.at(t0 + 1)
            xp, yp, zp = sc_soon.x, sc_soon.y, sc_soon.z
            observer = astropy.coordinates.SkyCoord(
                x, y, z, representation_type='cartesian', obstime=date,
                frame='heliocentricinertial', unit='m')
            forward = astropy.coordinates.SkyCoord(
                xp, yp, zp, representation_type='cartesian', obstime=date,
                frame='heliocentricinertial', unit='m', observer=observer)
            forward = forward.transform_to('helioprojective')
            
            # Set the reference pixel coordinates as the forward-direction
            image_wcs.wcs.crval = (
                forward.Tx.to(u.deg).value[0], forward.Ty.to(u.deg).value[0])
        else:
            # Set the reference pixel coordinates as 61 degrees HPC, which is
            # the center of WISPR's composite FOV (as detailed in the in-flight
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
    
    if output_quantity == 'flux':
        output_image = np.zeros((output_size_y, output_size_x))
    elif output_quantity in ('distance', 'dsun'):
        output_image = np.full((output_size_y, output_size_x), np.inf)
    
    # Pad these so we can compute a full gradient at each location
    x = np.arange(-1, output_image.shape[1] + 1)
    y = np.arange(-1, output_image.shape[0] + 1)
    
    do_interp = len(x) > 100 and len(y) > 100
    if do_interp:
        slice = np.s_[1::3]
    else:
        slice = np.s_[::1]
    xx, yy = np.meshgrid(x[slice], y[slice])
    # Compute the LOS direction of each pixel (as helioprojective coordinates)
    los = image_wcs.pixel_to_world(xx, yy).transform_to('helioprojective')
    Tx, Ty = los.Tx, los.Ty
    
    # Compute the (approximate) width and height of each pixel in degrees,
    # which we'll need later to estimate the filling factor of a pixel.
    # Do this in a way that handles coordinate wrap points
    spacing = 3 if do_interp else 1
    dTxdx2 = np.minimum(
        np.gradient(Tx.to(u.deg).value, spacing, axis=1)**2,
        np.gradient(Tx.to(u.deg).value % 360, spacing, axis=1)**2)
    dTxdy2 = np.minimum(
        np.gradient(Tx.to(u.deg).value, spacing, axis=0)**2,
        np.gradient(Tx.to(u.deg).value % 360, spacing, axis=0)**2)
    dTydx2 = np.minimum(
        np.gradient(Ty.to(u.deg).value, spacing, axis=1)**2,
        np.gradient(Ty.to(u.deg).value % 360, spacing, axis=1)**2)
    dTydy2 = np.minimum(
        np.gradient(Ty.to(u.deg).value, spacing, axis=0)**2,
        np.gradient(Ty.to(u.deg).value % 360, spacing, axis=0)**2)
    # Average of the width and height
    px_scale = 0.5 * (np.sqrt(dTxdx2 + dTydx2) + np.sqrt(dTxdy2 + dTydy2))
    
    # Turn the LOS directions into distant points and transform them to HCI
    # (x,y,z) points
    obstime = date
    sc_coord = astropy.coordinates.SkyCoord(
        sc.x, sc.y, sc.z, frame='heliocentricinertial',
        representation_type='cartesian', unit='m', obstime=obstime)
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
        # Undo the padding
        xx_full, yy_full = np.meshgrid(x[1:-1], y[1:-1])
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
        px_scale = scipy.interpolate.RegularGridInterpolator(
            (y[slice], x[slice]), px_scale, method='linear',
            bounds_error=False, fill_value=None)((yy_full, xx_full))
        x = x_new
    else:
        # Undo the padding
        x = x_old[1:-1, 1:-1]
        x_over_xdotx = x_over_xdotx[1:-1, 1:-1]
        px_scale = px_scale[1:-1, 1:-1]
    
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
    
    try:
        output_quantity_flag = {
            'flux': 1,
            'distance': 2,
            'dsun': 3,
            }[output_quantity]
    except KeyError:
        raise ValueError(
            f"Invalid value {output_quantity} for output_quantity")
    
    for i, parcel in enumerate(parcels):
        # Draw each parcel onto the output canvas
        parcel_pos = parcel_poses[i]
        d_p_sc = np.linalg.norm(parcel_pos - sc_pos)
        if (dmin is not None and d_p_sc < dmin
                or dmax is not None and d_p_sc > dmax):
            continue
        d_p_sun = np.linalg.norm(parcel_pos)
        if (dsunmin is not None and d_p_sun < dsunmin
                or dsunmax is not None and d_p_sun > dsunmax):
            continue
        if only_side_of_sun:
            component = np.dot(parcel_pos, sc_pos)
            if only_side_of_sun == 'near' and component < 0:
                continue
            elif only_side_of_sun == 'far' and component > 0:
                continue
                
        
        # Apply all brightness scalings that apply to the parcel as a whole (to
        # a good approximation)
        I_scale = 1
        # Scale for the parcel--Sun distance
        if scale_sun_dist:
            I_scale *= 1 / (d_p_sun / u.R_sun.to(u.m))**2
        
        scattering_angle = utils.angle_between_vectors(
            *(-parcel_pos),
            *(sc_pos - parcel_pos))[0]
        
        if use_density:
            # Intensity is proportional to density
            with parcel.at_temp(t0) as p:
                I_scale *= p.rho
        if expansion:
            # Solar wind parcels grow in volume as r^2
            # For our spherical parcels, that means radius grows as r^2/3
            radius_scale = (d_p_sun / (10 * u.R_sun.to(u.m))) ** (2/3)
            parcel_size = parcel_width * radius_scale
            
            with parcel.at_temp(t0) as p:
                # This is a blend of the LOS component perpendicular and
                # parallel to the parcel--Sun radial line (as those components
                # expand differently with radius)
                los = np.sqrt((p.rperp * np.sin(scattering_angle))**2
                            + (p.rpar * np.cos(scattering_angle))**2)
            I_scale *= los
        else:
            parcel_size = parcel_width
        
        if thomson:
            thomson_factor = 1 + np.cos(scattering_angle)**2
            I_scale *= thomson_factor
        
        if isinstance(I_scale, np.ndarray):
            # Make sure we have an honest-to-goodness number and not a numpy scalar.
            I_scale = I_scale.item()
        
        _synth_data_one_parcel(sc_pos, parcel_pos, x, x_over_xdotx,
                               px_scale, parcel_size, output_image,
                               d_p_sc, px[i], py[i], antialias,
                               output_quantity_flag, I_scale)
    
    if output_quantity == 'distance':
        output_image[np.isinf(output_image)] = np.nan
    
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
        # .item() calls extract the single element of a 1-element array, and
        # are a no-op for np scalars
        image_wcs.wcs.aux.hgln_obs = c.lon.to(u.deg).value.item()
        image_wcs.wcs.aux.hglt_obs = c.lat.to(u.deg).value.item()
        image_wcs.wcs.aux.dsun_obs = c.radius.to(u.m).value.item()
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
def _synth_data_one_parcel(sc_pos, parcel_pos, x, x_over_xdotx,
                           px_scale, parcel_width, output_image,
                           d_p_sc, start_x, start_y, antialias,
                           output_quantity_flag, I_scale):
    # Check if this time point is valid for this parcel
    if np.isnan(parcel_pos[0]):
        return
    
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
                    i, j, x, x_over_xdotx, px_scale[i,j],
                    parcel_pos, sc_pos, antialias,
                    parcel_width, d_p_sc, output_image,
                    output_quantity_flag, I_scale):
                # Flux was contributed
                n += 1
                continue
            # Flux was not contributed---we've reached the right-hand edge of
            # the parcel in this row
            break
        
        # Iterate left from the start point
        for j in range(start_x-1, -1, -1):
            if _synth_data_one_pixel(
                    i, j, x, x_over_xdotx, px_scale[i,j],
                    parcel_pos, sc_pos, antialias,
                    parcel_width, d_p_sc, output_image,
                    output_quantity_flag, I_scale):
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
                    i, j, x, x_over_xdotx, px_scale[i,j], 
                    parcel_pos, sc_pos, antialias,
                    parcel_width, d_p_sc, output_image,
                    output_quantity_flag, I_scale):
                # Flux was contributed
                n += 1
                continue
            # Flux was not contributed---we've reached the right-hand edge of
            # the parcel in this row
            break
        
        for j in range(start_x-1, -1, -1):
            if _synth_data_one_pixel(
                    i, j, x, x_over_xdotx, px_scale[i,j],
                    parcel_pos, sc_pos, antialias,
                    parcel_width, d_p_sc, output_image,
                    output_quantity_flag, I_scale):
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
def _synth_data_one_pixel(i, j, x, x_over_xdotx, px_scale,
                          parcel_pos, sc_pos, antialias,
                          parcel_width, d_p_sc, output_image,
                          output_quantity_flag, I_scale):
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
    if t <= 0 or t > 1:
        return False
    
    # Compute the actual coordinates of the closest-approach point
    # This is how we would do it, but let's not save intermediary values
    # closest_approach = t * x[i, j] + sc_pos
    
    # Compute how close the closest approach is to the parcel center
    d_p_close_app = np.sqrt(
              (t * x[i, j, 0] + sc_pos[0] - parcel_pos[0])**2
            + (t * x[i, j, 1] + sc_pos[1] - parcel_pos[1])**2
            + (t * x[i, j, 2] + sc_pos[2] - parcel_pos[2])**2
            )
    
    # For a distant parcel that's smaller than a pixel, as it passes from
    # close to LOS A, to in-between, to close to LOS B, the Gaussian
    # amplitude will oscillate down and up. We *should* be integrating the
    # 2D gaussian throughout the patch we're seeing in this pixel, but as
    # more hacky anti-aliasing, let's find the projected pixel width at the
    # parcel and search within that radius of the closest approach and use
    # the highest Gaussian value in that region.
    if antialias:
        pixel_size_at_parcel = d_p_sc * np.tan(px_scale / 2 * np.pi/180)
        d_p_close_app = max(0, d_p_close_app - pixel_size_at_parcel)
    
    # Truncate the gaussian
    if d_p_close_app > parcel_width:
        return False
    
    if output_quantity_flag == 1:
        flux = np.exp(-d_p_close_app**2 / (parcel_width/6)**2 / 2)
        
        if d_p_sc < parcel_width:
            # Ramp down the flux as the parcel gets really close to the
            # s/c, to avoid flashiness, etc. (In reality, once the sc enters
            # the parcel, it's integrating shorter columns, so the brightness
            # should drop off.)
            flux *= d_p_sc / parcel_width
        
        # We're just using this to adjust the scaling of the output values
        rsun = 695700000.0 # m
        
        # We have a 1/r^2 scaling for the parcel--SC distance, but a r^2
        # scaling for that distance because we're integrating a wider column
        # through the parcel the further away it is. But that caps out when the
        # full parcel fits in our column. Instead of doing a really careful
        # treatment, let's just add two r scalings, for width and height, but
        # cap them at the value that has the parcel fill the pixel width/height
        flux *= 1 / (d_p_sc / rsun)**2
        
        max_d_width = parcel_width / 2 / np.tan(px_scale / 2 * np.pi/180)
        
        flux *= (min(d_p_sc, max_d_width) / rsun) ** 2
        
        flux *= I_scale
        
        output_image[i, j] += flux
    elif output_quantity_flag == 2:
        output_image[i, j] = min(output_image[i, j], d_p_sc)
    elif output_quantity_flag == 3:
        output_image[i, j] = min(
            output_image[i, j], np.sqrt(np.sum(parcel_pos**2)))
    
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
            utils.angle_between_vectors(sc.vx, sc.vy, 0, -sc.x, -sc.y, 0))
    v_p = parcel.v
    dphi = np.atleast_1d(
            utils.angle_between_vectors(sc.x, sc.y, 0, parcel.x, parcel.y, 0))
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
    sc_direction = utils.angle_between_vectors(
            sc.vx, sc.vy, 0, -sc.x, -sc.y, 0)
    return elongation - sc_direction


@dataclasses.dataclass
class SimulationData:
    sc: Thing
    t: list
    parcels: list[Thing] = dataclasses.field(default_factory=list)
    encounter: int = None
    
    def plot_overhead(self, t0=None, mark_epsilon=False, mark_FOV_pos=False,
                      mark_FOV=False, fov_bins=[], mark_derot_ax=False,
                      detail=False, fovdat=None, ax=None, point_scale=1,
                      focus_sc=True):
        scale_factor = u.R_sun.to(u.m)
        if ax is None:
            ax = plt.gca()
        if t0 is None:
            t0 = np.mean(self.t)
        sc = self.sc.at(t0)
        
        ax.scatter(0, 0, c='yellow', s=100 * point_scale**2, zorder=18,
                   edgecolors='black')
        ax.scatter(sc.x, sc.y, zorder=10, s=100 * point_scale**2,
                   edgecolors='white')
        ax.plot(sc.at(self.t).x, sc.at(self.t).y, zorder=9, lw=5 * point_scale)

        pxs = []
        pys = []
        for parcel in self.parcels:
            with parcel.at_temp(self.t) as p:
                ax.plot(p.x, p.y, 'C1', alpha=.3, lw=1 * point_scale)
            with parcel.at_temp(t0) as p:
                pxs.append(p.x)
                pys.append(p.y)
        ax.scatter(pxs, pys, color='C1',
                   s=(36 if detail else 24) * point_scale**2)

        if mark_epsilon:
            x = np.nanmean(pxs)
            y = np.nanmean(pys)
            ax.plot([0, sc.x[0], x], [0, sc.y[0], y], color='gray')
        
        if mark_FOV_pos:
            x = np.nanmean(pxs)
            y = np.nanmean(pys)
            dx = sc.at(t0+.5).x - sc.x
            dy = sc.at(t0+.5).y - sc.y
            ax.plot([x, sc.x[0], sc.x[0] - dx[0]],
                    [y, sc.y[0], sc.y[0] + dy[0]], color='gray')

        if mark_derot_ax:
            length = (10 if detail else 30) * scale_factor
            ax.arrow(
                sc.x[0] - length/2, sc.y[0], length, 0, color='.7', zorder=18)
            ax.arrow(
                sc.x[0], sc.y[0]-length/2, 0, length, color='.7', zorder=18)

        if detail:
            half_window = 8 * (u.R_sun.to(u.m))
            ax.set_xlim(sc.x - half_window, sc.x + half_window)
            ax.set_ylim(sc.y - half_window, sc.y + half_window)
            ax.plot([0, sc.x[0]], [0, sc.y[0]], color='w', alpha=.5)
            t = np.arctan2(sc.y[0], sc.x[0])
            r = sc.r[0]
            rs = np.arange(0, r, u.R_sun.to(u.m))
            ax.scatter(
                rs * np.cos(t), rs * np.sin(t), color='white', s=25, alpha=.5)
        elif focus_sc:
            margin = 1
            xs = list(sc.at(self.t).x)
            xmin, xmax = np.nanmin(xs), np.nanmax(xs)
            xrange = xmax - xmin
            
            ys = list(sc.at(self.t).y)
            ymin, ymax = np.nanmin(ys), np.nanmax(ys)
            yrange = ymax - ymin
            
            if yrange == 0:
                yrange = xrange
            if xrange == 0:
                xrange = yrange
            
            tbounds = np.array([self.t[0], self.t[-1]])
            for parcel in self.parcels:
                with parcel.at_temp(tbounds) as p:
                    xs.extend(p.x)
                    ys.extend(p.y)
            
            xs = [x for x in xs
                  if xmin - margin * xrange <= x <= xmax + margin * xrange]
            ax.set_xlim(np.nanmin(xs)-1, np.nanmax(xs)+1)

            ys = [y for y in ys
                  if ymin - margin * yrange <= y <= ymax + margin * yrange]
            
            xmin, xmax = np.nanmax(xs), np.nanmin(xs)
            if np.nanmax(np.abs(ys)) < 2 * (xmax - xmin) and xmin < 0 < xmax:
                # Include the Sun in the plot y-range if it doesn't warp the
                # aspect ratio too much and if the Sun is already included in
                # the x plotting range
                ys.append(0)
            else:
                # Put "To Sun" arrows?
                pass

            ax.set_ylim(np.nanmin(ys)-1, np.nanmax(ys) + 1)
        else:
            ax.autoscale()
            ax.autoscale(False)

        # Label axes in R_sun without having to re-scale every coordinate
        formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: f"{x / u.R_sun.to(u.m):.0f}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        tick_spacing = 20 * u.R_sun.to(u.m)
        xmin, xmax = ax.get_xlim()
        while xmax - xmin < tick_spacing * 5:
            tick_spacing /= 2
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(tick_spacing))
        
        if mark_FOV:
            if isinstance(fovdat[0], np.ndarray):
                fovdat = [fovdat]
            for fov, ls in zip(fovdat, ['-', '--', '-.', ':']):
                x1, x2 = 0, fov[0].shape[1]
                y1 = fov[0].shape[0] / 2
                y2 = y1
                edge1 = fov[1].pixel_to_world(x1, y1)
                edge1 = edge1.transform_to('helioprojective')
                edge2 = fov[1].pixel_to_world(x2, y2)
                edge2 = edge2.transform_to('helioprojective')
                lon1 = edge1.Tx.to_value(u.deg)
                lon2 = edge2.Tx.to_value(u.deg)
                
                to_sun_x = -sc.x
                to_sun_y = -sc.y
                to_sun_theta = np.arctan2(to_sun_y, to_sun_x)
                t1 = to_sun_theta - lon1 * np.pi/180
                t2 = to_sun_theta - lon2 * np.pi/180
                
                size = (max(fov_bins) * scale_factor
                        if len(fov_bins) else tick_spacing)
                x1 = size * np.cos(t1) + sc.x
                x2 = size * np.cos(t2) + sc.x
                y1 = size * np.sin(t1) + sc.y
                y2 = size * np.sin(t2) + sc.y

                ax.plot([x1, sc.x, x2], [y1, sc.y, y2],
                        color='w', alpha=0.75, lw=2.5, zorder=18, ls=ls)
                if len(fov_bins):
                    rs = np.array(fov_bins) * scale_factor
                    ts = np.linspace(t1, t2, 50)
                    for r in rs:
                        binx = r * np.cos(ts) + sc.x
                        biny = r * np.sin(ts) + sc.y
                        ax.plot(binx, biny, color='w', alpha=0.75, lw=2.5,
                                zorder=18, ls=ls)

        ax.set_aspect('equal')
        ax.set_facecolor('black')
        ax.set_xlabel("X ($R_\odot$)")
        if not detail:
            ax.set_ylabel("Y ($R_\odot$)")
    
    def plot_and_synthesize(
            self, t0, include_overhead=True, include_overhead_detail=False,
            mark_epsilon=False, mark_FOV_pos=False, mark_FOV=False,
            mark_bins=False, mark_derot_ax=False, synthesize=True,
            radiants=False, vmin=0, vmax=None, parcel_width=1, synth_kw={},
            synth_fixed_fov=None, synth_celest_wcs=False, synth_wcs=None,
            output_quantity='flux', use_default_figsize=False, figsize=None,
            synth_colorbar=False, point_scale=1, focus_sc=True,
            as_column=False):
        sc = self.sc.at(t0)
        n_plots = include_overhead + include_overhead_detail + synthesize
        if use_default_figsize:
            figsize = None
        elif figsize is None:
            figsize = (7 * n_plots, 7)
        if as_column:
            fig, axs = plt.subplots(n_plots, 1, squeeze=False, figsize=figsize)
        else:
            fig, axs = plt.subplots(1, n_plots, squeeze=False, figsize=figsize)
        axs = list(axs.flatten())
        all_axes = axs[:]
        
        ax_overhead = axs.pop(0) if include_overhead else None
        ax_overhead_detail = axs.pop(0) if include_overhead_detail else None
        ax_syn = axs.pop(0) if synthesize else None

        # Even if we're not showing it, make an image so we can grab the FOV
        if synth_wcs is not None and synth_wcs.pixel_shape is not None:
            x, y = synth_wcs.pixel_shape
        else:
            x, y = 400, 250
        image, wcs = synthesize_image(
            self.sc, self.parcels, t0, image_wcs=synth_wcs,
            output_size_x=x, output_size_y=y if synthesize else 1,
            parcel_width=parcel_width, fixed_fov_range=synth_fixed_fov,
            celestial_wcs=synth_celest_wcs, output_quantity=output_quantity,
            projection='CAR' if synth_fixed_fov else 'ARC', **synth_kw)
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
            
            # Recreate axis w/ WCS projection (I wish there were a better way!)
            ax_syn.remove()
            ax_idx = all_axes.index(ax_syn)
            if as_column:
                ax_syn = fig.add_subplot(n_plots, 1, n_plots, projection=wcs)
            else:
                ax_syn = fig.add_subplot(1, n_plots, n_plots, projection=wcs)
            all_axes[ax_idx] = ax_syn
            if vmax is None:
                if output_quantity == 'dsun':
                    vmax = 2 * sc.at(t0).r / u.R_sun.to(u.m)
                else:
                    vmax = image.max()
                    if vmax == 0:
                        vmax = 1
            im = ax_syn.imshow(image, origin='lower',           aspect='equal', cmap=cmap,
                            norm=matplotlib.colors.PowerNorm(
                                gamma=gamma, vmin=vmin, vmax=vmax))
            
            lon, lat = ax_syn.coords
            lat.set_major_formatter('dd')
            lon.set_major_formatter('dd')
            if synth_celest_wcs:
                ax_syn.set_xlabel("Fixed Longitude")
            elif wcs.wcs.ctype[0].startswith("HP"):
                ax_syn.set_xlabel("HP Longitude")
            elif wcs.wcs.ctype[0].startswith("PS"):
                ax_syn.set_xlabel("PSP Frame Longitude")
            ax_syn.set_ylabel(" ")
            ax_syn.coords.grid(color='white', alpha=0.1)
            
            if radiants:
                for parcel in self.parcels:
                    radiant = calculate_radiant(self.sc, parcel, t0)
                    if radiant is None:
                        continue
                    x, y = wcs.world_to_pixel(radiant * u.rad, 0 * u.rad)
                    ax_syn.scatter(x, y, s=10, marker='x', color='green')
            if synth_colorbar:
                ax = [ax for ax in (ax_syn, ax_overhead, ax_overhead_detail)
                      if ax is not None]
                plt.colorbar(im, ax=ax)
        
        fov_bins = [20, 40, 60] if mark_bins else []
        if ax_overhead is not None:
            self.plot_overhead(
                ax=ax_overhead, t0=t0, mark_epsilon=mark_epsilon,
                mark_FOV_pos=mark_FOV_pos, mark_FOV=mark_FOV,
                mark_derot_ax=mark_derot_ax, fov_bins=fov_bins, detail=False,
                fovdat=(image, wcs), point_scale=point_scale,
                focus_sc=focus_sc)
        
        if ax_overhead_detail is not None:
            self.plot_overhead(
                ax=ax_overhead_detail, t0=t0, mark_epsilon=mark_epsilon,
                mark_FOV_pos=mark_FOV_pos, mark_FOV=mark_FOV,
                mark_derot_ax=mark_derot_ax, fov_bins=[], detail=True,
                fovdat=(image, wcs), point_scale=point_scale)
        return all_axes


def create_simdat_from_spice(E, nt=400):
    coords, times = planets.get_orbital_plane(
        'psp', E, npts=10000, return_times=True, expand_psp_orbit=False)
    f = coords.represent_as('spherical').distance < 0.25 * u.au
    coords = coords[f]
    times = times[f]
    x, y, z = (coords.x.to_value(u.m), coords.y.to_value(u.m),
               coords.z.to_value(u.m))
    sc = ArrayThing(times, x, y, z)
    t = np.linspace(times.min(), times.max(), nt)
    simdat = SimulationData(sc=sc, t=t, parcels=[], encounter=E)
    return simdat


def add_random_parcels(simdat, v=100_000, n_parcels=500, theta_dist=0):
    t_min = 120 * u.R_sun.to(u.m) / v
    
    for this_phi, this_theta, t_start in zip(
            np.random.uniform(0, 2*np.pi, n_parcels),
            np.random.uniform(-np.pi, np.pi, n_parcels) * theta_dist,
            np.random.uniform(simdat.t[0] - t_min, simdat.t[-1], n_parcels)):
        r = 1.1 * u.R_sun
        x = r * np.cos(this_phi) * np.cos(this_theta)
        y = r * np.sin(this_phi) * np.cos(this_theta)
        z = r * np.sin(this_theta)
        
        simdat.parcels.append(LinearThing(
            x=x.value, y=y.value, z=z.value,
            vx=(v * x / r).value, vy=(v * y / r).value, vz=(v * z / r).value,
            t=t_start, t_min=t_start))


def add_regular_near_impacts(simdat, v=100_000, hit_spacing=1, miss_t=1,
                             miss_t_before=None, miss_t_after=None):
    if miss_t_before is None:
        miss_t_before = miss_t
    if miss_t_after is None:
        miss_t_after = miss_t

    t_impacts = np.arange(simdat.t[0], simdat.t[-1], hit_spacing * 3600)
    for t_impact in t_impacts:
        with simdat.sc.at_temp(t_impact) as sc:
            if np.isnan(miss_t_before):
                dt = miss_t_after
            elif np.isnan(miss_t_after):
                dt = -miss_t_before
            else:
                dt = np.random.uniform(-miss_t_before, miss_t_after)
            dt *= 3600
            simdat.parcels.append(LinearThing(
                x=sc.x, y=sc.y, z=sc.z,
                vx=v*sc.x/sc.r, vy=v*sc.y/sc.r, vz=v*sc.z/sc.r,
                t=t_impact + dt, t_min=t_impact + dt - sc.r / v))


def add_random_near_impacts(simdat, v=100_000, n_parcels=40, miss_t=12*3600,
                            miss_t_before=None, miss_t_after=None,):
    if miss_t_before is None:
        miss_t_before = miss_t
    if miss_t_after is None:
        miss_t_after = miss_t

    for this_t, dt in zip(
            np.random.uniform(simdat.t.min(), simdat.t.max(), n_parcels),
            np.random.uniform(-miss_t_before, miss_t_after, n_parcels)):
        this_sc = simdat.sc.at(this_t)
        x, y, z = this_sc.x, this_sc.y, this_sc.z
        r = this_sc.r
        simdat.parcels.append(LinearThing(
            x=x, y=y, z=z,
            vx=v * x / r, vy=v * y / r, vz=v * z / r,
            t=this_t+dt, t_min=this_t + dt - r/v))


def clear_near_impacts(simdat, close_thresh):
    parcels = []
    for parcel in simdat.parcels:
        diff = simdat.sc - parcel
        dists = diff.at(simdat.t).r
        if np.nanmin(dists) * u.m > close_thresh * u.R_sun:
            parcels.append(parcel)
    simdat.parcels = parcels


def rad_var_v(r, V0=325*u.km/u.s, alpha=0.2):
    # From https://www.sciencedirect.com/science/article/pii/S027311771930746X
    # Higher alpha = steeper rise
    return V0 / (1 + np.exp(np.e - alpha * r / u.R_sun))


def gen_parcel_path(V0=325*u.km/u.s, alpha=0.2, end_point=250*u.R_sun):
    r_start = 1*u.R_sun
    t_start = 0*u.s
    dt = 10 * u.min
    
    rs = [r_start]
    ts = [t_start]
    
    r = r_start
    t = t_start
    while r < end_point:
        r = r + rad_var_v(r, V0, alpha) * dt
        t = t + dt
        rs.append(r)
        ts.append(t)
    rs = u.Quantity(rs).to(u.m)
    ts = u.Quantity(ts)
    vs = rad_var_v(rs, V0, alpha)

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

def add_random_parcels_rad_grad(simdat, n_parcels=500, V0=325*u.km/u.s,
                                alpha=0.2):
    t_min = 120 * u.R_sun.to(u.m) / 200_000
    rs, ts, vs, rhos, rperp, rpar = gen_parcel_path(V0, alpha)
    
    for this_theta, t_start in zip(
            np.random.uniform(0, 2*np.pi, n_parcels),
            np.random.uniform(simdat.t[0] - t_min, simdat.t[-1], n_parcels)):
        x = rs * np.cos(this_theta)
        y = rs * np.sin(this_theta)
        z = rs * 0
        tvals = t_start + ts.to_value(u.s)
        simdat.parcels.append(ArrayThing(
            tvals, x.value, y.value, z.value,
            t=t_start, rholist=rhos, rperplist=rperp, rparlist=rpar,
            t_min=tvals[0], t_max=tvals[-1]))


def add_regular_near_impacts_rad_grad(simdat, V0=325*u.km/u.s, alpha=0.2,
                                      hit_spacing=1, miss_t=1,
                                      miss_t_before=None, miss_t_after=None):
    rs, ts, vs, rhos, rperp, rpar = gen_parcel_path(V0, alpha)
    rs = rs.to_value(u.m)
    ts = ts.to_value(u.s)
    
    if miss_t_before is None:
        miss_t_before = miss_t
    if miss_t_after is None:
        miss_t_after = miss_t

    t_impacts = np.arange(simdat.t[0], simdat.t[-1], hit_spacing * 3600)
    for t_impact in t_impacts:
        with simdat.sc.at_temp(t_impact) as sc_impact:
            if np.isnan(miss_t_before):
                dt = miss_t_after
            elif np.isnan(miss_t_after):
                dt = -miss_t_before
            else:
                dt = np.random.uniform(-miss_t_before, miss_t_after)
            dt *= 3600
            
            r_impact = sc_impact.r
            target_t = np.interp(r_impact, rs, ts)
            these_ts = ts + (t_impact - target_t) + dt
            
            simdat.parcels.append(ArrayThing(
                these_ts, rs * sc_impact.x / r_impact,
                rs * sc_impact.y / r_impact, rs * sc_impact.z / r_impact,
                t=these_ts[0], rholist=rhos, rperplist=rperp, rparlist=rpar,
                t_min=these_ts[0], t_max=these_ts[-1]))


def rotate_parcels_into_orbital_plane(simdat):
    obstime = utils.from_timestamp(simdat.t[0])
    for parcel in simdat.parcels:
        if isinstance(parcel, LinearThing):
            c = astropy.coordinates.SkyCoord(
                x=parcel.x_t0, y=parcel.y_t0, z=parcel.z_t0,
                v_x=parcel.vx_t0/u.s, v_y=parcel.vy_t0/u.s, v_z=parcel.vz_t0/u.s,
                obstime=obstime, frame='psporbitalframe',
                representation_type='cartesian')
            c = c.transform_to('heliocentricinertial').cartesian
            parcel.x_t0 = c.x.value
            parcel.y_t0 = c.y.value
            parcel.z_t0 = c.z.value
            parcel.vx_t0 = c.differentials['s'].d_x.value
            parcel.vy_t0 = c.differentials['s'].d_y.value
            parcel.vz_t0 = c.differentials['s'].d_z.value
        elif isinstance(parcel, ArrayThing):
            c = astropy.coordinates.SkyCoord(
                x=parcel.xlist, y=parcel.ylist, z=parcel.zlist,
                obstime=obstime, frame='psporbitalframe',
                representation_type='cartesian')
            c = c.transform_to('heliocentricinertial').cartesian
            parcel.xlist = c.x.value
            parcel.ylist = c.y.value
            parcel.zlist = c.z.value
            