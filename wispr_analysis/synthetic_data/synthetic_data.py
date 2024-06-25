from contextlib import ExitStack, contextmanager
import copy
import dataclasses

import astropy.coordinates
import astropy.time
import astropy.units as u
from astropy.visualization import quantity_support
from astropy.wcs import WCS
import matplotlib.ticker
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
import sunpy.coordinates

from .. import orbital_frame, planets, utils


quantity_support()


class Thing:
    """Represents a generic object with a position and a velocity in 2-space
    
    Subclasses have ``x``, ``y``, ``vx``, and ``vy`` attributes which provide
    those quantities. Instances can be set to a specific time, which then
    determines how those attributes are computed
    """
    t: u.Quantity = 0 * u.s
    t_min: u.Quantity = None
    t_max: u.Quantity = None
    
    def strip_units(self):
        """
        Returns a copy of this object with all values converted from
        Quantities to numpy objects in SI units.
        """
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
                # Maintain units
                return np.nan * quantity
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
        """ Returns True if the two objects are set to the same time 
        
        Parameters
        ----------
        other : `Thing`
            Another `Thing` to compare to
        """
        return np.all(self.t == other.t)

    @u.quantity_input
    def at(self, t: u.s):
        """ Returns a copy of this object at time ``t``. """
        t = np.atleast_1d(t)
        out = self.copy()
        out.set_t(t)
        return out
    
    @contextmanager
    @u.quantity_input
    def at_temp(self, t: (u.s, None)):
        """Temporarily sets this objects time to ``t``, without copying"""
        if t is None:
            yield self
            return
        old_t = self.t
        self.set_t(t)
        yield self
        self.t = old_t

    @u.quantity_input
    def set_t(self, t: u.s):
        """ Sets the object's time to ``t``. """
        t = np.atleast_1d(t)
        self.t = t

    @u.quantity_input(t=(u.s, None))
    def in_front_of(self, other, t=None):
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
            in_front = np.atleast_1d(np.abs(angle) < np.pi/2 * u.rad)
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
    x_t0: u.Quantity
    """The x position at t=0"""
    y_t0: u.Quantity
    """The y position at t=0"""
    z_t0: u.Quantity
    """The z position at t=0"""
    vx_t0: u.Quantity
    """The constant x velocity"""
    vy_t0: u.Quantity
    """The constant y velocity"""
    vz_t0: u.Quantity
    """The constant z velocity"""
    rperp0: float = 1
    rpar0: float = 1
    rho0: float = 1
    rperp_r2: bool = False
    density_r2: bool = False

    @u.quantity_input(x=u.m, y=u.m, z=u.m,
                 vx=u.m/u.s, vy=u.m/u.s, vz=u.m/u.s,
                 t=u.s, t_min=(u.s, None), t_max=(u.s, None))
    def __init__(self, x=0*u.m, y=0*u.m, z=0*u.m,
                 vx=0*u.m/u.s, vy=0*u.m/u.s, vz=0*u.m/u.s,
                 t=0*u.s, t_min=None, t_max=None,
                 rperp=1, rpar=1, rho=1, density_r2=False, rperp_r2=False):
        """
        Accepts physical parameters, as well as the corresponding time
        
        Parameters
        ----------
        x, y, z : ``float`` or ``Quantity``
            The position of the object at the reference time
        vx, vy, vz : ``float`` or ``Quantity``
            The velocity of the object at the reference time
        t : ``float`` or ``Quantity``
            The reference time
        t_min, t_max : ``float`` or ``Quantity``
            Can be set to limit the time range over which this parcel is valid.
            For times outside this range, positions and velocities will be
            computed as ``nan``
        rperp, rpar : ``float`` or ``Quantity``
            The radius of the parcel at the reference time measured
            perpendicular to and parallel to the parcel--Sun line.
        rho : ``float`` or ``Quantity``
            The density of the parcel at the reference time
        density_r2 : ``bool``
            Whether ``rho`` should be scaled by 1/r^2, with r being the object's
            distance from the Sun
        rperp_r2 : ``bool``
            Whether ``rperp`` should be scaled by 1/r^2, with r being the object's
            distance from the Sun
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
        """ The object's ``x`` position at the currently-set time"""
        x = self.x_t0 + self.vx_t0 * self.t
        x = self.process_t_bounds(x)
        return x
    
    @property
    def y(self):
        """ The object's ``y`` position at the currently-set time"""
        y = self.y_t0 + self.vy_t0 * self.t
        y = self.process_t_bounds(y)
        return y
    
    @property
    def z(self):
        """ The object's ``z`` position at the currently-set time"""
        z = self.z_t0 + self.vz_t0 * self.t
        z = self.process_t_bounds(z)
        return z
    
    @property
    def vx(self):
        """ The object's ``x`` velocity, which is constant"""
        vx = self.process_t_bounds(self.vx_t0)
        return vx
    
    @vx.setter
    @u.quantity_input
    def vx(self, value: u.m/u.s):
        self.vx_t0 = value
    
    @property
    def vy(self):
        """ The object's ``y`` velocity, which is constant"""
        vy = self.process_t_bounds(self.vy_t0)
        return vy
    
    @vy.setter
    @u.quantity_input
    def vy(self, value: u.m/u.s):
        self.vy_t0 = value
    
    @property
    def vz(self):
        """ The object's ``z`` velocity, which is constant"""
        vz = self.process_t_bounds(self.vz_t0)
        return vz
    
    @vz.setter
    @u.quantity_input
    def vz(self, value: u.m/u.s):
        self.vz_t0 = value
    
    @property
    def rperp(self):
        rperp = self.rperp0
        if self.rperp_r2:
            scale = 1 / self.r**2
            rperp *= scale.si.value
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
            scale = 1 / self.r**2
            rho *= scale.si.value
        return rho
    
    @rho.setter
    def rho(self, value):
        self.rho0 = value

    @u.quantity_input
    def offset_by_time(self, dt: u.s):
        """
        Creates a copy of this object that represents, at the current time,
        where this object will be ``dt`` into the future
        """
        out = self.copy()
        out.x_t0 += out.vx_t0 * dt
        out.y_t0 += out.vy_t0 * dt
        out.z_t0 += out.vz_t0 * dt
        return out

    def strip_units(self):
        out = super().strip_units()
        for attr in ('x_t0', 'y_t0', 'z_t0', 'vx_t0', 'vy_t0', 'vz_t0',
                     'rperp0', 'rpar0', 'rho0'):
            value = getattr(out, attr)
            if isinstance(value, u.Quantity):
                setattr(out, attr, value.si.value)
        return out


@dataclasses.dataclass
class ArrayThing(Thing):
    """
    Represents an object whose position over time is specified numerically.
    
    Positions are provided at a number of points in time, and those positions
    are interpolated between as needed.
    """
    
    xlist: u.Quantity
    """The array of x points"""
    ylist: u.Quantity
    """The array of y points"""
    zlist: u.Quantity
    """The array of z points"""
    tlist: u.Quantity
    """The array of times"""
    rperplist: np.ndarray = 0
    rparlist: np.ndarray = 0
    rholist: np.ndarray = 0
    scale_rho_r2: bool = False

    @u.quantity_input(tlist=u.s, xlist=u.m, ylist=u.m, zlist=u.m,
                 t=0*u.s, t_min=(u.s, None), t_max=(u.s, None))
    def __init__(self, tlist, xlist=0*u.m, ylist=0*u.m, zlist=0*u.m,
            t=0*u.s, t_min=None, t_max=None, rperplist=1, rparlist=1,
            rholist=1, default_density_r2=False):
        """
        Parameters
        ----------
        tlist : ``np.ndarray`` or ``Quantity``
            The list of time points at which positions are provided
        xlist, ylist, zlist : ``np.ndarray`` or float or ``Quantity``
            The specified positions. If either is a single number, that number
            is used at all points in time.
        t : float or ``Quantity``
            The time the object is currently at.
        t_min, t_max : ``float`` or ``Quantity``
            Can be set to limit the time range over which this parcel is valid.
            For times outside this range, positions and velocities will be
            computed as ``nan``
        rperplist, rparlist : ``float`` or ``np.ndarray`` or ``Quantity``
            The radius of the parcel at each point in time, measured
            perpendicular to and parallel to the parcel--Sun line.
        rholist: ``float`` or ``np.ndarray`` or ``Quantity``
            The density of the parcel at each point in time
        default_density_r2 : ``bool``
            If ``rholist`` is not provided, this controls whether the default
            density should scale with 1/r^2
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
        """ The object's ``x`` position at the currently-set time"""
        return self._access_quantity(self.xlist)
    
    @property
    def y(self):
        """ The object's ``y`` position at the currently-set time"""
        return self._access_quantity(self.ylist)
    
    @property
    def z(self):
        """ The object's ``z`` position at the currently-set time"""
        return self._access_quantity(self.zlist)
    
    @property
    def vx(self):
        """ The object's ``x`` velocity at the currently-set time"""
        dt = .0001
        if isinstance(self.t, u.Quantity):
            dt *= u.s
        vx = self._finite_difference(self.xlist, dt)
        return vx
    
    @property
    def vy(self):
        """ The object's ``y`` velocity at the currently-set time"""
        dt = .0001
        if isinstance(self.t, u.Quantity):
            dt *= u.s
        vy = self._finite_difference(self.ylist, dt)
        return vy
    
    @property
    def vz(self):
        """ The object's ``z`` velocity at the currently-set time"""
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
            scale = 1 / self.r**2
            values *= scale.si.value
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
        """
        Creates a copy of this object that represents, at the current time,
        where this object will be ``dt`` into the future
        """
        out = self.copy()
        out.tlist += dt
        return out

    def strip_units(self):
        out = super().strip_units()
        for attr in ('xlist', 'ylist', 'zlist', 'tlist', 'rperplist',
                     'rparlist', 'rholist'):
            value = getattr(out, attr)
            if isinstance(value, u.Quantity):
                setattr(out, attr, value.si.value)
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
        dt = .0001 * u.s
        with (self.at_temp(self.t - dt/2)):
            before = self.x
        with (self.at_temp(self.t + dt/2)):
            after = self.x
        return (after - before) / dt
    
    @property
    def vy(self):
        dt = .0001 * u.s
        with (self.at_temp(self.t - dt/2)):
            before = self.y
        with (self.at_temp(self.t + dt/2)):
            after = self.y
        return (after - before) / dt
    
    @property
    def vz(self):
        dt = .0001 * u.s
        with (self.at_temp(self.t - dt/2)):
            before = self.z
        with (self.at_temp(self.t + dt/2)):
            after = self.z
        return (after - before) / dt


def calc_hpc(sc: "Thing", parcels: list["Thing"], t=None):
    """Computes object(s)' HPC coordinates as viewed by a given spacecraft

    Parameters
    ----------
    sc : `Thing`
        The observer
    parcels : `Thing` or ``List[Thing]``
        The object(s) being observed

    Returns
    -------
    Tx, Ty : ``Quantity``
        The HPC coordinate(s) of the object(s)
    """
    was_not_list = False
    if isinstance(parcels, Thing):
        parcels = [parcels]
        was_not_list = True
    
    with ExitStack() as stack:
        sc = stack.enter_context(sc.at_temp(t))
        parcels = [stack.enter_context(p.at_temp(t)) for p in parcels]
    
        px = u.Quantity([p.x for p in parcels])
        py = u.Quantity([p.y for p in parcels])
        pz = u.Quantity([p.z for p in parcels])
        
        Tx, Ty = xyz_to_hpc(px, py, pz, sc.x, sc.y, sc.z)
        if was_not_list:
            Tx = Tx[0]
            Ty = Ty[0]
        return Tx, Ty
    

def xyz_to_hpc(xs, ys, zs, scx, scy, scz):
    """Computes object(s)' HPC coordinates as viewed by a given spacecraft

    Parameters
    ----------
    xs, ys, zs : ``float`` or ``np.ndarray`` or ``Quantity``
        The position(s) of the object(s) being observed
    scx, scy, scz : ``float`` or ``np.ndarray`` or ``Quantity``
        The position of the observer

    Returns
    -------
    Tx, Ty : ``Quantity``
        The HPC coordinate(s) of the object(s)
    """
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
    
    Tx = p_hpc.Tx.to(u.deg)
    Ty = p_hpc.Ty.to(u.deg)
    
    return Tx, Ty


def hpc_to_elpa(Tx, Ty):
    """Converts HPC coordinates to elongation and position angle

    Parameters
    ----------
    Tx, Ty : ``float`` or ``np.ndarray`` or ``Quantity``
        The input HPC coordinates

    Returns
    -------
    elongation, pa : ``float`` or ``np.ndarray`` or ``Quantity``
        The output elongation and position angle
    """
    if not isinstance(Tx, u.Quantity):
        Tx = np.deg2rad(Tx)
    if not isinstance(Ty, u.Quantity):
        Ty = np.deg2rad(Ty)
    
    elongation = np.arctan2(
        np.sqrt(np.cos(Ty)**2 * np.sin(Tx)**2 + np.sin(Ty)**2),
        np.cos(Ty) * np.cos(Tx)
    )
    pa = np.arctan2(
         -np.cos(Ty) * np.sin(Tx),
         np.sin(Ty)
    )
    
    if not isinstance(elongation, u.Quantity):
        elongation = np.rad2deg(elongation)
    if not isinstance(pa, u.Quantity):
        pa = np.rad2deg(pa)
    return elongation, pa


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
    fov : ``int``, optional
        Camera field-of-view width
    projection : ``str``, optional
        Projection to use
    output_size_x, output_size_y : ``int``, optional
        Size of the generated image
    parcel_width : ``int`` or ``Quantity``, optional
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
    dsunmin, dsunmax : ``float``
        If given, only render parcels when their distance to the Sun is
        within this range.
    only_side_of_sun : ``str``
        Can be set to 'near' or 'far', in which case only parcels on the near or
        far side of the Sun will be rendered.
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
    scale_sun_dist : ``bool``
        Whether to scale parcels' brightness inversely with the square of their
        distance from the Sun.

    Returns
    -------
    output_image : ``np.ndarray``
        The output image
    image_wcs : ``WCS``
        The corresponding WCS
    """
    output_quantity = output_quantity.lower()
    with sc.at_temp(t0) as sc:
        sc = sc.strip_units()
    # Copy the list
    parcels = parcels[:]
    for i in range(len(parcels)):
        with parcels[i].at_temp(t0) as p:
            parcels[i] = p.strip_units()
    date = astropy.time.Time(t0, format='unix').fits
    if isinstance(parcel_width, u.Quantity):
        parcel_width = parcel_width.to(u.m).value
    else:
        parcel_width *= u.R_sun.to(u.m)
    if isinstance(dmin, u.Quantity):
        dmin = dmin.si.value
    if isinstance(dmax, u.Quantity):
        dmax = dmax.si.value
    if isinstance(t0, u.Quantity):
        t0 = t0.si.value
    
    # Build output image WCS
    if image_wcs is None:
        image_wcs = WCS(naxis=2)
        if point_forward:
            x, y, z = sc.x, sc.y, sc.z
            sc_soon = sc.copy()
            sc_soon.t += 1
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
    
    # If there are a lot of pixels, then when calculating the lines of sight of
    # each pixel, we'll calculate on a grid and interpolate (this affects only
    # figuring out where each line of sight is pointed---calculating the flux on
    # each LOS is still done individually for each pixel)
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
        parcel_poses[i] = parcel.x[0], parcel.y[0], parcel.z[0]
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
        parcel_pos = parcel_poses[i]
        # Draw each parcel onto the output canvas
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
            I_scale *= parcel.rho
        if expansion:
            # Solar wind parcels grow in volume as r^2
            # For our spherical parcels, that means radius grows as r^2/3
            radius_scale = (d_p_sun / (10 * u.R_sun.to(u.m))) ** (2/3)
            parcel_size = parcel_width * radius_scale

            # This is a blend of the LOS component perpendicular and
            # parallel to the parcel--Sun radial line (as those components
            # expand differently with radius)
            los = np.sqrt((parcel.rperp * np.sin(scattering_angle))**2
                        + (parcel.rpar * np.cos(scattering_angle))**2)
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


def calculate_radiant(sc, parcel, t0=0*u.s):
    t0 = np.atleast_1d(t0)
    if len(t0) > 1 or t0 != 0:
        sc = sc.at(t0)
        parcel = parcel.at(t0)
    infront = np.atleast_1d(parcel.in_front_of(sc))
    if not np.any(infront):
        return np.full(max(len(t0), len(np.atleast_1d(parcel.x))), np.nan) * u.deg
    v_sc = sc.v
    e_sc = np.atleast_1d(
            utils.angle_between_vectors(
                sc.vx, sc.vy, 0*u.m/u.s, -sc.x, -sc.y, 0*u.m))
    v_p = parcel.v
    dphi = np.atleast_1d(
            utils.angle_between_vectors(
                sc.x, sc.y, 0*u.m, parcel.x, parcel.y, 0*u.m))
    epsilons = np.linspace(0, np.pi, 300)[None, :] << u.rad
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
            sc.vx, sc.vy, 0*u.km/u.s, -sc.x, -sc.y, 0*u.m)
    return elongation - sc_direction


@dataclasses.dataclass
class SimulationData:
    """
    Container for a spacecraft and a list of parcels
    """
    sc: Thing
    """
    The spacecraft, which sets the camera location
    """
    t: u.Quantity
    """A list of time points for this scenario"""
    parcels: list[Thing] = dataclasses.field(default_factory=list)
    """A list of plasma parcels"""
    encounter: int = None
    """If set, the PSP encounter this scenario models"""
    
    def plot_overhead(self, t0=None, mark_epsilon=False, mark_FOV_pos=False,
                      mark_FOV=False, fov_bins=[], mark_derot_ax=False,
                      detail=False, fovdat=None, ax=None, point_scale=1,
                      focus_sc=True):
        """Produces a top-down view of this scenario

        Parameters
        ----------
        t0 : ``float`` or ``Quantity``, optional
            The time to depict. If not given, the average time is used
        mark_epsilon : ``bool``, optional
            Draws lines marking the elongation of the average position of the parcels
        mark_FOV_pos : ``bool``, optional
            Draws lines marking the FOV position angle of the average position
            of the parcels
        mark_FOV : ``bool``, optional
            Draws lines showing the extent of the camera FOV. Requires
            ``fovdat`` to be provided.
        fov_bins : ``list[Quantity]``, optional
            A list of distances from the spacecraft. If given, the marked FOV
            will be divided into regions at these distances.
        mark_derot_ax : ``bool``, optional
            Overlays reference axes for the derotated frame
        detail : ``bool``, optional
            Renders in a mode zoomed in on the spacecraft
        fovdat : ``tuple`` or ``list[tuple]``, optional
            A tuple of a synthesized image and its WCS, for determining the
            camera FOV. Can also be a list of tuples, to mark multiple FOVs.
        ax : ``Axes``, optional
            The Axes on which to plot
        point_scale : ``float``, optional
            Scale factor for the sizes of points
        focus_sc : ``bool``, optional
            Whether the FOV should be set by the extent of the spacecraft orbit.
            If False, the FOV instead includes all parcel trajectories as well.
        """
        if ax is None:
            ax = plt.gca()
        if t0 is None:
            t0 = np.mean(self.t)
        sc = self.sc.at(t0)
        
        ax.scatter(0*u.R_sun, 0*u.R_sun, c='yellow', s=100 * point_scale**2, zorder=18,
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
        pxs = u.Quantity(pxs)
        pys = u.Quantity(pys)
        ax.scatter(pxs, pys, color='C1',
                   s=(36 if detail else 24) * point_scale**2)

        if mark_epsilon:
            x = np.nanmean(pxs)
            y = np.nanmean(pys)
            ax.plot(u.Quantity([0*u.m, sc.x[0], x]), u.Quantity([0*u.m, sc.y[0], y]), color='gray')
        
        if mark_FOV_pos:
            x = np.nanmean(pxs)
            y = np.nanmean(pys)
            dx = sc.at(t0+.5*u.s).x - sc.x
            dy = sc.at(t0+.5*u.s).y - sc.y
            ax.plot(u.Quantity([x, sc.x[0], sc.x[0] - dx[0]]),
                    u.Quantity(  [y, sc.y[0], sc.y[0] + dy[0]]), color='gray')

        if mark_derot_ax:
            length = (10 if detail else 30) * u.R_sun
            ax.arrow(
                sc.x[0] - length/2, sc.y[0], length, 0, color='.7', zorder=18)
            ax.arrow(
                sc.x[0], sc.y[0]-length/2, 0, length, color='.7', zorder=18)

        if detail:
            half_window = 8 * u.R_sun
            ax.set_xlim(sc.x - half_window, sc.x + half_window)
            ax.set_ylim(sc.y - half_window, sc.y + half_window)
            ax.plot(u.Quantity([0*u.m, sc.x[0]]), u.Quantity([0*u.m, sc.y[0]]), color='w', alpha=.5)
            t = np.arctan2(sc.y[0], sc.x[0])
            r = sc.r[0]
            rs = _arange_from_units(0*u.R_sun, r.to(u.R_sun), 1*u.R_sun)
            ax.scatter(
                rs * np.cos(t), rs * np.sin(t), color='white', s=25, alpha=.5)
        elif focus_sc:
            margin = 1
            xs = u.Quantity(sc.at(self.t).x)
            xmin, xmax = np.nanmin(xs), np.nanmax(xs)
            xrange = xmax - xmin
            
            ys = u.Quantity(sc.at(self.t).y)
            ymin, ymax = np.nanmin(ys), np.nanmax(ys)
            yrange = ymax - ymin
            
            if yrange == 0:
                yrange = xrange
            if xrange == 0:
                xrange = yrange
            
            tbounds = u.Quantity([self.t[0], self.t[-1]])
            for parcel in self.parcels:
                with parcel.at_temp(tbounds) as p:
                    xs = np.concatenate((xs, p.x))
                    ys = np.concatenate((ys, p.y))
            
            xs = u.Quantity([x for x in xs
                             if xmin - margin * xrange <= x <= xmax + margin * xrange])
            ax.set_xlim(np.nanmin(xs)-1*u.R_sun, np.nanmax(xs)+1*u.R_sun)

            ys = u.Quantity([y for y in ys
                             if ymin - margin * yrange <= y <= ymax + margin * yrange])
            
            xmin, xmax = np.nanmax(xs), np.nanmin(xs)
            if np.nanmax(np.abs(ys)) < 2 * (xmax - xmin) and xmin < 0 < xmax:
                # Include the Sun in the plot y-range if it doesn't warp the
                # aspect ratio too much and if the Sun is already included in
                # the x plotting range
                ys.append(0*u.m)
            else:
                # Put "To Sun" arrows?
                pass

            ax.set_ylim(np.nanmin(ys)-1*u.R_sun, np.nanmax(ys) + 1*u.R_sun)
        else:
            ax.autoscale()
            ax.autoscale(False)
        
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
                lon1 = edge1.Tx
                lon2 = edge2.Tx
                
                to_sun_x = -sc.x
                to_sun_y = -sc.y
                to_sun_theta = np.arctan2(to_sun_y, to_sun_x)
                t1 = to_sun_theta - lon1
                t2 = to_sun_theta - lon2

                size = (max(fov_bins) if len(fov_bins)
                        else np.ptp(plt.xlim()) / 8 * u.R_sun)
                x1 = size * np.cos(t1) + sc.x
                x2 = size * np.cos(t2) + sc.x
                y1 = size * np.sin(t1) + sc.y
                y2 = size * np.sin(t2) + sc.y

                ax.plot(u.Quantity([x1, sc.x, x2]), u.Quantity([y1, sc.y, y2]),
                        color='w', alpha=0.75, lw=2.5, zorder=18, ls=ls)
                if len(fov_bins):
                    rs = u.Quantity(fov_bins)
                    ts = np.linspace(t1, t2, 50)
                    for r in rs:
                        binx = r * np.cos(ts) + sc.x
                        biny = r * np.sin(ts) + sc.y
                        ax.plot(binx, biny, color='w', alpha=0.75, lw=2.5,
                                zorder=18, ls=ls)

        ax.set_aspect('equal')
        ax.set_facecolor('black')
        ax.set_xlabel(r"X ($R_\odot$)")
        if not detail:
            ax.set_ylabel(r"Y ($R_\odot$)")
    
    def plot_and_synthesize(
            self, t0, include_overhead=True, include_overhead_detail=False,
            synthesize=True, vmin=0, vmax=None, parcel_width=1*u.R_sun, synth_kw={},
            synth_fixed_fov=None, synth_celest_wcs=False, synth_wcs=None,
            output_quantity='flux', use_default_figsize=False, figsize=None,
            synth_colorbar=False, focus_sc=True,
            as_column=False, **kwargs):
        """Helper to plot overhead view and synthesized image size-by-size

        Parameters
        ----------
        t0 : ``Quantity``
            The time at which to plot
        include_overhead : ``bool``, optional
            Whether to include the overhead view
        include_overhead_detail : ``bool``, optional
            Whether to include a zoomed-in overhead view
        synthesize : ``bool``, optional
            Whether to include a synthetic image
        vmin, vmax : ``int``, optional
            The colorbar range to use for the synthetic image
        parcel_width : ``Quantity``, optional
            The width to use for parcels in the synthetic image
        synth_kw : ``dict``, optional
            A dictionary of arguments to pass to `synthesize_image`
        synth_fixed_fov : ``tuple``, optional
            If set, the synthetic image is in a fixed direction, with the
            elements of this tuple setting the starting and stopping latitude.
        synth_celest_wcs : ``bool``, optional
            Whether to synthesize onto a celestial WCS instead of
            helioprojective
        synth_wcs : ``WCS``, optional
            The WCS to use when rendering the synthetic image
        output_quantity : ``str``, optional
            The quantity to render in the synthetic image
        use_default_figsize : ``bool``, optional
            Whether to use the default figure size of the current Matplotlib
            style
        figsize : ``tuple``, optional
            The figure size to use
        synth_colorbar : ``bool``, optional
            Whether to show a colorbar for the synthetic image
        as_column : ``bool``, optional
            Whether to arrange the plots in a column instead of a row

        Returns
        -------
        axes : ``List[Axes]``
            The axes which were plotted on
        """
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
            im = ax_syn.imshow(image, origin='lower',
                               aspect='equal', cmap=cmap,
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
            
            if synth_colorbar:
                ax = [ax for ax in (ax_syn, ax_overhead, ax_overhead_detail)
                      if ax is not None]
                plt.colorbar(im, ax=ax)

        if ax_overhead is not None:
            self.plot_overhead(
                ax=ax_overhead, t0=t0, detail=False,
                fovdat=(image, wcs), focus_sc=focus_sc, **kwargs)
        
        if ax_overhead_detail is not None:
            if 'focus_sc' in kwargs:
                del kwargs['focus_sc']
            self.plot_overhead(
                ax=ax_overhead_detail, t0=t0, detail=True,
                fovdat=(image, wcs), **kwargs)
        return all_axes


def create_simdat_from_spice(E, nt=400):
    """Creates a `SimulationData` containing a spacecraft moving along a real
    PSP trajectory, calculated from SPICE

    Parameters
    ----------
    E : ``int`` or ``str``
        The PSP encounter to reproduce
    nt : ``int``, optional
        The number of points in time at which to compute the trajectory

    Returns
    -------
    simdat : `SimulationData`
        A `SimulationData` containing a spacecraft object and the corresponding
        points in time
    """
    coords, times = planets.get_orbital_plane(
        'psp', E, npts=10000, return_times=True, expand_psp_orbit=False)
    f = coords.represent_as('spherical').distance < 0.25 * u.au
    coords = coords[f]
    times = times[f]
    sc = ArrayThing(times << u.s, coords.x, coords.y, coords.z)
    t = np.linspace(times.min(), times.max(), nt) << u.s
    simdat = SimulationData(sc=sc, t=t, parcels=[], encounter=E)
    return simdat


def _random_from_units(low, high, n=1):
    return np.random.uniform(low.si.value, high.si.value, n) << low.si.unit


def _arange_from_units(start, stop, step=1):
    return np.arange(start.si.value, stop.si.value, step.si.value) << start.si.unit


def add_random_parcels(simdat, v=100*u.km/u.s, n_parcels=500, theta_dist=0):
    """Adds constant-velocity parcels with random trajectories and launch times

    Parameters
    ----------
    simdat : `SimulationData`
        The SimulationData to add parcels to
    v : ``Quantity``, optional
        The velocity of the parcels
    n_parcels : ``int``, optional
        The number of parcels to add
    theta_dist : ``float``, optional
        A scale factor from 0 to 1 controlling how far out of the HCI equatorial
        plane the parcels can point
    """
    t_min = 120 * u.R_sun / v
    
    for this_phi, this_theta, t_start in zip(
            np.random.uniform(0, 2*np.pi, n_parcels) << u.rad,
            np.random.uniform(-np.pi, np.pi, n_parcels) * theta_dist * u.rad,
            _random_from_units(simdat.t[0] - t_min, simdat.t[-1], n_parcels)):
        r = 1.1 * u.R_sun
        x = r * np.cos(this_phi) * np.cos(this_theta)
        y = r * np.sin(this_phi) * np.cos(this_theta)
        z = r * np.sin(this_theta)
        
        simdat.parcels.append(LinearThing(
            x=x, y=y, z=z,
            vx=(v * x / r), vy=(v * y / r), vz=(v * z / r),
            t=t_start, t_min=t_start))


def add_regular_near_impacts(simdat, v=100*u.km/u.s, hit_spacing=1,
                             miss_t=1*u.hr, miss_t_before=None,
                             miss_t_after=None):
    if miss_t_before is None:
        miss_t_before = miss_t
    if miss_t_after is None:
        miss_t_after = miss_t

    t_impacts = _arange_from_units(simdat.t[0], simdat.t[-1], hit_spacing)
    for t_impact in t_impacts:
        with simdat.sc.at_temp(t_impact) as sc:
            if np.isnan(miss_t_before):
                dt = miss_t_after
            elif np.isnan(miss_t_after):
                dt = -miss_t_before
            else:
                dt = _random_from_units(-miss_t_before, miss_t_after)
            simdat.parcels.append(LinearThing(
                x=sc.x, y=sc.y, z=sc.z,
                vx=v*sc.x/sc.r, vy=v*sc.y/sc.r, vz=v*sc.z/sc.r,
                t=t_impact + dt, t_min=t_impact + dt - sc.r / v))


def add_random_near_impacts(simdat, v=100*u.km/u.s, n_parcels=40,
                            miss_t=12*u.hr, miss_t_before=None,
                            miss_t_after=None):
    if miss_t_before is None:
        miss_t_before = miss_t
    if miss_t_after is None:
        miss_t_after = miss_t

    for this_t, dt in zip(
            _random_from_units(simdat.t.min(), simdat.t.max(), n_parcels),
            _random_from_units(-miss_t_before, miss_t_after, n_parcels)):
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
        if np.nanmin(dists) > close_thresh:
            parcels.append(parcel)
    simdat.parcels = parcels


def rad_var_v(r, V0=325*u.km/u.s, alpha=0.2):
    # From https://www.sciencedirect.com/science/article/pii/S027311771930746X
    # Higher alpha = steeper rise
    return V0 / (1 + np.exp(np.e - alpha * r / u.R_sun))


def gen_parcel_path(V0=325*u.km/u.s, alpha=0.2, end_point=250*u.R_sun):
    """Generates a trajectory with a radial gradient to the velocity"""
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
    """Adds random parcels with a radial gradient in their velocity

    Parameters
    ----------
    simdat : `SimulationData`
        The SimulationData to add parcels to
    n_parcels : ``int``, optional
        The number of parcels to add    
    V0 : ``Quantity``, optional
        The V0 parameter for the velocities
    alpha : ``float``, optional
        The alpha parameter for the gradient
    """
    t_min = 120 * u.R_sun / (200 * u.km / u.s)
    rs, ts, vs, rhos, rperp, rpar = gen_parcel_path(V0, alpha)
    
    for this_theta, t_start in zip(
            np.random.uniform(0, 2*np.pi, n_parcels),
            _random_from_units(simdat.t[0] - t_min, simdat.t[-1], n_parcels)):
        x = rs * np.cos(this_theta)
        y = rs * np.sin(this_theta)
        z = rs * 0
        tvals = t_start + ts
        simdat.parcels.append(ArrayThing(
            tvals, x, y, z,
            t=t_start, rholist=rhos, rperplist=rperp, rparlist=rpar,
            t_min=tvals[0], t_max=tvals[-1]))


def add_regular_near_impacts_rad_grad(simdat, V0=325*u.km/u.s, alpha=0.2,
                                      hit_spacing=1*u.hr, miss_t=1*u.hr,
                                      miss_t_before=None, miss_t_after=None):
    rs, ts, vs, rhos, rperp, rpar = gen_parcel_path(V0, alpha)
    
    if miss_t_before is None:
        miss_t_before = miss_t
    if miss_t_after is None:
        miss_t_after = miss_t

    t_impacts = _arange_from_units(simdat.t[0], simdat.t[-1], hit_spacing)
    for t_impact in t_impacts:
        with simdat.sc.at_temp(t_impact) as sc_impact:
            if np.isnan(miss_t_before):
                dt = miss_t_after
            elif np.isnan(miss_t_after):
                dt = -miss_t_before
            else:
                dt = np.random.uniform(-miss_t_before, miss_t_after)
            
            r_impact = sc_impact.r
            target_t = np.interp(r_impact, rs, ts)
            these_ts = ts + (t_impact - target_t) + dt
            
            simdat.parcels.append(ArrayThing(
                these_ts, rs * sc_impact.x / r_impact,
                rs * sc_impact.y / r_impact, rs * sc_impact.z / r_impact,
                t=these_ts[0], rholist=rhos, rperplist=rperp, rparlist=rpar,
                t_min=these_ts[0], t_max=these_ts[-1]))


def rotate_parcels_into_orbital_plane(simdat):
    """Rotates all parcels so that any parcel that was in the HCI equatorial
    plane is now in the PSP orbital plane

    Parameters
    ----------
    simdat : `SimulationData`
        The SimulationData to adjust in-place
    """
    obstime = utils.from_timestamp(simdat.t[0])
    for parcel in simdat.parcels:
        if isinstance(parcel, LinearThing):
            c = astropy.coordinates.SkyCoord(
                x=parcel.x_t0, y=parcel.y_t0, z=parcel.z_t0,
                v_x=parcel.vx_t0, v_y=parcel.vy_t0, v_z=parcel.vz_t0,
                obstime=obstime, frame='psporbitalframe',
                representation_type='cartesian')
            c = c.transform_to('heliocentricinertial').cartesian
            parcel.x_t0 = c.x
            parcel.y_t0 = c.y
            parcel.z_t0 = c.z
            parcel.vx_t0 = c.differentials['s'].d_x
            parcel.vy_t0 = c.differentials['s'].d_y
            parcel.vz_t0 = c.differentials['s'].d_z
        elif isinstance(parcel, ArrayThing):
            c = astropy.coordinates.SkyCoord(
                x=parcel.xlist, y=parcel.ylist, z=parcel.zlist,
                obstime=obstime, frame='psporbitalframe',
                representation_type='cartesian')
            c = c.transform_to('heliocentricinertial').cartesian
            parcel.xlist = c.x
            parcel.ylist = c.y
            parcel.zlist = c.z
            