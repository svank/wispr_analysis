import copy
from dataclasses import dataclass


from astropy.wcs import WCS
import numpy as np
import reproject
import scipy


class Thing:
    """Represents a generic object with a position and a velocity in 2-space
    
    Subclasses have `x`, `y`, `vx`, and `vy` attributes which provide those
    quantities. Instances can be set to a specific time, which then determines
    how those attributes are computed
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
     
    def set_t(self, t):
        """ Sets the object's time to ``t``. """
        t = np.atleast_1d(t)
        self.t = t
    
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
        if t is not None:
            other = other.at(t)
            self = self.at(t)
        elif not self.is_same_time(other):
            raise ValueError(
                    "Objects are not set at same time---must specify `t`")
        separation_vector = self - other
        angle = angle_between_vectors(
            other.vx,
            other.vy,
            separation_vector.x,
            separation_vector.y)
        in_front = np.atleast_1d(np.abs(angle) < np.pi/2)
        in_front[separation_vector.r == 0] = False
        return in_front
    
    @property
    def r(self):
        """
        Convenience access to sqrt(x**2 + y**2)
        """
        return np.sqrt(self.x**2 + self.y**2)
    
    @property
    def v(self):
        """
        Convenience access to sqrt(vx**2 + vy**2)
        """
        return np.sqrt(self.vx**2 + self.vy**2)
    
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
    vx_t0: float = 0
    vy_t0: float = 0
    
    def __init__(self, x=0, y=0, vx=0, vy=0, t=0, t_min=None, t_max=None):
        """
        Accepts physical parameters, as well as the corresponding time
        
        That is, `x`, `y`, etc. need not be specified for `t=0` if the
        appropriate time is provided.
        
        """
        self.vx_t0 = vx
        self.vy_t0 = vy
        self.t = t
        
        self.x_t0 = x - vx * t
        self.y_t0 = y - vy * t
        
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
    
    def offset_by_time(self, dt):
        out = self.copy()
        out.x_t0 += out.vx_t0 * dt
        out.y_t0 += out.vy_t0 * dt
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
    tlist: np.ndarray = 0
    
    def __init__(self, tlist, xlist=0, ylist=0, t=0, t_min=None, t_max=None):
        """
        Parameters
        ----------
        tlist : np.ndarray
            The list of time points at which positions are provided
        xlist, ylist : np.ndarray or float
            The specified positions. If either is a single number, that number
            is used at all points in time.
        t : float
            The time the object is currently at.
        """
        xlist = np.atleast_1d(xlist)
        ylist = np.atleast_1d(ylist)
        tlist = np.atleast_1d(tlist)
        
        # Check that tlist is sorted
        if np.any(np.diff(tlist) < 0):
            raise ValueError("tlist must be sorted")
        
        if len(xlist) == 1:
            xlist = np.repeat(xlist, len(tlist))
        if len(ylist) == 1:
            ylist = np.repeat(ylist, len(tlist))
        
        if len(xlist) != len(tlist):
            raise ValueError("Invalid length for xlist")
        if len(ylist) != len(tlist):
            raise ValueError("Invalid length for ylist")
        
        self.xlist = xlist
        self.ylist = ylist
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
    
    def _finite_difference(self, interpolator, dt):
        try:
            y1 = interpolator(self.t - dt/2)
            y2 = interpolator(self.t + dt/2)
        except ValueError:
            try:
                y1 = interpolator(self.t)
                y2 = interpolator(self.t + dt)
            except ValueError:
                y1 = interpolator(self.t - dt)
                y2 = interpolator(self.t)
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
        return self.thing1.at(self.t).x - self.thing2.at(self.t).x
    
    @property
    def y(self):
        return self.thing1.at(self.t).y - self.thing2.at(self.t).y
    
    @property
    def vx(self):
        dt = .0001
        x_before = self.at(self.t - dt/2).x
        x_after = self.at(self.t + dt/2).x
        return (x_after - x_before) / dt
    
    @property
    def vy(self):
        dt = .0001
        y_before = self.at(self.t - dt/2).y
        y_after = self.at(self.t + dt/2).y
        return (y_after - y_before) / dt


def signed_angle_between_vectors(x1, y1, x2, y2):
    """Returns a signed angle between two vectors"""
    # Rotate so v1 is our x axis. We want the angle v2 makes to the x axis.
    # Its components in this rotated frame are its dot and cross products
    # with v1.
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    y1 = np.atleast_1d(y1)
    y2 = np.atleast_1d(y2)
    
    dot_product = x1 * x2 + y1 * y2
    cross_product = x1 * y2 - y1 * x2
    
    angle = np.arctan2(cross_product, dot_product)
    v1_is_zero = ((x1 == 0) * (y1 == 0))
    v2_is_zero = ((x2 == 0) * (y2 == 0))
    angle[v1_is_zero + v2_is_zero] = np.nan
    return angle


def angle_between_vectors(x1, y1, x2, y2):
    """Returns an unsigned angle between two vectors"""
    return np.abs(signed_angle_between_vectors(x1, y1, x2, y2))


def calc_epsilon(sc, p, t=0):
    """
    Calculates (unsigned) elongation angle epsilon for spacecraft and parcel positions
    """
    t = np.atleast_1d(t)
    if len(t) > 1 or t != 0:
        sc = sc.at(t)
        p = p.at(t)
    # First find distances between objects
    d_sc_sun = sc.r
    d_p_sun = p.r
    d_sc_p = (sc - p).r
    with np.errstate(invalid='ignore', divide='ignore'):
        # Law of cosines:
        # d_p_sun^2 = d_sc_sun^2 + d_sc_p^2 - 2*d_sc_sun*d_sc_p*cos(epsilon)
        return np.arccos((d_p_sun**2 - d_sc_sun**2 - d_sc_p**2)
                / (-2 * d_sc_sun * d_sc_p))


def calc_FOV_pos(sc, p, t=0):
    """
    Calculates FOV position given spacecraft and parcel positions
    
    FOV angle is measured relative to the spacecraft's forward direction and
    increases to the right
    """
    offset = (p - sc).at(t)
    return -signed_angle_between_vectors(sc.vx, sc.vy, offset.x, offset.y)


def synthesize_image(sc, parcels, t0, fov=95, projection='ARC',
        output_size_x=200, output_size_y=200, parcel_width=1, image_wcs=None,
        psychadelic=False, celestial_wcs=False):
    sc = sc.at(t0)
    
    # Build output image WCS
    if image_wcs is None:
        image_wcs = WCS(naxis=2)
        ## Find elongation of s/c forward direction by computing the angle
        ## between it and the sunward direction
        #forward_elongation = angle_between_vectors(sc.vx, sc.vy, -sc.x, -sc.y)
        ## Set the reference pixel coordiantes as the forward-direction
        ## elongation for longitude, and zero latitude (assume s/c is in
        ## ecliptic plane)
        #image_wcs.wcs.crval = forward_elongation[0] * 180 / np.pi, 0
        
        # Set the reference pixel coordiantes as 61 degress HPC, which is the
        # center of WISPR's compositve FOV (as detailed in the in-flight
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
    
    # Build a "blob" image on a small canvas
    parcel_amp = 1
    parcel_res = 21
    parcel_stdev = parcel_res / 6
    x = np.arange(parcel_res) - parcel_res / 2 + .5
    X, Y = np.meshgrid(x, x)
    parcel_image = parcel_amp * np.exp(-(X**2 + Y**2)/2/parcel_stdev**2)
    
    # Build a WCS for the blob
    parcel_wcs = WCS(naxis=2)
    parcel_wcs.wcs.crpix = parcel_res/2 + .5, parcel_res/2 + .5
    parcel_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    parcel_wcs.wcs.cunit = "deg", "deg"
    
    if psychadelic:
        output_image = np.zeros((output_size_y, output_size_x, 3))
    else:
        output_image = np.zeros((output_size_y, output_size_x))
    
    for p in parcels:
        # For each blob, calculate a position, update the WCS, and then project
        # the blob image onto the output image.
        # Is this overkill? Probably?
        parcel = p.at(t0)
        if np.isnan(parcel.x):
            continue
        
        # We'll compute a scaling factor to reduce parcels' brightness as we
        # fly though them, to try to reduce flashiness as that happens.
        parcel_distance = (parcel - sc).r
        inside_parcel_scaling = 1
        if parcel_distance < parcel_width / 2:
            inside_parcel_scaling = parcel_distance / (parcel_width / 2)
        if not parcel.in_front_of(sc):
            if parcel_distance > parcel_width / 2:
                continue
            inside_parcel_scaling = .5 * (1 - inside_parcel_scaling)
        elif inside_parcel_scaling < 1:
            inside_parcel_scaling = .5 + .5 * inside_parcel_scaling
        # Square it just for a bit more flashiness suppression
        inside_parcel_scaling = inside_parcel_scaling**2

        
        # Compute the apparent angular size of the parcel
        parcel_angular_width = 2 * np.arctan(parcel_width / 2 / (sc - parcel).r)
        parcel_angular_width *= 180 / np.pi # degrees
        cdelt = parcel_angular_width / parcel_res
        try:
            cdelt = cdelt[0]
        except TypeError:
            pass
        parcel_wcs.wcs.cdelt = cdelt, cdelt
        horiz_pos = calc_epsilon(sc, parcel) * 180 / np.pi
        vert_pos = 0
        parcel_wcs.wcs.crval = horiz_pos[0], vert_pos
        subimage = reproject.reproject_adaptive(
                (parcel_image, parcel_wcs), image_wcs, output_image.shape[:2],
                boundary_mode='grid-constant', boundary_fill_value=0,
                roundtrip_coords=False, return_footprint=False,
                conserve_flux=True)
        # Scale the brightness of the reprojected blob
        subimage = subimage / parcel_distance**2 / parcel.r**2
        subimage *= inside_parcel_scaling
        
        if psychadelic:
            # Add a dimension and assign a random color to the parcel. Ensure
            # the same color is chosen for each frame.
            np.random.seed(id(p) % (2**32 - 1))
            color = np.random.random(3)
            subimage = subimage[:, :, None] * color[None, None, :]
        
        # Build up the output image by adding together all the reprojected blobs
        output_image += subimage
    
    if celestial_wcs:
        image_wcs.wcs.ctype = f"RA---{projection}", f"DEC--{projection}"
        to_sun_x = -sc.x
        to_sun_y = -sc.y
        to_sun_theta = np.arctan2(to_sun_y, to_sun_x) * 180 / np.pi
        fov_center = to_sun_theta - 61
        image_wcs.wcs.crval = fov_center, 0
        cdelt = image_wcs.wcs.cdelt[0]
        image_wcs.wcs.cdelt = -cdelt, cdelt
        
    return output_image, image_wcs


def calculate_radiant(sc, parcel, t0=0):
    t0 = np.atleast_1d(t0)
    if len(t0) > 1 or t0 != 0:
        sc = sc.at(t0)
        parcel = parcel.at(t0)
    infront = np.atleast_1d(parcel.in_front_of(sc))
    if not np.any(infront):
        return np.full(max(len(t0), len(np.atleast_1d(parcel.x))), np.nan)
    v_sc = sc.v
    e_sc = np.atleast_1d(angle_between_vectors(sc.vx, sc.vy, -sc.x, -sc.y))
    v_p = parcel.v
    dphi = np.atleast_1d(angle_between_vectors(sc.x, sc.y, parcel.x, parcel.y))
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
    elongation : scalar or ndarray
        Elongation values to be converted. Measured in radians, with zero
        representing the Sunward direction. Unsigned.
    
    Returns
    -------
    fov : ndarray
        Field-of-view position in radians
    """
    sc_direction = signed_angle_between_vectors(sc.vx, sc.vy, -sc.x, -sc.y)
    return np.where(
            sc_direction < 0,
            -(elongation + sc_direction),
            elongation - sc_direction)
