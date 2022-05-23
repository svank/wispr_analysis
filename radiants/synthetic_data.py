from dataclasses import dataclass


from astropy.wcs import WCS
import numba
import numpy as np

@dataclass
class Thing:
    """Represents a generic object with a position and a velocity in 2-space"""
    x: float = 0
    y: float = 0
    vx: float = 0
    vy: float = 0
    
    def at(self, t):
        """Returns a copy of the object ``t`` seconds into the future.
        
        That object's position is updated according to ``t`` and the object's
        velocity.
        """
        return Thing(x=self.x + self.vx * t, y=self.y + self.vy * t, vx=self.vx, vy=self.vy)
    
    def in_front_of(self, other, t=0):
        """Returns whether this object is in front of the given object.
        
        "In front" is defined relative to the forward direction of the other
        object.
        
        Arguments
        ---------
        other : `Thing`
            Another `Thing` instance
        t : float
            Optionally, do the comparison ``t`` seconds into the future.
        
        Returns
        -------
        in_front : boolean
            ``True`` if this object is in front of ``other``.
        """
        if t != 0:
            other = other.at(t)
            self = self.at(t)
        angle = angle_between_vectors(
            other.vx,
            other.vy,
            self.x - other.x,
            self.y - other.y)
        return np.abs(angle) < np.pi/2
    
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
    
    def __sub__(self, other):
        """Converts this object to another reference frame
        
        ``Thing1 - Thing2`` generates an object representing ``Thing1`` in the
        reference frame of ``Thing2`` (i.e. with ``Thing2``'s position as the
        origin, and with velocities measured relative to those of ``Thing2``).
        """
        return Thing(x=self.x - other.x, y=self.y - other.y,
                vx=self.vx - other.vx, vy=self.vy - other.vy)


@numba.jit
def signed_angle_between_vectors(x1, y1, x2, y2):
    """Returns a signed angle between two vectors"""
    # Rotate so v1 is our x axis. We want the angle v2 makes to the x axis.
    # Its components in this rotated frame are its dot and cross products
    # with v1.
    if np.any((x1 == 0) * (y1 == 0)) or np.any((x2 == 0) * (y2 == 0)):
        raise ValueError("Angles are undefined with a zero-vector")
    dot_product = x1 * x2 + y1 * y2
    cross_product = x1 * y2 - y1 * x2
    
    return np.arctan2(cross_product, dot_product)


@numba.njit
def angle_between_vectors(x1, y1, x2, y2):
    """Returns an unsigned angle between two vectors"""
    return np.abs(signed_angle_between_vectors(x1, y1, x2, y2))


def calc_epsilon(sc, p, t=0):
    """
    Calculates elongation angle epsilon given spacecraft and parcel positions
    """
    # First find distances between objects
    d_sc_sun = np.sqrt(sc.at(t).x**2 + sc.at(t).y**2)
    d_p_sun = np.sqrt(p.at(t).x**2 + p.at(t).y**2)
    d_sc_p = np.sqrt((sc.at(t).x - p.at(t).x)**2 + (sc.at(t).y - p.at(t).y)**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        # Law of cosines:
        # d_p_sun^2 = d_sc_sun^2 + d_sc_p^2 - 2*d_sc_sun*d_sc_p*cos(epsilon)
        return np.arccos((d_p_sun**2 - d_sc_sun**2 - d_sc_p**2)
                / (-2 * d_sc_sun * d_sc_p))


def calc_FOV_pos(sc, p, t=0):
    """Calculates FOV position given spacecraft and parcel positions"""
    # Find the angle between the spacecraft's nominal to-Sun direction (that is, exactly 90 degrees from its travel direction) and the sc-p line
    offset = (p - sc).at(t)
    return angle_between_vectors(-sc.vy, sc.vx, offset.x, offset.y)


def synthesize_image(sc, parcels, t0, fov=90, projection='ARC',
        output_size_x=200, output_size_y=200, parcel_width=1, image_wcs=None):
    # Build output image WCS
    sc = sc.at(t0)
    parcels = [parcel.at(t0) for parcel in parcels]
    
    if image_wcs is None:
        image_wcs = WCS(naxis=2)
        forward_elongation = angle_between_vectors(sc.vx, sc.vy, -sc.x, -sc.y)
        image_wcs.wcs.crval = forward_elongation * 180 / np.pi, 0
        image_wcs.wcs.crpix = output_size_x/2 + .5, output_size_y/2 + .5
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
    
    output_image = np.zeros((output_size_y, output_size_x))
    
    for parcel in parcels:
        if not parcel.in_front_of(sc):
            continue
        parcel_angular_width = 2 * np.arctan(parcel_width / 2 / (sc - parcel).r)
        parcel_angular_width *= 180 / np.pi # degrees
        cdelt = parcel_angular_width / parcel_res
        parcel_wcs.wcs.cdelt = cdelt, cdelt
        # For each blob, calculate a position, update the WCS, and then project
        # the blob image onto the output image.
        # Is this overkill? Maybe?
        horiz_pos = calc_epsilon(sc, parcel) * 180 / np.pi
        vert_pos = 0
        parcel_wcs.wcs.crval = horiz_pos, vert_pos
        subimage = reproject.reproject_adaptive(
                (parcel_image, parcel_wcs), image_wcs, output_image.shape,
                boundary_mode='grid-constant', boundary_fill_value=0,
                roundtrip_coords=False, return_footprint=False,
                conserve_flux=True)
        output_image += subimage / (sc - parcel).r**2
    
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
