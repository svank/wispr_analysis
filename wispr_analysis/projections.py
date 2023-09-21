from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import reproject
import scipy.optimize

from . import utils


class HprWcs(utils.FakeWCS):
    """Implements Helioprojective-Cartesian coordinates in a radial projection
    
    This implements the minimal set of the WCS API to support reprojection,
    plus some helper functions.
    """
    pa_of_ecliptic = 90
    
    def __init__(self, wcs, ref_pa, ref_y, dpa,
            ref_elongation, ref_x, delongation):
        """Create an `HprWcs`

        Parameters
        ----------
        wcs : ``WCS``
            A WCS describing the "other side" of the reprojection---i.e.,
            describing the input data that will be reprojected to radial
            coordinates. This is needed to copy some metadata.
        ref_pa : float, optional
            The position angle of the reference pixel
        ref_y : float, optional
            The reference pixel
        dpa : float, optional
            The degrees-per-pixel spacing of the output image along the PA axis
        ref_elongation : float, optional
            The elongation of the reference pixel
        ref_x, ref_x : float, optional
            The reference pixel
        delongation : float, optional
            The degrees-per-pixel spacing of the output image along the
            elongation axis
        """
        super().__init__(wcs)
        self.ref_pa = ref_pa
        self.ref_elongation = ref_elongation
        self.ref_x = ref_x
        self.ref_y = ref_y
        self.dpa = dpa
        self.delongation = delongation
    
    @classmethod
    def hpc_to_hpr(cls, lon, lat):
        lon = np.asarray(lon) * np.pi / 180
        lat = np.asarray(lat) * np.pi / 180
        
        # Expressions from Snyder (1987)
        # https://pubs.er.usgs.gov/publication/pp1395
        # Eqn (5-3a)
        elongation = 2 * np.arcsin(np.sqrt(
            np.sin(lat/2)**2 + np.cos(lat) * np.sin(lon/2)**2
        ))
        # Eqn (5-4b)
        pa = np.arctan2(np.cos(lat) * np.sin(lon), np.sin(lat))
        
        elongation *= 180 / np.pi
        pa *= 180 / np.pi
        pa += (cls.pa_of_ecliptic - 90)
        
        return elongation, pa
    
    @classmethod
    def hpr_to_hpc(cls, elongation, pa):
        elongation = np.asarray(elongation) * np.pi / 180
        pa = np.asarray(pa) - (cls.pa_of_ecliptic - 90)
        pa *= np.pi / 180
        
        # Expressions from Snyder (1987)
        # https://pubs.er.usgs.gov/publication/pp1395
        # Eqn (5-5)
        lat = np.arcsin(np.sin(elongation) * np.cos(pa))
        # Eqn (5-6)
        lon = np.arctan2(np.sin(elongation) * np.sin(pa), np.cos(elongation))
        
        lat *= 180 / np.pi
        lon *= 180 / np.pi
        return lon, lat
    
    def pix_to_hpr(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        pa = (y - self.ref_y) * self.dpa + self.ref_pa
        elongation = ((x - self.ref_x) * self.delongation
                + self.ref_elongation)
        return elongation, pa
    
    def hpr_to_pix(self, elongation, pa):
        elongation = np.asarray(elongation)
        pa = np.asarray(pa)
        x = (elongation - self.ref_elongation) / self.delongation + self.ref_x
        y = (pa - self.ref_pa) / self.dpa + self.ref_y
        
        return x, y
    
    def pixel_to_world_values(self, x, y):
        hpr = self.pix_to_hpr(x, y)
        hpc = self.hpr_to_hpc(*hpr)
        return hpc
    
    def world_to_pixel_values(self, lon, lat):
        hpr = self.hpc_to_hpr(lon, lat)
        pix = self.hpr_to_pix(*hpr)
        return pix


def reproject_to_radial(data, wcs, out_shape=None, dpa=None, delongation=None,
        ref_pa=100, ref_elongation=13, ref_x=None, ref_y=None):
    """Helper to reproject from helioprojective-cartesian to radial
    
    In the radial frame, the two axes are position angle (degrees around from a
    reference position---by default, increasing clockwise from 0 degrees at
    solar North) and elongation (degrees from disc center).
    
    This function accepts parameters for WCS-like specification of the output
    radial projection. If not provided, defaults suitable for a WISPR-I image
    are used.

    Parameters
    ----------
    data : ``np.ndarray``
        The input image
    wcs : ``WCS``
        A ``WCS`` describing the input data
    out_shape : tuple, optional
        The shape for the output array
    dpa : float, optional
        The degrees-per-pixel spacing of the output image along the PA axis
    delongation : float, optional
        The degrees-per-pixel spacing of the output image along the elongation
        axis
    ref_pa : float, optional
        The position angle of the reference pixel
    ref_elongation : float, optional
        The elongation of the reference pixel
    ref_x, ref_y : float, optional
        The reference pixel

    Returns
    -------
    reprojected : ``np.ndarray``
        The reprojected image
    wcs_out : `HprWcs`
        A WCS-like object which converts between pixels and
        helioprojective-cartesian coordinates using the specified radial
        projection.
    """
    if out_shape is None:
        # Defaults that are decent for a full-res WISPR-I image
        out_shape = list(data.shape)
        out_shape[1] -= 250
        out_shape[1] *= 2
        out_shape[0] += 350
    if dpa is None:
        dpa = wcs.wcs.cdelt[1] * 1.5
    if delongation is None:
        delongation = wcs.wcs.cdelt[0] * .75
    if ref_x is None:
        ref_x = 0
    if ref_y is None:
        ref_y = out_shape[0] // 2
    wcs_out = HprWcs(wcs, ref_pa=ref_pa, ref_y=ref_y, dpa=-dpa,
            ref_elongation=ref_elongation, ref_x=ref_x, delongation=delongation)
    reprojected = reproject.reproject_adaptive(
        (data, wcs), wcs_out, out_shape,
        center_jacobian=False, roundtrip_coords=False, return_footprint=False)
    return reprojected, wcs_out


def reproject_from_radial(data, wcs, out_shape=None, dpa=None, delongation=None,
        ref_pa=100, ref_elongation=13, ref_x=None, ref_y=None):
    """Helper to reproject from helioprojective-radial to cartesian
    
    In the radial frame, the two axes are position angle (degrees around from a
    reference position---by default, increasing clockwise from 0 degrees at
    solar North) and elongation (degrees from disc center).
    
    This function accepts parameters for WCS-like specification of the input
    data's radial projection. If not provided, defaults suitable for a WISPR-I
    image are used.

    Parameters
    ----------
    data : ``np.ndarray``
        The input image
    wcs : ``WCS``
        A ``WCS`` describing the output, non-radial projection
    out_shape : tuple, optional
        The shape for the output array
    dpa : float, optional
        The degrees-per-pixel spacing of the input image along the PA axis
    delongation : float, optional
        The degrees-per-pixel spacing of the input image along the elongation
        axis
    ref_pa : float, optional
        The position angle of the reference pixel
    ref_elongation : float, optional
        The elongation of the reference pixel
    ref_x, ref_x : float, optional
        The reference pixel

    Returns
    -------
    reprojected : ``np.ndarray``
        The reprojected image
    wcs : ``WCS``
        The WCS of the output image
    """
    if out_shape is None:
        # Defaults that are decent for a full-res WISPR-I image
        out_shape = list(data.shape)
        out_shape[1] -= 250
        out_shape[1] *= 2
        out_shape[0] += 350
    if dpa is None:
        dpa = wcs.wcs.cdelt[1] * 1.5
    if delongation is None:
        delongation = wcs.wcs.cdelt[0] * .75
    if ref_x is None:
        ref_x = 0
    if ref_y is None:
        ref_y = out_shape[0] // 2
    wcs_in = HprWcs(wcs, ref_pa=ref_pa, ref_y=ref_y, dpa=-dpa,
        ref_elongation=ref_elongation, ref_x=ref_x, delongation=delongation)
    reprojected = reproject.reproject_adaptive(
        (data, wcs_in), wcs, out_shape,
        center_jacobian=False, roundtrip_coords=False, return_footprint=False)
    return reprojected, wcs


def label_radial_axes(wcs, ax=None):
    """Helper function to label plot axes as elongation and position angle
    
    Note that these radial coordinates can't be used with ``WCSAxes`` or via
    ``plt.subplot(111, projection=my_hpr_wcs)``, since Astropy doesn't
    understand this coordinate system. `HprWcs` outputs
    helioprojective-cartesian coordinates, so this helper function is required
    to label with helioprojective-radial coordinates.

    Parameters
    ----------
    wcs : `HprWcs`
        An `HprWcs` describing the image being plotting, such as the one
        returned by `reproject_to_radial`.
    ax : ``Axes``, optional
        A matplotlib ``Axes`` to label. If not provided, the currently-active
        ``Axes`` are used.
    """
    if ax is None:
        ax = plt.gca()
    
    xmin, xmax = ax.get_xlim()
    emin, emax = wcs.pix_to_hpr([xmin, xmax], [1, 1])[0]
    emin = int(np.ceil(emin/10)) * 10
    emax = int(np.floor(emax/10)) * 10
    spacing = 10 if (emax - emin) < 80 else 20
    tick_values = range(emin, emax+1, spacing)
    xtick_locs = [wcs.hpr_to_pix(elongation, 0)[0]
                  for elongation in tick_values]
    xtick_labels = [f"{elongation}°" for elongation in tick_values]
    ax.set_xticks(xtick_locs, xtick_labels)
    ax.set_xlabel("Elongation")
    
    ymin, ymax = ax.get_ylim()
    pmin, pmax = wcs.pix_to_hpr([1, 1], [ymin, ymax])[1]
    if pmin > pmax:
        pmin, pmax = pmax, pmin
    pmin = int(np.ceil(pmin/10)) * 10
    pmax = int(np.floor(pmax/10)) * 10
    spacing = 10 if (pmax - pmin) < 80 else 20
    tick_values = range(pmin, pmax+1, spacing)
    ytick_locs = [wcs.hpr_to_pix(30, pa)[1]
                  for pa in tick_values]
    ytick_labels = [f"{pa}°" for pa in tick_values]
    ax.set_yticks(ytick_locs, ytick_labels)
    ax.set_ylabel("Position Angle")


def produce_radec_for_hp_wcs(wcs_hp, ref_wcs_hp=None, ref_wcs_ra=None,
        ref_hdr=None):
    """Produces an RA/Dec WCS for an HP WCS, from a pair of RA/Dec and HP WCSs
    
    The intended use case is producing composite images, where an output HP WCS
    is constructed from scratch, and a corresponding RA/Dec WCS in the same
    projection is desired. To produce it, the RA/Dec and HP WCSs of one of the
    input images are used.
    """
    if ref_hdr:
        with utils.ignore_fits_warnings():
            ref_wcs_hp = WCS(ref_hdr)
            ref_wcs_ra = WCS(ref_hdr, key='A')
    # Begin to create the output WCS
    wcs_ra = wcs_hp.deepcopy()
    wcs_ra.wcs.ctype = 'RA---ARC', 'DEC--ARC'
    # Update the reference coordinate to the RA/Dec coordinate corresponding to
    # the original HP reference coordinate.
    wcs_ra.wcs.crval = ref_wcs_ra.all_pix2world(
            *ref_wcs_hp.all_world2pix(*wcs_hp.wcs.crval, 0), 0)
    wcs_ra.wcs.cdelt = -wcs_hp.wcs.cdelt[0], wcs_hp.wcs.cdelt[1]
    
    # We now need to find the rotation of the RA/Dec frame. We do that
    # iteratively, using a set of reference HP coordinates for which we compute
    # the corresponding RA/Dec coordinates using the reference WCSs
    pts_x = np.linspace(50, ref_wcs_hp.pixel_shape[0] - 50, 5)
    pts_y = np.linspace(50, ref_wcs_hp.pixel_shape[1] - 50, 5)
    pts_x, pts_y = np.meshgrid(pts_x, pts_y)
    
    pts_hp = ref_wcs_hp.all_pix2world(pts_x, pts_y, 0)
    pts_ra = ref_wcs_ra.all_pix2world(pts_x, pts_y, 0)
    pts_x, pts_y = wcs_hp.all_world2pix(*pts_hp, 0)
    
    pts_ra[0] *= np.pi / 180
    pts_ra[1] *= np.pi / 180
    
    def f(angle):
        angle = angle[0]
        wcs_ra.wcs.pc = np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
        pts_ra_trial = wcs_ra.all_pix2world(pts_x, pts_y, 0)
        
        ra1, dec1 = pts_ra_trial
        ra2, dec2 = pts_ra
        
        ra1 = ra1 * np.pi / 180
        dec1 = dec1 * np.pi / 180
        
        sindec1 = np.sin(dec1)
        cosdec1 = np.cos(dec1)
        sindec2 = np.sin(dec2)
        cosdec2 = np.cos(dec2)
        dra = ra1 - ra2
        
        # Great-circle distance---handles the wraparound in RA
        gcd = np.arctan(
                np.sqrt((cosdec2 * np.sin(dra))**2
                    + (cosdec1 * sindec2 - sindec1 * cosdec2 * np.cos(dra))**2 )
                / (sindec1 * sindec2 + cosdec1 * cosdec2 * np.cos(dra))
                )
        return np.sum(np.square(gcd))
    
    res = scipy.optimize.minimize(f, 0, bounds=[[-np.pi, np.pi]])
    angle = res.x[0]
    wcs_ra.wcs.pc = np.array([[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])
    return wcs_ra


def overlay_radial_grid(image, wcs, ax=None):
    """
    Overlays an (elongation, pa) grid on a non-radially-projected image.
    
    Draws a grid in ten-degree increments.
    
    Parameters
    ----------
    image : ``np.ndarray``
        The data array being plotted. Needed to determine the size of the image
        array.
    wcs : `HprWcs`
        The `HprWcs` controlling the transformation to radial
        coordinates.
    ax : `matplotlib.axes.Axes`
        Optional, the axes to plot on
    """
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    xx, yy = np.meshgrid(x, y)
    hplon, hplat = wcs.pixel_to_world_values(xx, yy)
    elongation, pa = HprWcs.hpc_to_hpr(hplon, hplat)
    
    if ax is None:
        ax = plt.gca()
    ax.contour(elongation, levels=np.arange(10, 180, 10),
            colors='w', alpha=.5, linewidths=.5)
    ax.contour(pa, levels=np.arange(0, 180, 10),
            colors='w', alpha=.5, linewidths=.5)
