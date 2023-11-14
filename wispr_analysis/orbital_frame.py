import astropy.coordinates
import astropy.units as u
import astropy.wcs
import numpy as np
import spiceypy as spice

from . import planets


class PSPOrbitalFrame(astropy.coordinates.BaseCoordinateFrame):
    """
    This is a frame that represents a celestial sphere with its equator aligned
    with PSP's orbital plane.
    
    This is a time-varying frame, as PSP's orbital plane changes with every
    Venus encounter.
    
    Note: This frame does not attempt to account carefully for the origin of
    the coordinate system, and whether that is the Sun or PSP, etc. Formally,
    the origin is the barycenter of the solar system, as it is just a rotated
    ICRS frame. In practice, I use this frame with PSP taken as the
    origin---this does not affect where the stars appear due to their distance,
    and allows us to project into an unrotating, PSP-following frame.
    
    The zero point of longitude is set so that, when a full encounter is
    reprojected into this frame, it conveniently spans a range centered in
    [0, 360] without wrapping around.

    Parameters
    ----------
    obstime : str
        The time of the observation, defining the PSP orbital plane.
    """
    default_representation = astropy.coordinates.representation.SphericalRepresentation
    obstime = astropy.coordinates.TimeAttribute()


@astropy.coordinates.frame_transform_graph.transform(
    astropy.coordinates.DynamicMatrixTransform,
    astropy.coordinates.ICRS,
    PSPOrbitalFrame)
def icrs_to_psp(ICRScoord, PSPframe):
    obstime = (PSPframe.obstime
               or getattr(ICRScoord, 'obstime', '2020-01-01T01:01:01'))
    (perifocal_distance, eccentricity, inclination,
            lon_asc_node, arg_periapsis) = planets.get_orbital_elements(
            obstime, 'psp', coord_frame='J2000')
    
    # Rotate around the ascending-node axis by inclination
    matrix = spice.axisar(
        [np.cos(lon_asc_node), np.sin(lon_asc_node), 0],
        -inclination.to(u.rad).value)
    
    # Rotate around the vertical axis to position zero longitude conveniently
    matrix2 = spice.axisar(
        [0, 0, 1],
        -87.5 * np.pi/180)
    matrix = matrix2 @ matrix
    
    return matrix


@astropy.coordinates.frame_transform_graph.transform(
    astropy.coordinates.DynamicMatrixTransform,
    PSPOrbitalFrame,
    astropy.coordinates.ICRS)
def psp_to_icrs(PSPcoord, ICRSframe):
    obstime = (PSPcoord.obstime
               or getattr(ICRSframe, 'obstime', '2020-01-01T01:01:01'))
    (perifocal_distance, eccentricity, inclination,
            lon_asc_node, arg_periapsis) = planets.get_orbital_elements(
            obstime, 'psp', coord_frame='J2000')
    
    # Rotate around the vertical axis to put zero longitude where it should be
    matrix = spice.axisar(
        [0, 0, 1],
        87.5 * np.pi/180)
    
    # Rotate around the ascending-node axis by inclination
    matrix2 = spice.axisar(
        [np.cos(lon_asc_node), np.sin(lon_asc_node), 0],
        inclination.to(u.rad).value)
    matrix = matrix2 @ matrix
    
    return matrix


def orbital_plane_wcs_frame_mapping(wcs):
    ctypes = {c[:4] for c in wcs.wcs.ctype}
    if not ({'PSLN', 'PSLT'} <= ctypes):
        return None
    
    dateobs = wcs.wcs.dateavg or wcs.wcs.dateobs or None
    orbital_frame = PSPOrbitalFrame(obstime=dateobs)
    return orbital_frame


def orbital_plane_frame_wcs_mapping(frame, projection='TAN'):
    if not isinstance(frame, PSPOrbitalFrame):
        return None
    wcs = astropy.wcs.WCS(naxis=2)
    if frame.obstime:
        wcs.wcs.dateobs = frame.obstime.utc.isot
    wcs.wcs.ctype = f'PSLN-{projection}', f'PSLT-{projection}'
    wcs.wcs.cunit = ['deg', 'deg']
    return wcs


astropy.wcs.utils.WCS_FRAME_MAPPINGS.append([orbital_plane_wcs_frame_mapping])
astropy.wcs.utils.FRAME_WCS_MAPPINGS.append([orbital_plane_frame_wcs_mapping])
astropy.wcs.wcsapi.fitswcs.CTYPE_TO_UCD1_CUSTOM.append(
    {
        "PSLN": "custom:pos.psporbitalplane.lon",
        "PSLT": "custom:pos.psporbitalplane.lat",
    })