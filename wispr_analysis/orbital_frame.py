import sunpy.coordinates.frameattributes
import astropy.coordinates
import astropy.units as u
import astropy.wcs
import numpy as np
import spiceypy as spice
import sunpy.coordinates

from . import planets


class PSPOrbitalFrame(astropy.coordinates.BaseCoordinateFrame):
    """
    This is a heliocentric frame with its equator aligned with PSP's orbit plane
    
    This is a time-varying frame, as PSP's orbital plane changes with every
    Venus encounter.
    
    This frame is registered under the WCS keys POLN and POLT.
    
    Parameters
    ----------
    obstime
        The time of the observation, defining the PSP orbital plane.
    """
    default_representation = astropy.coordinates.representation.SphericalRepresentation
    obstime = astropy.coordinates.TimeAttribute()


@astropy.coordinates.frame_transform_graph.transform(
    astropy.coordinates.DynamicMatrixTransform,
    sunpy.coordinates.HeliocentricInertial,
    PSPOrbitalFrame)
def hci_to_psp_orbital(HCIcoord, PSPframe):
    obstime = PSPframe.obstime or HCIcoord.obstime
    (perifocal_distance, eccentricity, inclination,
            lon_asc_node, arg_periapsis) = planets.get_orbital_elements(
            obstime, 'psp', coord_frame='HCI')
    
    # Rotate around the ascending-node axis by inclination
    matrix = spice.axisar(
        [np.cos(lon_asc_node), np.sin(lon_asc_node), 0],
        -inclination.to(u.rad).value)
    
    # Rotate around the vertical axis to position zero longitude conveniently
    matrix2 = spice.axisar([0, 0, 1], -10 * np.pi/180)
    matrix = matrix2 @ matrix
    
    return matrix


@astropy.coordinates.frame_transform_graph.transform(
    astropy.coordinates.DynamicMatrixTransform,
    PSPOrbitalFrame,
    sunpy.coordinates.HeliocentricInertial)
def psp_orbital_to_hci(PSPcoord, HCIframe):
    obstime = PSPcoord.obstime or HCIframe.obstime
    (perifocal_distance, eccentricity, inclination,
            lon_asc_node, arg_periapsis) = planets.get_orbital_elements(
            obstime, 'psp', coord_frame='HCI')
    
    # Rotate around the vertical axis to put zero longitude where it should be
    matrix = spice.axisar([0, 0, 1], 10 * np.pi/180)
    
    # Rotate around the ascending-node axis by inclination
    matrix2 = spice.axisar(
        [np.cos(lon_asc_node), np.sin(lon_asc_node), 0],
        inclination.to(u.rad).value)
    matrix = matrix2 @ matrix
    
    return matrix


class PSPFrame(astropy.coordinates.BaseCoordinateFrame):
    """
    This is an observer-centric frame with its equator aligned with PSP's
    orbital plane, and with longitude fixed to an inertial reference point (i.e.
    fixed to the stars).
    
    This is a time-varying frame, as PSP's orbital plane changes with every
    Venus encounter.
    
    This frame is registered under the WCS keys PSLN and PSLT.

    Parameters
    ----------
    obstime
        The time of the observation, defining the PSP orbital plane.
    observer
        The observer, defining the origin of the coordinate system.
    """
    default_representation = astropy.coordinates.representation.SphericalRepresentation
    obstime = astropy.coordinates.TimeAttribute()
    observer = sunpy.coordinates.frameattributes.ObserverCoordinateAttribute(
        sunpy.coordinates.HeliographicStonyhurst)


@astropy.coordinates.frame_transform_graph.transform(
    astropy.coordinates.DynamicMatrixTransform,
    sunpy.coordinates.Helioprojective,
    PSPFrame)
def hpc_to_psp(HPcoord, PSPframe):
    if not _observers_are_equal(HPcoord.observer, PSPframe.observer):
        raise ValueError("Observers are not equal")
    if not _times_are_equal(HPcoord.obstime, PSPframe.obstime):
        raise ValueError("Obstimes are not equal")
    obstime = HPcoord.obstime or PSPframe.obstime
    observer = HPcoord.observer or PSPframe.observer
    orbital_north = astropy.coordinates.SkyCoord(
        0*u.m, 0*u.m, 1000*u.m, representation_type='cartesian',
        frame=PSPOrbitalFrame, obstime=obstime, observer=observer)
    orbital_north = orbital_north.transform_to('helioprojective')
    angle = np.arctan2(orbital_north.Tx, orbital_north.Ty)
    
    matrix = spice.axisar([1, 0, 0], angle.to_value(u.rad))
    
    observer = observer.transform_to(PSPOrbitalFrame(obstime=obstime))
    lon = observer.lon - 180 * u.deg
    matrix2 = spice.axisar([0, 0, 1], -lon.to_value(u.rad))
    matrix = matrix2 @ matrix
    
    return matrix


@astropy.coordinates.frame_transform_graph.transform(
    astropy.coordinates.DynamicMatrixTransform,
    PSPFrame,
    sunpy.coordinates.Helioprojective)
def psp_to_hpc(PSPcoord, HPframe):
    if not _observers_are_equal(PSPcoord.observer, HPframe.observer):
        raise ValueError("Observers are not equal")
    if not _times_are_equal(PSPcoord.obstime, HPframe.obstime):
        raise ValueError("Obstimes are not equal")
    obstime = PSPcoord.obstime or HPframe.obstime
    observer = PSPcoord.observer or HPframe.observer
    orbital_north = astropy.coordinates.SkyCoord(
        0*u.m, 0*u.m, 1000*u.m, representation_type='cartesian',
        frame=PSPOrbitalFrame, obstime=obstime, observer=observer)
    orbital_north = orbital_north.transform_to('helioprojective')
    angle = np.arctan2(orbital_north.Tx, orbital_north.Ty)
    
    matrix = spice.axisar([1, 0, 0], -angle.to_value(u.rad))
    
    observer = observer.transform_to(PSPOrbitalFrame(obstime=obstime))
    lon = observer.lon - 180 * u.deg
    matrix2 = spice.axisar([0, 0, 1], lon.to_value(u.rad))
    matrix = matrix @ matrix2
    
    return matrix


def orbital_plane_wcs_frame_mapping(wcs):
    ctypes = {c[:4] for c in wcs.wcs.ctype}
    if ({'PSLN', 'PSLT'} <= ctypes):
        dateobs = wcs.wcs.dateavg or wcs.wcs.dateobs or None
        observer = _get_observer_from_wcs(wcs, dateobs)
        return PSPFrame(obstime=dateobs, observer=observer)
    if ({'POLN', 'POLT'} <= ctypes):
        dateobs = wcs.wcs.dateavg or wcs.wcs.dateobs or None
        return PSPOrbitalFrame(obstime=dateobs)
    return None


def orbital_plane_frame_wcs_mapping(frame, projection='TAN'):
    if isinstance(frame, PSPOrbitalFrame):
        wcs = astropy.wcs.WCS(naxis=2)
        if frame.obstime:
            wcs.wcs.dateobs = frame.obstime.utc.isot
        wcs.wcs.ctype = f'POLN-{projection}', f'POLT-{projection}'
        wcs.wcs.cunit = ['deg', 'deg']
        return wcs
    
    if isinstance(frame, PSPFrame):
        wcs = astropy.wcs.WCS(naxis=2)
        if frame.obstime:
            wcs.wcs.dateobs = frame.obstime.utc.isot
        _set_wcs_aux_obs_coord(wcs, frame.observer)
        wcs.wcs.ctype = f'PSLN-{projection}', f'PSLT-{projection}'
        wcs.wcs.cunit = ['deg', 'deg']
        return wcs
    return None


astropy.wcs.utils.WCS_FRAME_MAPPINGS.append([orbital_plane_wcs_frame_mapping])
astropy.wcs.utils.FRAME_WCS_MAPPINGS.append([orbital_plane_frame_wcs_mapping])
# TODO: Once astropy releases with PR 15626, these can/should get "custom:"
# prefixes
astropy.wcs.wcsapi.fitswcs.CTYPE_TO_UCD1_CUSTOM.append(
    {
        "PSLN": "pos.pspframe.lon",
        "PSLT": "pos.pspframe.lat",
        "POLN": "pos.psporbitalframe.lon",
        "POLT": "pos.psporbitalframe.lat",
    })


def _get_observer_from_wcs(wcs, dateobs):
    # Ripped from sunpy
    rsun = wcs.wcs.aux.rsun_ref
    if rsun is not None:
        rsun *= u.m
    
    required_attrs = {sunpy.coordinates.HeliographicStonyhurst:
                        ['hgln_obs', 'hglt_obs', 'dsun_obs'],
                      sunpy.coordinates.HeliographicCarrington:
                        ['crln_obs', 'hglt_obs', 'dsun_obs']}
    
    observer = None
    for frame, attr_names in required_attrs.items():
        attrs = [getattr(wcs.wcs.aux, attr_name) for attr_name in attr_names]
        if all([attr is not None for attr in attrs]):
            kwargs = {'obstime': dateobs}
            if rsun is not None:
                kwargs['rsun'] = rsun
            if issubclass(frame, sunpy.coordinates.HeliographicCarrington):
                kwargs['observer'] = 'self'

            observer = frame(attrs[0] * u.deg,
                             attrs[1] * u.deg,
                             attrs[2] * u.m,
                             **kwargs)
            return observer


def _set_wcs_aux_obs_coord(wcs, obs_frame):
    # Ripped from sunpy
    # Sometimes obs_coord can be a SkyCoord, so convert down to a frame
    if hasattr(obs_frame, 'frame'):
        obs_frame = obs_frame.frame

    if isinstance(obs_frame, sunpy.coordinates.HeliographicStonyhurst):
        wcs.wcs.aux.hgln_obs = obs_frame.lon.to_value(u.deg)
    elif isinstance(obs_frame, sunpy.coordinates.HeliographicCarrington):
        wcs.wcs.aux.crln_obs = obs_frame.lon.to_value(u.deg)
    else:
        raise ValueError('obs_coord must be in a Stonyhurst or Carrington frame')
    # These two keywords are the same for Carrington and Stonyhurst
    wcs.wcs.aux.hglt_obs = obs_frame.lat.to_value(u.deg)
    wcs.wcs.aux.dsun_obs = obs_frame.radius.to_value(u.m)


def _observers_are_equal(obs_1, obs_2):
    # Ripped from sunpy, modified
    if obs_1 is None and obs_2 is None:
        raise ValueError("An observer must be set for this transformation.")
    
    if obs_1 is obs_2:
        return True

    # obs_1 != obs_2
    if obs_1 is None:
        return True
    if obs_2 is None:
        return True
    
    return np.atleast_1d(u.allclose(obs_1.lat, obs_2.lat) and
                         u.allclose(obs_1.lon, obs_2.lon) and
                         u.allclose(obs_1.radius, obs_2.radius) and
                         _times_are_equal(obs_1.obstime, obs_2.obstime)).all()


def _times_are_equal(time_1, time_2):
    # Ripped from sunpy, modified
    # Checks whether times are equal
    if isinstance(time_1, astropy.time.Time) and isinstance(time_2, astropy.time.Time):
        # We explicitly perform the check in TAI to avoid possible numerical precision differences
        # between a time in UTC and the same time after a UTC->TAI->UTC conversion
        return np.all(time_1.tai == time_2.tai)

    # We also deem the times equal if one is None
    return time_1 is None or time_2 is None