from datetime import datetime, timezone
import os
import pickle

import astropy.coordinates
import astropy.units as u
import numpy as np
import spiceypy as spice
import sunpy.coordinates

from . import utils


KERNELS_LOADED = False
planets = [
        'Mercury', 'Venus', 'Earth', 'Mars',
        'Jupiter', 'Saturn', 'Uranus', 'Neptune']


perihelia_dates = {
    1: '2018-11-06 03:27:00',
    2: '2019-04-04 22:39:00',
    3: '2019-09-01 17:50:00',
    4: '2020-01-29 09:37:00',
    5: '2020-06-07 08:23:00',
    6: '2020-09-27 09:16:00',
    7: '2021-01-17 17:40:00',
    8: '2021-04-29 08:48:00',
    9: '2021-08-09 19:11:00',
    10: '2021-11-21 08:23:00',
    11: '2022-02-25 15:38:00',
    12: '2022-06-01 22:51:00',
    13: '2022-09-06 06:04:00',
    14: '2022-12-11 13:16:00',
    15: '2023-03-17 20:30:00',
    16: '2023-06-22 03:46:00',
    17: '2023-09-27 23:28:00',
    18: '2023-12-29 00:54:00',
    19: '2024-03-30 02:20:00',
    20: '2024-06-30 03:46:00',
    21: '2024-09-30 05:13:00',
    22: '2024-12-24 11:41:00',
    23: '2025-03-22 22:25:00',
    24: '2025-06-19 09:09:00'}


def get_psp_perihelion_date(encounter):
    if isinstance(encounter, str):
        if encounter[0] == 'E':
            encounter = encounter[1:]
        encounter = int(encounter)
    return perihelia_dates[encounter]


def get_psp_orbit_number(date):
    ts = utils.to_timestamp(date)
    if ts < utils.to_timestamp('2018-11-01 00:00:00'):
        raise ValueError('Date is before early cutoff')
    if ts > utils.to_timestamp('2025-07-01 00:00:00'):
        raise ValueError('Date is before late cutoff')
    timestamps = utils.to_timestamp(perihelia_dates.values())
    delta = np.abs(np.array(timestamps) - ts)
    i = np.argmin(delta)
    return list(perihelia_dates.keys())[i]


def load_kernels(kernel_dir='spice_kernels', force=False):
    """
    Recursively scans a directory and loading each file as a SPICE kernel.
    
    Does nothing if kernels have already been loaded.
    
    Parameters
    ----------
    kernel_dir : str
        The path to search for kernels.
    """
    global KERNELS_LOADED
    if KERNELS_LOADED and not force:
        return
    for root, _, files in os.walk(kernel_dir):
        for kern in files:
            spice.furnsh(os.path.join(root, kern))
    KERNELS_LOADED = True


def clear_kernels():
    spice.kclear()
    global KERNELS_LOADED
    KERNELS_LOADED = False


def _to_hp(planet_pos, sc_pos, date):
    if not isinstance(sc_pos, astropy.coordinates.SkyCoord):
        sc_pos = astropy.coordinates.SkyCoord(
                *sc_pos,
                frame=sunpy.coordinates.frames.HeliocentricInertial,
                representation_type='cartesian',
                unit='km',
                obstime=date)
    c = astropy.coordinates.SkyCoord(
            *planet_pos,
            frame=sunpy.coordinates.frames.HeliocentricInertial,
            representation_type='cartesian',
            unit='km',
            observer=sc_pos,
            obstime=date)
    return c.transform_to(sunpy.coordinates.frames.Helioprojective)


def locate_planets(date, only=None, cache_dir=None, sc_pos=None):
    """
    Returns the helioprojective coordinates of planets as seen by PSP
    
    Parameters
    ----------
    date : ``str`` or FITS header or ``float``
        The date of the observation. If a FITS header, DATE-AVG is extracted
        and used. If a string, must be in the format "YYYY-MM-DD HH:MM:SS.SSS".
        If a number, interpreted as a UTC timestamp. Note that the timestamp in
        WISPR images is the beginning time, not the average time, so providing
        the FITS header is preferred.
    only : ``list`` or ``str``
        If provided, a planet or list of specific planets to locate (otherwise,
        all planets are located). Should be planet names.
    cache_dir : ``str``
        An optional directory to search for cached positions, as saved by
        `cache_planet_pos`.
    
    Returns
    -------
    planet_poses : list of ``SkyCoord``
        A Helioprojective SkyCoord for each of the eight planets, in order.
    """
    if only is None:
        only = planets
    if isinstance(only, str):
        only = [only]
    only = set(planet.lower() for planet in only)
    
    date = format_date(date)
    
    cache_fname = f"locate_planets-{date}-{only}.pkl"
    if cache_dir is not None and sc_pos is None:
        cache_path = os.path.join(cache_dir, cache_fname)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    
    load_kernels()
    et = spice.str2et(date)
    
    sc_pos_orig = sc_pos
    if sc_pos is None:
        spacecraft_id = '-96'
        state, _ = spice.spkezr(spacecraft_id, et, 'HCI', 'None', 'Sun')
        sc_pos = state[:3]
    
    planet_poses = []
    for planet in planets:
        if planet.lower() not in only:
            continue
        if planet not in ("Mercury", "Venus", "Earth"):
            planet = planet + " Barycenter"
        state, _ = spice.spkezr(planet, et, 'HCI', 'None', 'Sun')
        planet_poses.append(_to_hp(state[:3], sc_pos, date))
    
    if cache_dir is not None and sc_pos_orig is None:
        cache_path = os.path.join(cache_dir, cache_fname)
        with open(cache_path, 'wb') as f:
            pickle.dump(planet_poses, f)
    return planet_poses


def locate_psp(date, cache_dir=None):
    """
    Returns the heliocentric coordinate and velocity of PSP
    
    Parameters
    ----------
    date : ``str`` or FITS header or ``float``
        The date of the observation. If a FITS header, DATE-AVG is extracted
        and used. If a string, must be in the format "YYYY-MM-DD HH:MM:SS.SSS".
        If a number, interpreted as a UTC timestamp. Note that the timestamp in
        WISPR images is the beginning time, not the average time, so providing
        the FITS header is preferred.
    
    Returns
    -------
    psp_pos : ``SkyCoord``
        A HeliocentricInertial SkyCoord for PSP, with velocity information.
    """
    date = format_date(date)
    
    cache_fname = f"locate_psp-{date}.pkl"
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, cache_fname)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    
    load_kernels()
    et = spice.str2et(date)
    
    state, _ = spice.spkezr('-96', et, 'HCI', 'None', 'Sun')
    x, y, z = state[:3] * u.km
    vx, vy, vz = state[3:] * u.km / u.s
    psp_pos = astropy.coordinates.SkyCoord(x=x, y=y, z=z,
                                           v_x=vx, v_y=vy, v_z=vz,
                                           representation_type='cartesian',
                                           frame='heliocentricinertial',
                                           obstime=date)
    
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, cache_fname)
        with open(cache_path, 'wb') as f:
            pickle.dump(psp_pos, f)
    
    return psp_pos


def cache_planet_pos(date, cache_dir):
    """
    Computes and caches planet positions
    
    Parameters
    ----------
    date : ``str`` or FITS header or ``float``
        The date of the observation. If a FITS header, DATE-AVG is extracted
        and used. If a string, must be in the format "YYYY-MM-DD HH:MM:SS.SSS".
        If a number, interpreted as a UTC timestamp.
    cache_dir : ``str``
        The directory to in which to save cached positions.
    """
    date = format_date(date)
    
    planet_poses = locate_planets(date)
    
    with open(os.path.join(cache_dir, str(date)), 'wb') as f:
        pickle.dump(planet_poses, f)


def format_date(date):
    """
    Parses a date to the format required by SPICE
    
    Parameters
    ----------
    date : ``str`` or FITS header or ``float``
        The date of the observation. If a FITS header, DATE-AVG is extracted
        and used. If a string, passed through unaltered, unless the string is
        'E##', then it is interpreted as the perihelion date for that
        encounter. If a number, interpreted as an encounter number if an
        ``int`` < 30, or else as a UTC timestamp.
    
    Returns
    -------
    date : ``str``
        The date in "YYYY-MM-DD HH:MM:SS.SSS" format
    """
    if isinstance(date, (int, float)):
        if isinstance(date, int) and date < 30:
            try:
                date = get_psp_perihelion_date(date)
            except KeyError:
                raise ValueError("Invalid encounter number")
        else:
            date = datetime.fromtimestamp(
                date, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(date, str) and date[0] == 'E' and len(date) == 3:
        try:
            date = get_psp_perihelion_date(date)
        except KeyError:
            raise ValueError("Invalid encounter number")
    elif not isinstance(date, str):
        # Treat as FITS header
        date = date['date-avg'].replace('T', ' ')
    return date


def get_orbital_plane(body, date, observer=None, return_times=False, expand_psp_orbit=True, npts=720):
    """
    Generates coordinates of a set of points along an orbital plane
    
    Parameters
    ----------
    body : ``str``
        The body whose orbital plane should be found. Provide a body name
        recognized by SPICE, or 'PSP'.
    date
        A point in time to start from. Should be anything recognized by
        `format_date`.
    observer : array
        The (x,y,z) coordinates of the observer. If given, returned coordinates
        are Helioprojective. If not given, returned coordinates are
        HeliocentricIntertial.
    return_times : ``bool``
        If True, the times of each point in the orbit are returned in addition
        to the points themselves.
    expand_psp_orbit : ``bool``
        Only takes effect if ``body`` and ``observer`` are both 'PSP'. If True,
        the PSP orbit is expanded radially so that the helioprojective
        coordinates of PSP's orbit near PSP come out sensibly.
    npts : ``int``
        Number of points along the orbit to calculate
    """
    date = format_date(date)
    load_kernels()
    if body.lower() == 'psp':
        body = '-96'
    et = spice.str2et(date)
    state, ltime = spice.spkezr('-96', et, 'HCI', 'None', 'Sun')
    
    mu = 1.32712440018e11
    elts = spice.oscelt(state, et, mu)
    
    a = elts[0] / (1 - elts[1])
    period = 2*np.pi * np.sqrt(a**3 / mu)
    
    times = et + np.linspace(-period//2, period//2, npts)
    
    coords = []
    for t in times:
        state = spice.conics(elts, t)
        coords.append(state[:3])
    coords = np.array(coords)
    
    if observer is not None:
        if body == '-96' and expand_psp_orbit:
            # Expand the orbit so the projected plane comes out right
            c = astropy.coordinates.CartesianRepresentation(*coords.T)
            s = c.represent_as(astropy.coordinates.SphericalRepresentation)
            s2 = astropy.coordinates.SphericalRepresentation(
                    lon=s.lon, lat=s.lat, distance=4*s.distance)
            c2 = s2.represent_as(astropy.coordinates.CartesianRepresentation)
            coords = np.array([c2.x, c2.y, c2.z]).T
        
        coords = _to_hp(coords.T, observer, date)
    else:
        coords = astropy.coordinates.SkyCoord(
                *coords.T,
                frame=sunpy.coordinates.frames.HeliocentricInertial,
                representation_type='cartesian',
                unit='km',
                obstime=date)
    if return_times:
        times = np.array([spice.et2datetime(t).timestamp() for t in times])
        return coords, times
    return coords
