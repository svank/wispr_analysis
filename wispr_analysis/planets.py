from datetime import datetime, timezone
import os
import pickle

import astropy.coordinates
import numpy as np
import spiceypy as spice
import sunpy.coordinates


KERNELS_LOADED = False
planets = [
        'Mercury', 'Venus', 'Earth', 'Mars',
        'Jupiter', 'Saturn', 'Uranus', 'Neptune']


def load_kernels(kernel_dir='spice_kernels'):
    """
    Recursively scans a directory and loading each file as a SPICE kernel.
    
    Does nothing if kernels have already been loaded.
    
    Parameters
    ----------
    kernel_dir : str
        The path to search for kernels.
    """
    global KERNELS_LOADED
    if KERNELS_LOADED:
        return
    for root, _, files in os.walk(kernel_dir):
        for kern in files:
            spice.furnsh(os.path.join(root, kern))
    KERNELS_LOADED = True


def _to_hp(planet_pos, sc_pos, date):
    sc = astropy.coordinates.SkyCoord(
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
            observer=sc,
            obstime=date)
    return c.transform_to(sunpy.coordinates.frames.Helioprojective)


def locate_planets(date, cache_dir=None):
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
    cache_dir : ``str``
        An optional directory to search for cached positions, as saved by
        `cache_planet_pos`.
    
    Returns
    -------
    planet_poses : list of ``SkyCoord``
        A Helioprojective SkyCoord for each of the eight planets, in order.
    """
    date = format_date(date)
    
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, str(date))
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    
    load_kernels()
    et = spice.str2et(date)
    
    spacecraft_id = '-96'
    state, ltime = spice.spkezr(spacecraft_id, et, 'HCI', 'None', 'Sun')
    sc_pos = state[:3]
    
    planet_poses = []
    for planet in planets:
        if planet not in ("Mercury", "Venus", "Earth"):
            planet = planet + " Barycenter"
        state, ltime = spice.spkezr(planet, et, 'HCI', 'None', 'Sun')
        planet_poses.append(_to_hp(state[:3], sc_pos, date))
    return planet_poses


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
        and used. If a string, passed through unaltered. If a number,
        interpreted as a UTC timestamp.
    Returns
    -------
    date : ``str``
        The date in "YYYY-MM-DD HH:MM:SS.SSS" format
    """
    if isinstance(date, (int, float)):
        date = datetime.fromtimestamp(
            date, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    elif not isinstance(date, str):
        # Treat as FITS header
        date = date['date-avg'].replace('T', ' ')
    return date


def get_orbital_plane(body, date, observer=None):
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
    
    times = et * 3600*24 + np.linspace(-period//2, period//2, 720)
    
    coords = []
    for t in times:
        state = spice.conics(elts, t)
        coords.append(state[:3])
    coords = np.array(coords)
    
    if observer is not None:
        if body == '-96':
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
    return coords
