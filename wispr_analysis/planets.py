import os

import astropy.coordinates
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


def locate_planets(date):
    """
    Returns the helioprojective coordinates of planets as seen by PSP
    
    Parameters
    ----------
    date : ``str`` or FITS header
        The date of the observation. If a FITS header, DATE-AVG is extracted
        and used. If a string, must be in the format "YYYY-MM-DD HH:MM:SS.SSS"
    
    Returns
    -------
    planet_poses : list of ``SkyCoord``
        A Helioprojective SkyCoord for each of the eight planets, in order.
    """
    load_kernels()
    if not isinstance(date, str):
        # Treat as FITS header
        date = date['date-avg'].replace('T', ' ')
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
