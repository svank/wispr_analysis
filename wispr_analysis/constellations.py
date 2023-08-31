import functools
import os

from astropy.coordinates import SkyCoord, ICRS
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
# For Helioprojective treatment
import sunpy.coordinates


GOOD_CONSTELLATIONS = [
    'Andromeda',
    'Aquarius',
    'Aquila',
    'Aries',
    'Bootes',
    'Cancer',
    'Carina',
    'Canis Major',
    'Capricornus',
    'Cassiopeia',
    'Centaurus',
    'Cetus',
    'Cygnus',
    'Draco',
    'Eridanus',
    'Gemini',
    'Hercules',
    'Hydra',
    'Leo',
    'Libra',
    'Lupus',
    'Monoceros',
    'Ophiuchus',
    'Orion',
    'Pegasus',
    'Phoenix',
    'Pisces',
    'Sagittarius',
    'Scorpius',
    'Serpens',
    'Taurus',
    'Ursa Major',
    'Ursa Minor',
    'Virgo',
    ]


@functools.cache
def load_constellation_data():
    data_path = os.path.join(
        os.path.dirname(__file__), "data")
    
    name_dat = open(
        os.path.join(data_path, "constellation_names.eng.fab")).readlines()
    # Maps constellation abbreviations to full names
    name_map = {line.split()[0]: line.split('"')[1] for line in name_dat}

    star_dat = open(
        os.path.join(data_path, "hipparchos_catalog.tsv")).readlines()
    # Will map Hipparchos IDs to (RA, Dec)
    star_map = {}
    for line in star_dat[43:-1]:
        try:
            id, RA, dec, Vmag = line.split(";")
        except ValueError:
            continue
        try:
            if float(Vmag) > 6.7:
                continue
        except ValueError:
            continue
        star_map[id.strip()] = (RA, dec)

    line_dat = open(
        os.path.join(data_path, "constellationship.fab")).readlines()
    # Will map constellation name to a list of star pairs defining each line
    line_dict = {}
    # We'll keep all the RAs and decs we need for constellations here, so later
    # we can run all the coordinate transformations at once (much faster!)
    RAs, decs = [], []
    # This maps a Hipparcos ID to a position in the RA/dec lists
    id_to_elem = {}
    for const in line_dat:
        pieces = const.split()
        name = name_map[pieces[0]]
        if name not in GOOD_CONSTELLATIONS:
            continue
        star_ids = pieces[2:]
        for id in star_ids:
            if id not in id_to_elem:
                id_to_elem[id] = len(RAs)
                RA, dec = star_map[id]
                RAs.append(RA)
                decs.append(dec)
        star_elem_ids = [id_to_elem[id] for id in star_ids]
        line_dict[name] = list(zip(star_elem_ids[::2], star_elem_ids[1::2]))
    RAs = np.array(RAs)
    decs = np.array(decs)
    distances = np.full(len(RAs), 1e10)
    coords = SkyCoord(
        RAs, decs, distances, frame=ICRS, unit=(u.hourangle, u.deg, u.au))
    return line_dict, coords
    
    
def plot_constellations(wcs, ax=None):
    if ax is None:
        ax = plt.gca()
    
    constellations, coords = load_constellation_data()
    x, y = wcs.world_to_pixel(coords)
    
    ax.autoscale()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    
    for name, pairs in constellations.items():
        x_used = []
        y_used = []
        for start, stop in pairs:
            start = x[start], y[start]
            stop = x[stop], y[stop]
            
            xgood = (xlim[0] <= start[0] <= xlim[1]
                     or xlim[0] <= stop[0] <= xlim[1])
            ygood = (ylim[0] <= start[1] <= ylim[1]
                     or ylim[0] <= stop[1] <= ylim[1])
            if xgood and ygood:
                ax.plot(
                    [start[0], stop[0]],
                    [start[1], stop[1]],
                    color='#ffe282', linewidth=.4, alpha=.7)
                x_used.extend((start[0], stop[0]))
                y_used.extend((start[1], stop[1]))
        if len(x_used) > 6:
            plt.text(
                np.mean(x_used),
                np.mean(y_used),
                name,
                color='#ffe282', alpha=.7)
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)