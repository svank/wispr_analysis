import functools
import os

from astropy.coordinates import SkyCoord, ICRS
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
# For Helioprojective treatment
import sunpy.coordinates


# This is a subjective list of nice constellations to plot that could be used,
# but currently isn't
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
    xs, ys = wcs.world_to_pixel(coords)
    
    # When a line segment goes off the edge of the image, we want to plot it,
    # but we don't want to push out the axis bounds. So we capture and later
    # restore the current axis bounds.
    ax.autoscale()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    
    for name, pairs in constellations.items():
        x_all = []
        y_all = []
        n_good = 0
        if len(pairs) < 2:
            # Skip the really small, trivial constellations
            continue
        for start, stop in pairs:
            start = xs[start], ys[start]
            stop = xs[stop], ys[stop]
            if not np.all(np.isfinite(start)) or not np.all(np.isfinite(stop)):
                continue
            x_all.extend((start[0], stop[0]))
            y_all.extend((start[1], stop[1]))
            
            p1good = (xlim[0] <= start[0] <= xlim[1]
                      and ylim[0] <= start[1] <= ylim[1])
            p2good = (xlim[0] <= stop[0] <= xlim[1]
                      and ylim[0] <= stop[1] <= ylim[1])
            # Skip this segment if neither end is in the plot
            if p1good or p2good:
                ax.plot(
                    [start[0], stop[0]],
                    [start[1], stop[1]],
                    color='#ffe282', linewidth=.4, alpha=.7)
                n_good += 1
        if n_good > 0:
            x = np.mean(x_all)
            y = np.mean(y_all)
            text = ax.text(
                x, y,
                name,
                # Ensure the text isn't drawn outside the axis bounds
                clip_on=True,
                color='#ffe282', alpha=.7)
            # Don't let text that falls outside the axis bounds affect the size
            # of the whole figure
            text.set_in_layout(False)
            # Ensure the text is clipped at exactly the edge of the axes
            text.set_clip_box(
                mtrans.TransformedBbox(
                    mtrans.Bbox([[0,0], [1,1]]),
                    ax.transAxes))
    # Restore the axis bounds
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)