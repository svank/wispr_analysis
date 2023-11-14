from .. import planets, utils

import os

from astropy.io import fits
import pytest
import numpy as np
import spiceypy as spice


@pytest.mark.parametrize('only', [None, 'Venus', ['Venus'], ['Earth', 'Venus']])
def test_locate_planets_cache(tmp_path, mocker, only):
    planets.load_kernels(os.path.join(utils.data_path(), 'spice_kernels'))
    
    real_spkezr = spice.spkezr
    def spkezr(id, et, *args, **kwargs):
        if id == '-96':
            return (
                    np.array([2.66925117e+07, -9.53139787e+07, 6.40231055e+06,
                              1.48270646e+01, 2.61921828e+01, -1.77890244e+00]),
                    330.85513358725734)
        # Pass through for planets
        return real_spkezr(id, et, *args, **kwargs)
    mocker.patch(
            planets.__name__+'.spice.spkezr',
            wraps=spkezr)
    
    t = utils.to_timestamp('psp_L3_wispr_20181101T004548_V3_1221.fits')
    r1 = planets.locate_planets(t, cache_dir=tmp_path, only=only)
    
    planets.clear_kernels()
    
    r2 = planets.locate_planets(t, cache_dir=tmp_path, only=only)
    
    assert r1 == r2


def test_formate_date():
    path = os.path.join(utils.test_data_path(),
            'WISPR_files_headers_only',
            '20181101',
            'psp_L3_wispr_20181101T013048_V3_1221.fits')
    
    with utils.ignore_fits_warnings():
        header = fits.getheader(path)
    
    # Just hack this up so the date read from the header matches the date in
    # the filename
    header['date-avg'] = header['date-beg'].split('.')[0]
    t1 = planets.format_date(header)
    t2 = planets.format_date(utils.to_timestamp(path))
    
    assert t1 == t2
    
    assert planets.format_date('hi') == 'hi'
    
    assert planets.format_date('E07') == '2021-01-17 17:40:00'
    assert planets.format_date(7) == '2021-01-17 17:40:00'


def test_get_psp_orbit_number():
    assert planets.get_psp_orbit_number('2018-11-06 03:27:00') == 1
    assert planets.get_psp_orbit_number('2018-11-03 03:27:00') == 1
    assert planets.get_psp_orbit_number('2018-11-09 03:27:00') == 1
    assert planets.get_psp_orbit_number('2019-04-04 03:27:00') == 2
    
    with pytest.raises(ValueError):
        planets.get_psp_orbit_number('2018-10-30 03:27:00')
    with pytest.raises(ValueError):
        planets.get_psp_orbit_number('2025-07-31 03:27:00')