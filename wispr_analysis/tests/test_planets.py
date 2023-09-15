from .. import planets, utils

from datetime import datetime
import os

from astropy.io import fits
import pytest


def test_locate_planets_cache(tmp_path, mocker):
    locate_planets_orig = planets.locate_planets
    
    t = utils.to_timestamp('psp_L3_wispr_20181101T004548_V3_1221.fits')
    mocker.patch(
            planets.__name__ + ".locate_planets",
            return_value='first_timestamp')
    planets.cache_planet_pos(t, cache_dir=tmp_path)
    
    t2 = t + 20
    mocker.patch(
            planets.__name__ + ".locate_planets",
            return_value='second_timestamp')
    planets.cache_planet_pos(t2, cache_dir=tmp_path)
    
    planets.locate_planets = locate_planets_orig
    
    assert planets.locate_planets(t, cache_dir=tmp_path) == 'first_timestamp'
    assert planets.locate_planets(t2, cache_dir=tmp_path) == 'second_timestamp'


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
