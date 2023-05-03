from .. import star_tools


from itertools import product


from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pytest
from pytest import approx


def test_load_stars(mocker):
    star_data = [''] * 43
    star_data.append('1;2 30 00;+10 30 00;2')
    star_data.append('2;2 30 30;+10 30 30;5')
    star_data.append('3;12 30 30;-10 30 30;10')
    star_data.append('4;8 00 00;-01 00 00;10')
    star_data.append('')
    
    class MockFile():
        def readlines(self):
            return star_data
    
    mocker.patch(star_tools.__name__+'.open',
            return_value=MockFile())
    
    star_cat = star_tools.load_stars()
    
    data = star_cat.get_stars(2.5/24*360, 10.5)
    assert len(data) == 2
    assert data[0] == (2.5/24*360, 10.5, 2)
    assert data[1] == ((2.5 + .5/60)/24*360, 10.5 + .5/60, 5)
    
    data = star_cat.get_stars(12.4/24*360, -10.6)
    assert len(data) == 1
    assert data[0] == ((12.5 + .5/60)/24*360, -10.5 - .5/60, 10)
    
    data = star_cat.get_stars(0, 0)
    assert len(data) == 0
    
    data = list(star_cat.stars_between([(2/24*360, 3/24*360)], 10, 11))
    assert len(data) == 2
    assert data[0] == (2.5/24*360, 10.5, 2)
    assert data[1] == ((2.5 + .5/60)/24*360, 10.5 + .5/60, 5)
    
    data = list(star_cat.stars_between(
        [(2/24*360, 3/24*360), (11/24*360, 13/24*360)], -11, 11))
    assert len(data) == 3
    assert data[0] == (2.5/24*360, 10.5, 2)
    assert data[1] == ((2.5 + .5/60)/24*360, 10.5 + .5/60, 5)
    assert data[2] == ((12.5 + .5/60)/24*360, -10.5 - .5/60, 10)
    
    data = list(star_cat.stars_between(
        [(2/24*360, 13/24*360)], -11, 11))
    assert len(data) == 4
    assert data[0] == (2.5/24*360, 10.5, 2)
    assert data[1] == ((2.5 + .5/60)/24*360, 10.5 + .5/60, 5)
    assert data[2] == (8/24*360, -1, 10)
    assert data[3] == ((12.5 + .5/60)/24*360, -10.5 - .5/60, 10)

