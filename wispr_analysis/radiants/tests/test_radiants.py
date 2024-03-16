import os

import astropy.units as u
import numpy as np
import pytest
import scipy.ndimage

from .. import radiants, synthetic_data as sd
from ... import planets, utils


def generate_strips():
    # A cloud of parcels
    sc = sd.LinearThing(x=-6, y=-100, vx=3)
    ts = np.linspace(0, 6, 50)

    parcels = []
    np.random.seed(4)
    n_parcels = 30
    for dx, dy in zip(np.random.random(n_parcels), np.random.random(n_parcels)):
        dx = (dx - .5) * 2 * 17.5
        dy = (dy - .5) * 2 * 9
        v = 1.5
        x = 11 + dx
        y = -92 + dy
        r = np.sqrt(x**2 + y**2)
        vx = x / r * v
        vy = y / r * v
        parcels.append(sd.LinearThing(x=x, y=y, vy=vy, vx=vx))
        
    fov = 70
    strips = []
    for t in ts:
        strip, _ = sd.synthesize_image(sc, parcels, t, output_size_x=45,
                output_size_y=1, projection='CAR', fov=fov, parcel_width=1*u.m)
        strip = scipy.ndimage.gaussian_filter(strip, 0.7)
        strips.append(strip)
    strips = np.vstack(strips)
    
    fov_angles = np.linspace(-fov/2, fov/2, strips.shape[1])
    return strips, ts, fov_angles


@pytest.mark.array_compare(file_format='fits', atol=0.25)
def test_get_speeds():
    strips, _, _ = generate_strips()
    
    kwargs = dict(apodize_rolloff=5)
    v_d, fstrips_d = radiants.get_speeds(strips, **kwargs,
            dx=2, dt=3)
    v_a, fstrips_a = radiants.get_speeds(strips, **kwargs,
            spatial_axis=np.array([2, 4, 6]), temporal_axis=np.array([3, 6, 9]))
    
    np.testing.assert_array_equal(v_d, v_a)
    np.testing.assert_array_equal(fstrips_d, fstrips_a)
    
    return np.stack((v_d, fstrips_d.real, fstrips_d.imag))


@pytest.mark.array_compare()
def test_select_speed_range():
    strips, _, _ = generate_strips()
    
    kwargs = dict(apodize_rolloff=3, filter_rolloff=5)
    filtered_strips_d = radiants.select_speed_range(0, 50, strips, dx=2, dt=3,
            **kwargs)
    filtered_strips_a = radiants.select_speed_range(0, 50, strips, 
            spatial_axis=np.array([2, 4, 6]), temporal_axis=np.array([3, 6, 9]),
            **kwargs)
    
    np.testing.assert_array_equal(filtered_strips_d, filtered_strips_a)
    
    return filtered_strips_a


def test_find_radiant():
    strips, ts, fov_angles = generate_strips()
    rads, rad_ts = radiants.find_radiant(strips, ts, fov_angles,
            window_size=43, v_halfwindow=1.1)
    
    np.testing.assert_array_equal(ts[21:-21], rad_ts)
    
    np.testing.assert_allclose(rads,
            np.array([-1.5909090909090935,
                      -1.5909090909090935,
                      -1.5909090909090935,
                      -1.5909090909090935,
                      -1.5909090909090935,
                      -1.5909090909090935,
                      -1.5909090909090935,
                      -1.5909090909090935]))


def test_calc_elongation_radiant():
    assert (radiants.calc_elongation_radiant(10, 200, 150)
            == pytest.approx(5.7))
    assert (radiants.calc_elongation_radiant(30, 200, 150)
            == pytest.approx(17.2))
    assert (radiants.calc_elongation_radiant(70, 200, 150)
            == pytest.approx(40.7))
    assert (radiants.calc_elongation_radiant(100, 200, 150)
            == pytest.approx(59.65))


def test_calc_fixed_angle_radiant(mocker):
    planets.load_kernels(utils.data_path('spice_kernels'))
    mocker.patch(radiants.__name__+'.planets.spice.spkezr',
                return_value=(
                    np.array([2.66925117e+07, -9.53139787e+07, 6.40231055e+06,
                              1.48270646e+01, 2.61921828e+01, -1.77890244e+00]),
                    330.85513358725734))
    file = utils.test_data_path(
            'WISPR_files_with_data_half_size', '20181101',
            'psp_L2_wispr_20181101T004548_V3_1221.fits')
    assert radiants.calc_fixed_angle_radiant(
        [file], 200) == pytest.approx(-26.89428378)