import astropy.units as u
import numpy as np
import pytest
import scipy.ndimage

from .. import radiants, synthetic_data as sd


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


@pytest.mark.array_compare(file_format='fits', atol=5e-19)
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
    
    np.testing.assert_array_equal(rads,
            np.array([-1.5909090909090935,
                      -1.5909090909090935,
                      -1.5909090909090935,
                      -1.5909090909090935,
                      -1.5909090909090935,
                      -1.5909090909090935,
                      -1.5909090909090935,
                      -1.5909090909090935]))
