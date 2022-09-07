from .. import synthetic_data as sd


import matplotlib.pyplot as plt
from matplotlib.testing.conftest import mpl_test_settings
from matplotlib.testing.decorators import image_comparison
import numpy as np
import pytest
from pytest import approx


def test_signed_and_unsigned_angle_between_vectors():
    data = (((1, 0), (0, 1), np.pi/2),
            ((0, -1), (1, 0), np.pi/2),
            ((1, 1), (0, 1), np.pi/4),
            ((1, 0), (1, 1), np.pi/4),
            ((0, 1), (0, -1), np.pi),
            ((1, 0), (-1, 0), np.pi),
            ((0, 1), (-1, 0), np.pi/2),
            ((0, 1), (-1, -1), np.pi/2 + np.pi/4))
    
    for v1, v2, answer in data:
        assert sd.signed_angle_between_vectors(*v1, *v2) == approx(answer)
        assert sd.signed_angle_between_vectors(*v2, *v1) == approx(-
                answer if answer != np.pi else answer)
        assert sd.angle_between_vectors(*v1, *v2) == approx(answer)
        assert sd.angle_between_vectors(*v2, *v1) == approx(answer)
    
    assert sd.signed_angle_between_vectors(1, 1, 1, 1) == approx(0, abs=2e-8)


def test_angle_between_vectors_exception_on_zero_vector():
    with pytest.raises(ValueError, match="zero-vector"):
        sd.signed_angle_between_vectors(0, 0, 1, 1)
    with pytest.raises(ValueError, match="zero-vector"):
        sd.signed_angle_between_vectors(1, 1, 0, 0)
    with pytest.raises(ValueError, match="zero-vector"):
        sd.signed_angle_between_vectors(0, 0, 1, 1)
    with pytest.raises(ValueError, match="zero-vector"):
        sd.signed_angle_between_vectors(1, 1, 0, 0)


def test_Thing_at():
    for x in np.linspace(-20, 20, 10):
        for y in np.linspace(-20, 20, 10):
            for vx in np.linspace(-10, 20, 10):
                for vy in np.linspace(-20, 10, 10):
                    for t in np.linspace(-3, 3, 11):
                        thing = sd.Thing(x=x, y=y, vx=vx, vy=vy).at(t)
                        assert thing.vx == vx
                        assert thing.vy == vy
                        assert thing.x == x + vx * t
                        assert thing.y == y + vy * t


def test_Thing_in_front():
    # Run a grid of velocity directions
    for vx in [-1, 0, 1]:
        for vy in [-1, 0, 1]:
            # Things should not change over time, the way they're set up here
            for t in np.linspace(-10, 10, 5):
                sc = sd.Thing(x=0, y=0, vx=vx, vy=vy)
                if vx == 0 and vy == 0:
                    # No velocity direction
                    continue
                # Make in-front and behind copies of our object
                sc_infront = sc.at(1)
                sc_behind = sc.at(-1)
                assert sc_infront.in_front_of(sc, t)
                assert not sc_behind.in_front_of(sc, t)
                
                # Offset the in-front and behind objects perpendicular to the
                # main object's velocity vector---should not change
                # in-front-ness
                for delta in np.logspace(-2, 2, 11):
                    sc_offset = sd.Thing(x=sc_infront.x + delta * -vy,
                                      y=sc_infront.y + delta * vx)
                    assert sc_offset.in_front_of(sc, 0)
                    sc_offset = sd.Thing(x=sc_infront.x + delta * vy,
                                      y=sc_infront.y + delta * -vx)
                    assert sc_offset.in_front_of(sc, 0)
                    
                    sc_offset = sd.Thing(x=sc_behind.x + delta * -vy,
                                      y=sc_behind.y + delta * vx)
                    assert not sc_offset.in_front_of(sc, 0)
                    sc_offset = sd.Thing(x=sc_behind.x + delta * vy,
                                      y=sc_behind.y + delta * -vx)
                    assert not sc_offset.in_front_of(sc, 0)
                
                # Have the behind object catch up, test passing t0 value
                sc_behind.vx *= 2
                sc_behind.vy *= 2
                assert not sc_behind.in_front_of(sc)
                assert sc_behind.in_front_of(sc, 2)
                
                # Fix the copied objects, continue testing passing t0 value
                sc_infront.vx = 0
                sc_infront.vy = 0
                sc_behind.vx = 0
                sc_behind.vy = 0
                
                assert sc_infront.in_front_of(sc, 0)
                assert sc_infront.in_front_of(sc, -1)
                assert not sc_infront.in_front_of(sc, 2)
                
                assert not sc_behind.in_front_of(sc, 0)
                assert sc_behind.in_front_of(sc, -2)
                assert not sc_behind.in_front_of(sc, 2)


def test_Thing_subtract():
    t1 = sd.Thing()
    t2 = sd.Thing(x=1, y=1, vx=10, vy=10)
    assert (t2-t1).r == np.sqrt(2)
    assert (t2-t1).at(1).r == np.sqrt(11**2 + 11**2)
    assert (t2-t1).at(-1).r == np.sqrt(9**2 + 9**2)
    assert (t2-t1).x == 1
    assert (t2-t1).y == 1
    assert (t2-t1).vx == 10
    assert (t2-t1).vy == 10
    
    t1 = sd.Thing(vx=10, vy=-10)
    assert (t2-t1).r == np.sqrt(2)
    assert (t2-t1).at(1).x == 1
    assert (t2-t1).at(1).y == 21


def test_elongation_to_FOV():
    sc = sd.Thing(x=0, y=-1, vx=0, vy=1)
    # Elongation is FOV
    elongations = np.linspace(0, 180) * np.pi / 180
    fovs = sd.elongation_to_FOV(sc, elongations)
    assert np.all(np.isclose(elongations, fovs))
    
    sc.vy = 0
    sc.vx = 1
    # FOV runs from -90 to 90 degrees
    fovs = sd.elongation_to_FOV(sc, elongations)
    assert np.all(np.isclose(np.linspace(-np.pi/2, np.pi/2), fovs))
    
    sc.vx = -1
    # FOV runs from 90 to -90 degrees
    fovs = sd.elongation_to_FOV(sc, elongations)
    assert np.all(np.isclose(np.linspace(np.pi/2, -np.pi/2), fovs))
    
    # Test vectorizing over time
    sc = sd.Thing(x=0, y=-1, vx=1).at(np.array([-1, 0, 1]))
    elongations = np.array([0, 0, 0])
    fovs = sd.elongation_to_FOV(sc, elongations)
    assert np.all(np.isclose([-np.pi/4, -np.pi/2, -np.pi/2 - np.pi/4], fovs))



@pytest.mark.parametrize("sc_vx_sign", [-1, 1])
def test_calculate_radiant(sc_vx_sign):
    sc = sd.Thing(x=0, y=-10, vx=sc_vx_sign * 1, vy=0)
    p = sc.at(0)
    p.vy = -1
    p.vx = 0
    p = p.at(-1)
    sc = sc.at(-1)
    
    ts = np.linspace(0, .9)
    radiants = sd.calculate_radiant(sc, p, t0=ts)
    # Convert to an FOV position so the values are nearly constant and easy to
    # check.
    radiants = sd.elongation_to_FOV(sc.at(ts), radiants)
    np.testing.assert_allclose(radiants, -1 * sc_vx_sign * np.pi/4, atol=0.006)
    
    np.testing.assert_array_equal(
            sd.calculate_radiant(sc.at(1e10), p),
            np.nan)


def test_calc_epsilon():
    sc = sd.Thing(x=-10, y=-10, vx=0, vy=0)
    p = sd.Thing(x=-5, y=-5, vx=0, vy=-5)
    
    assert sd.calc_epsilon(sc, p) == approx(0)
    assert sd.calc_epsilon(sc, p, t=1) == approx(np.pi/4)
    
    p.vy = 0
    sc.vy = 5
    sc.vx = 10
    
    assert sd.calc_epsilon(sc, p, t=1) == approx(np.pi/2)
    
    np.testing.assert_allclose(
            sd.calc_epsilon(sc, p, t=[0, 1]),
            [0, np.pi/2])


def test_FOV_pos():
    sc = sd.Thing(x=-10, y=-10, vx=0.000001, vy=0)
    p = sd.Thing(x=0, y=-10, vx=0, vy=0)
    
    assert sd.calc_FOV_pos(sc, p) == approx(0)
    
    p.y = 0
    p.vy = -20
    
    np.testing.assert_allclose(
            sd.calc_FOV_pos(sc, p, t=[0, 1]),
            [-np.pi/4, np.pi/4])
    
    sc.vx *= -1
    sc.x *= -1
    
    np.testing.assert_allclose(
            sd.calc_FOV_pos(sc, p, t=[0, 1]),
            [np.pi/4, -np.pi/4])


@image_comparison(baseline_images=['test_synthesize_image'],
        extensions=['pdf'])
def test_synthesize_image():
    sc = sd.Thing(x=-10, y=-10, vx=1, vy=0.2)
    p = sd.Thing(x=-5, y=-12, vx=-.5, vy=-.5)
    p2 = sd.Thing(x=5, y=-8, vx=.5, vy=.5)
    p3 = sd.Thing(x=-20, y=-20, vx=.5, vy=.5)
    
    image, wcs = sd.synthesize_image(sc, [p, p2], 1, fov=140)
    
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    ax.imshow(np.sqrt(image), origin='lower', cmap='Greys_r')
    
    lon, lat = ax.coords
    lat.set_major_formatter('dd')
    lon.set_major_formatter('dd')
    ax.set_xlabel("HP Longitude")
    ax.set_ylabel("HP Latitude")
    ax.coords.grid(color='white', alpha=0.7, ls='-')
    
    # Check that the range of values is right
    assert image.max() == approx(0.000456832948092)
    assert image.min() == approx(0)
    
    return fig
