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


def test_angle_between_vectors_zero_vector():
    assert np.isnan(sd.signed_angle_between_vectors(0, 0, 1, 1))
    assert np.isnan(sd.signed_angle_between_vectors(1, 1, 0, 0))
    assert np.isnan(sd.angle_between_vectors(0, 0, 1, 1))
    assert np.isnan(sd.angle_between_vectors(1, 1, 0, 0))


def test_elongation_to_FOV():
    sc = sd.LinearThing(x=0, y=-1, vx=0, vy=1)
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
    sc = sd.LinearThing(x=0, y=-1, vx=1).at(np.array([-1, 0, 1]))
    elongations = np.array([0, 0, 0])
    fovs = sd.elongation_to_FOV(sc, elongations)
    assert np.all(np.isclose([-np.pi/4, -np.pi/2, -np.pi/2 - np.pi/4], fovs))


@pytest.mark.parametrize("sc_vx_sign", [-1, 1])
def test_calculate_radiant(sc_vx_sign):
    # Create two objects which will collide
    sc = sd.LinearThing(x=0, y=-10, vx=sc_vx_sign * 1, vy=0)
    p = sc.copy()
    p.vy = -1
    p.vx = 0
    p = p.offset_by_time(-1)
    sc = sc.offset_by_time(-1)
    
    ts = np.linspace(0, .9)
    radiants = sd.calculate_radiant(sc, p, t0=ts)
    # Convert to an FOV position so the values are nearly constant and easy to
    # check.
    radiants = sd.elongation_to_FOV(sc.at(ts), radiants)
    np.testing.assert_allclose(radiants, -1 * sc_vx_sign * np.pi/4, atol=0.006)
    
    np.testing.assert_array_equal(
            sd.calculate_radiant(sc.offset_by_time(1e10), p),
            np.nan)


def test_calc_epsilon():
    sc = sd.LinearThing(x=-10, y=-10, vx=0, vy=0)
    p = sd.LinearThing(x=-5, y=-5, vx=0, vy=-5)
    
    assert sd.calc_epsilon(sc, p) == approx(0)
    assert sd.calc_epsilon(sc, p, t=1) == approx(np.pi/4)
    
    p.vy = 0
    sc.vy = 5
    sc.vx = 10
    
    assert sd.calc_epsilon(sc, p, t=1) == approx(np.pi/2)
    
    np.testing.assert_allclose(
            sd.calc_epsilon(sc, p, t=[0, 1]),
            [0, np.pi/2])


def test_calc_epsilon_signed():
    sc = sd.LinearThing(0, -10)
    for elongation in np.linspace(0, 359, 100):
        # Take this intended elongation angle and make it a theta relative to
        # the s/c
        theta_parcel = (90 - elongation) * np.pi / 180
        p = sd.LinearThing(
                sc.x + 5 * np.cos(theta_parcel),
                sc.y + 5 * np.sin(theta_parcel))
        
        signed_elongation = sd.calc_epsilon(
                sc, p, signed=True) * 180 / np.pi
        target_elongation = elongation
        if elongation > 180:
            target_elongation -= 360
        assert signed_elongation == approx(target_elongation)
        
        elongation = sd.calc_epsilon(sc, p) * 180 / np.pi
        target_elongation = elongation
        if elongation > 180:
            target_elongation = 360 - target_elongation
        assert elongation == approx(target_elongation)


def test_FOV_pos():
    sc = sd.LinearThing(x=-10, y=-10, vx=0.000001, vy=0)
    p = sd.LinearThing(x=0, y=-10, vx=0, vy=0)
    
    assert sd.calc_FOV_pos(sc, p) == approx(0)
    
    p = sd.LinearThing(x=0, y=0, vx=0, vy=-20)
    
    np.testing.assert_allclose(
            sd.calc_FOV_pos(sc, p, t=[0, 1]),
            [-np.pi/4, np.pi/4])
    
    sc = sd.LinearThing(x=10, y=-10, vx=-0.000001, vy=0)
    
    np.testing.assert_allclose(
            sd.calc_FOV_pos(sc, p, t=[0, 1]),
            [np.pi/4, -np.pi/4])


@image_comparison(baseline_images=['test_synthesize_image'],
        extensions=['pdf'])
def test_synthesize_image():
    sc = sd.LinearThing(x=-10, y=-10, vx=1, vy=0.2)
    p = sd.LinearThing(x=-5, y=-12, vx=-.5, vy=-.5)
    p2 = sd.LinearThing(x=5, y=-8, vx=.5, vy=.5)
    
    # Ensure a "behind" parcel doesn't show up
    p3 = sd.LinearThing(x=-20, y=-20, vx=.5, vy=.5)
    
    # Ensure nan positions are handled correctly
    p4 = p2.copy()
    p4.t_max = -99999
    
    image, wcs = sd.synthesize_image(sc, [p, p2, p3, p4], 1, fov=140)
    
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
    assert image.max() == approx(0.0004477736610888776)
    assert image.min() == approx(0)
    
    return fig
