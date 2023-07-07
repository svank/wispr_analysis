from .. import synthetic_data as sd


import matplotlib.pyplot as plt
from matplotlib.testing.conftest import mpl_test_settings
from matplotlib.testing.decorators import image_comparison
import numpy as np
import pytest
from pytest import approx


@pytest.mark.parametrize("v1,v2,answer",
        [((1, 0, 0), (0, 1, 0), np.pi/2),
        ((0, -1, 0), (1, 0, 0), np.pi/2),
        ((1, 1, 0), (0, 1, 0), np.pi/4),
        ((1, 0, 0), (1, 1, 0), np.pi/4),
        ((0, 1, 0), (0, -1, 0), np.pi),
        ((1, 0, 0), (-1, 0, 0), np.pi),
        ((0, 1, 0), (-1, 0, 0), np.pi/2),
        ((1, 1, 1), (1, 1, -1), np.arccos(1/3)),
        ((0, 1, 0), (-1, -1, 0), np.pi/2 + np.pi/4)])
def test_angle_between_vectors(v1, v2, answer):
    assert sd.angle_between_vectors(*v1, *v2) == approx(answer)
    assert sd.angle_between_vectors(*v2, *v1) == approx(answer)
    
    v1 = (v1[1], v1[2], v1[0])
    v2 = (v2[1], v2[2], v2[0])
    assert sd.angle_between_vectors(*v1, *v2) == approx(answer)
    assert sd.angle_between_vectors(*v2, *v1) == approx(answer)
    
    v1 = (v1[1], v1[2], v1[0])
    v2 = (v2[1], v2[2], v2[0])
    assert sd.angle_between_vectors(*v1, *v2) == approx(answer)
    assert sd.angle_between_vectors(*v2, *v1) == approx(answer)
    

def test_angle_between_same_vectors():
    assert sd.angle_between_vectors(1, 1, 1, 1, 1, 1) == approx(0, abs=2e-8)
    assert sd.angle_between_vectors(0, 1, 0, 0, 1, 0) == approx(0, abs=2e-8)


def test_angle_between_vectors_zero_vector():
    assert np.isnan(sd.angle_between_vectors(0, 0, 0, 1, 1, 0))
    assert np.isnan(sd.angle_between_vectors(1, 1, 0, 0, 0, 0))


def test_elongation_to_FOV():
    sc = sd.LinearThing(x=0, y=-1, vx=0, vy=1)
    # Elongation is FOV
    elongations = np.linspace(0, 180) * np.pi / 180
    fovs = sd.elongation_to_FOV(sc, elongations)
    np.testing.assert_allclose(elongations, fovs)
    
    sc.vy = 0
    sc.vx = 1
    # FOV runs from -90 to 90 degrees
    fovs = sd.elongation_to_FOV(sc, elongations)
    np.testing.assert_allclose(np.linspace(-np.pi/2, np.pi/2), fovs)
    
    # Test vectorizing over time
    sc = sd.LinearThing(x=0, y=-1, vx=1).at(np.array([-1, 0, 1]))
    elongations = np.array([0, 0, 0])
    fovs = sd.elongation_to_FOV(sc, elongations)
    np.testing.assert_allclose(
            [-np.pi/4, -np.pi/2, -np.pi/2 - np.pi/4], fovs)


def test_calculate_radiant():
    # Create two objects which will collide
    sc = sd.LinearThing(x=0, y=-100, vx=1, vy=0)
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
    np.testing.assert_allclose(radiants, -np.pi/4, atol=0.006)
    
    np.testing.assert_array_equal(
            sd.calculate_radiant(sc.offset_by_time(1e10), p),
            np.nan)


def test_calc_epsilon():
    sc = sd.LinearThing(x=-10, y=-10, vx=0, vy=0)
    p = sd.LinearThing(x=-5, y=-5, vx=0, vy=-5)
    
    Tx, Ty = sd.calc_hpc(sc, p)
    el, pa = sd.hpc_to_elpa(Tx, Ty)
    assert el == approx(0)
    
    Tx, Ty = sd.calc_hpc(sc, p, t=1)
    el, pa = sd.hpc_to_elpa(Tx, Ty)
    assert el == approx(45)
    
    p.vy = 0
    sc.vy = 5
    sc.vx = 10
    
    Tx, Ty = sd.calc_hpc(sc, p, t=1)
    el, pa = sd.hpc_to_elpa(Tx, Ty)
    assert el == approx(90)
    
    Tx, Ty = sd.calc_hpc(sc, p, t=[0, 1])
    el, pa = sd.hpc_to_elpa(Tx, Ty)
    np.testing.assert_allclose(el, [0, 90])


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
    
    return fig


def test_calc_hpc():
    sc = sd.LinearThing(y=-10, vy=10)
    
    p = sd.LinearThing(x=10)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx == approx(45)
    assert Ty == approx(0)
    
    p = sd.LinearThing(x=-10)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx == approx(-45)
    assert Ty == approx(0)
    
    p = sd.LinearThing(z=10)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx == approx(0)
    assert Ty == approx(45)
    
    p = sd.LinearThing(z=-10)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx == approx(0)
    assert Ty == approx(-45)
    
    p = sd.LinearThing(x=10/np.sqrt(2), z=-10/np.sqrt(2))
    Tx, Ty = sd.calc_hpc(sc, p)
    
    pa = 225 * np.pi / 180
    diff = (p - sc)
    el = sd.angle_between_vectors(-sc.x, -sc.y, -sc.z, diff.x, diff.y, diff.z)
    Tx_expected = np.arctan2(-np.sin(el) * np.sin(pa), np.cos(el))
    Ty_expected = np.arcsin(np.sin(el) * np.cos(pa))
    
    assert Tx == approx(Tx_expected[0] * 180 / np.pi)
    assert Ty == approx(Ty_expected[0] * 180 / np.pi)


def test_calc_hpc_from_x():
    sc = sd.LinearThing(x=10, vx=10)
    
    p = sd.LinearThing(y=10)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx == approx(45)
    assert Ty == approx(0)
    
    p = sd.LinearThing(y=-10)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx == approx(-45)
    assert Ty == approx(0)
    
    p = sd.LinearThing(z=10)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx == approx(0)
    assert Ty == approx(45)
    
    p = sd.LinearThing(z=-10)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx == approx(0)
    assert Ty == approx(-45)
    
    p = sd.LinearThing(y=10/np.sqrt(2), z=-10/np.sqrt(2))
    Tx, Ty = sd.calc_hpc(sc, p)
    
    pa = 225 * np.pi / 180
    diff = (p - sc)
    el = sd.angle_between_vectors(-sc.x, -sc.y, -sc.z, diff.x, diff.y, diff.z)
    Tx_expected = np.arctan2(-np.sin(el) * np.sin(pa), np.cos(el))
    Ty_expected = np.arcsin(np.sin(el) * np.cos(pa))
    
    assert Tx == approx(Tx_expected[0] * 180 / np.pi)
    assert Ty == approx(Ty_expected[0] * 180 / np.pi)
