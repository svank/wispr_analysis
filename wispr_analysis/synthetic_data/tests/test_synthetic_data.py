from .. import synthetic_data as sd


import astropy.units as u
import matplotlib.pyplot as plt
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


def setup_test_synthesize_image():
    sc = sd.LinearThing(x=-10, y=-10, vx=1, vy=0.2)
    p = sd.LinearThing(x=-5, y=-12, vx=-.5, vy=-.5)
    p2 = sd.LinearThing(x=5, y=-8, vx=.5, vy=.5)
    
    # Ensure a "behind" parcel doesn't show up
    p3 = sd.LinearThing(x=-20, y=-20, vx=.5, vy=.5)
    
    # Ensure nan positions are handled correctly
    p4 = p2.copy()
    p4.t_max = -99999
    
    # Make a close parcel we can filter out
    p5 = sd.LinearThing(x=-9, y=-9.8, vx=sc.vx, vy=sc.vy)
    
    # And a far parcel we can filter out
    p6 = sd.LinearThing(x=-10+100, y=-10+.2*100, vx=sc.vx, vy=sc.vy)
    
    return sc, [p, p2, p3, p4, p5, p6]


@pytest.mark.mpl_image_compare
def test_synthesize_image():
    sc, parcels = setup_test_synthesize_image()
    image, wcs = sd.synthesize_image(
        sc, parcels, 1, fov=140,
        parcel_width=1*u.m, dmin=2, dmax=90,
        thomson=False, use_density=False, expansion=False
        )
    
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    ax.imshow(np.sqrt(image), origin='lower', cmap='Greys_r')
    
    for coord in ax.coords:
        coord.set_major_formatter('dd')
    ax.set_xlabel("HP Longitude")
    ax.set_ylabel("HP Latitude")
    ax.coords.grid(color='white', alpha=0.7, ls='-')
    
    return fig


@pytest.mark.mpl_image_compare
def test_synthesize_image_quantity_distance():
    sc, parcels = setup_test_synthesize_image()
    image, wcs = sd.synthesize_image(
        sc, parcels, 1, fov=140,
        parcel_width=1*u.m, dmin=2, dmax=90,
        thomson=False, use_density=False, expansion=False,
        output_quantity="distance"
        )
    
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    ax.imshow(image, origin='lower', cmap='viridis')
    
    for coord in ax.coords:
        coord.set_major_formatter('dd')
    ax.set_xlabel("HP Longitude")
    ax.set_ylabel("HP Latitude")
    ax.coords.grid(color='white', alpha=0.7, ls='-')
    
    return fig


@pytest.mark.mpl_image_compare
def test_synthesize_image_quantity_rsun():
    sc, parcels = setup_test_synthesize_image()
    image, wcs = sd.synthesize_image(
        sc, parcels, 1, fov=140,
        parcel_width=1*u.m, dmin=2, dmax=90,
        thomson=False, use_density=False, expansion=False,
        output_quantity="dsun"
        )
    
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    ax.imshow(image, origin='lower', cmap='viridis')
    
    for coord in ax.coords:
        coord.set_major_formatter('dd')
    ax.set_xlabel("HP Longitude")
    ax.set_ylabel("HP Latitude")
    ax.coords.grid(color='white', alpha=0.7, ls='-')
    
    return fig


@pytest.mark.mpl_image_compare
def test_synthesize_image():
    sc, parcels = setup_test_synthesize_image()
    image, wcs = sd.synthesize_image(
        sc, parcels, 1, fov=140,
        parcel_width=1*u.m, dmin=2, dmax=90,
        thomson=False, use_density=False, expansion=False
        )
    
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    ax.imshow(np.sqrt(image), origin='lower', cmap='Greys_r')
    
    for coord in ax.coords:
        coord.set_major_formatter('dd')
    ax.set_xlabel("HP Longitude")
    ax.set_ylabel("HP Latitude")
    ax.coords.grid(color='white', alpha=0.7, ls='-')
    
    return fig


@pytest.mark.mpl_image_compare
def test_synthesize_image_point_forward():
    sc = sd.LinearThing(x=-10, y=-10, vx=1, vy=0.2)
    p = sd.LinearThing(x=-5, y=-12, vx=-.5, vy=-.5)
    p2 = sd.LinearThing(x=5, y=-8, vx=.5, vy=.5)
    
    # Ensure a "behind" parcel doesn't show up
    p3 = sd.LinearThing(x=-20, y=-20, vx=.5, vy=.5)
    
    # Ensure nan positions are handled correctly
    p4 = p2.copy()
    p4.t_max = -99999
    
    image, wcs = sd.synthesize_image(
        sc, [p, p2, p3, p4], 1, fov=140,
        parcel_width=1*u.m, point_forward=True,
        thomson=False, use_density=False, expansion=False
        )
    
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    ax.imshow(np.sqrt(image), origin='lower', cmap='Greys_r')
    
    for coord in ax.coords:
        coord.set_major_formatter('dd')
    ax.set_xlabel("HP Longitude")
    ax.set_ylabel("HP Latitude")
    ax.coords.grid(color='white', alpha=0.7, ls='-')
    
    return fig


def setup_parcels():
    sc = sd.LinearThing(y=-.25*u.au.to(u.m), vy=1)
    parcels = []
    for i in range(1, 6):
        parcels.append(sd.LinearThing(x=i*10*u.R_sun.to(u.m)))

    for theta in np.linspace(np.pi/4, 7/4*np.pi, 6):
        r = 20 * u.R_sun.to(u.m)
        parcels.append(sd.LinearThing(
            x=r*np.cos(theta), z=r*np.sin(theta)))
    
    return sc, parcels


@pytest.mark.mpl_image_compare
def test_synthesize_image_physics_less():
    sc, parcels = setup_parcels()
    image, wcs = sd.synthesize_image(
        sc, parcels, 0, output_size_x=200, output_size_y=100,
        parcel_width=12, projection='ARC', point_forward=True,
        thomson=False, use_density=False, expansion=False,
        )
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection=wcs)
    ax.imshow(np.sqrt(image), origin='lower', aspect='equal',
              cmap='Greys_r', vmin=0)
    for coord in ax.coords:
        coord.set_major_formatter('dd')
    return fig


@pytest.mark.mpl_image_compare
def test_synthesize_image_physics_radial_expansion():
    sc, parcels = setup_parcels()
    image, wcs = sd.synthesize_image(
        sc, parcels, 0, output_size_x=200, output_size_y=100,
        parcel_width=12, projection='ARC', point_forward=True,
        thomson=False, use_density=False, expansion=True,
        )
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection=wcs)
    ax.imshow(np.sqrt(image), origin='lower', aspect='equal',
              cmap='Greys_r', vmin=0)
    for coord in ax.coords:
        coord.set_major_formatter('dd')
    return fig


@pytest.mark.mpl_image_compare
def test_synthesize_image_physics_density_dropoff():
    sc, parcels = setup_parcels()
    image, wcs = sd.synthesize_image(
        sc, parcels, 0, output_size_x=200, output_size_y=100,
        parcel_width=12, projection='ARC', point_forward=True,
        thomson=False, use_density=True, expansion=False,
        )
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection=wcs)
    ax.imshow(np.sqrt(image), origin='lower', aspect='equal',
              cmap='Greys_r', vmin=0)
    for coord in ax.coords:
        coord.set_major_formatter('dd')
    return fig


@pytest.mark.mpl_image_compare
def test_synthesize_image_physics_thomson_scattering():
    sc, parcels = setup_parcels()
    image, wcs = sd.synthesize_image(
        sc, parcels, 0, output_size_x=200, output_size_y=100,
        parcel_width=12, projection='ARC', point_forward=True,
        thomson=True, use_density=False, expansion=False,
        )
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection=wcs)
    ax.imshow(np.sqrt(image), origin='lower', aspect='equal',
              cmap='Greys_r', vmin=0)
    for coord in ax.coords:
        coord.set_major_formatter('dd')
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
