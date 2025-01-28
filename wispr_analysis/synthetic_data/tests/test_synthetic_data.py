from .. import synthetic_data as sd
from ... import utils


import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest import approx


def test_calc_epsilon():
    sc = sd.LinearThing(x=-10*u.m, y=-10*u.m, vx=0*u.m/u.s, vy=0*u.m/u.s)
    p = sd.LinearThing(x=-5*u.m, y=-5*u.m, vx=0*u.m/u.s, vy=-5*u.m/u.s)
    
    Tx, Ty = sd.calc_hpc(sc, p)
    el, pa = sd.hpc_to_elpa(Tx, Ty)
    assert el.to_value(u.deg) == approx(0)
    
    Tx, Ty = sd.calc_hpc(sc, p, t=1*u.s)
    el, pa = sd.hpc_to_elpa(Tx, Ty)
    assert el.to_value(u.deg) == approx(45)
    
    p.vy = 0 * u.m/u.s
    sc.vy = 5 * u.m/u.s
    sc.vx = 10 * u.m/u.s
    
    Tx, Ty = sd.calc_hpc(sc, p, t=1*u.s)
    el, pa = sd.hpc_to_elpa(Tx, Ty)
    assert el.to_value(u.deg) == approx(90)
    
    Tx, Ty = sd.calc_hpc(sc, p, t=[0, 1]*u.s)
    el, pa = sd.hpc_to_elpa(Tx, Ty)
    np.testing.assert_allclose(el, [0, 90]*u.deg)


def setup_test_synthesize_image():
    sc = sd.LinearThing(x=-10*u.m, y=-10*u.m, vx=1*u.m/u.s, vy=0.2*u.m/u.s)
    p = sd.LinearThing(x=-5*u.m, y=-12*u.m, vx=-.5*u.m/u.s, vy=-.5*u.m/u.s)
    p2 = sd.LinearThing(x=5*u.m, y=-8*u.m, vx=.5*u.m/u.s, vy=.5*u.m/u.s)
    
    # Ensure a "behind" parcel doesn't show up
    p3 = sd.LinearThing(x=-20*u.m, y=-20*u.m, vx=.5*u.m/u.s, vy=.5*u.m/u.s)
    
    # Ensure nan positions are handled correctly
    p4 = p2.copy()
    p4.t_max = -99999 * u.s
    
    # Make a close parcel we can filter out
    p5 = sd.LinearThing(x=-9*u.m, y=-9.8*u.m, vx=sc.vx, vy=sc.vy)
    
    # And a far parcel we can filter out
    p6 = sd.LinearThing(x=(-10+100)*u.m, y=(-10+.2*100)*u.m, vx=sc.vx, vy=sc.vy)
    
    return sc, [p, p2, p3, p4, p5, p6]


@pytest.mark.mpl_image_compare
def test_synthesize_image():
    sc, parcels = setup_test_synthesize_image()
    image, wcs = sd.synthesize_image(
        sc, parcels, 1*u.s, fov=140,
        parcel_width=1*u.m, dmin=2*u.m, dmax=90*u.m,
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
        sc, parcels, 1*u.s, fov=140,
        parcel_width=1*u.m, dmin=2*u.m, dmax=90*u.m,
        thomson=False, use_density=False, expansion=False,
        output_quantity="distance"
        )
    
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    ax.imshow(image, origin='lower', cmap='viridis', interpolation_stage='data')
    
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
        sc, parcels, 1*u.s, fov=140,
        parcel_width=1*u.m, dmin=2*u.m, dmax=90*u.m,
        thomson=False, use_density=False, expansion=False,
        output_quantity="dsun"
        )
    
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    ax.imshow(image, origin='lower', cmap='viridis', interpolation_stage='data')
    
    for coord in ax.coords:
        coord.set_major_formatter('dd')
    ax.set_xlabel("HP Longitude")
    ax.set_ylabel("HP Latitude")
    ax.coords.grid(color='white', alpha=0.7, ls='-')
    
    return fig


@pytest.mark.mpl_image_compare
def test_synthesize_image_point_forward():
    sc = sd.LinearThing(x=-10*u.m, y=-10*u.m, vx=1*u.m/u.s, vy=0.2*u.m/u.s)
    p = sd.LinearThing(x=-5*u.m, y=-12*u.m, vx=-.5*u.m/u.s, vy=-.5*u.m/u.s)
    p2 = sd.LinearThing(x=5*u.m, y=-8*u.m, vx=.5*u.m/u.s, vy=.5*u.m/u.s)
    
    # Ensure a "behind" parcel doesn't show up
    p3 = sd.LinearThing(x=-20*u.m, y=-20*u.m, vx=.5*u.m/u.s, vy=.5*u.m/u.s)
    
    # Ensure nan positions are handled correctly
    p4 = p2.copy()
    p4.t_max = -99999 * u.s
    
    image, wcs = sd.synthesize_image(
        sc, [p, p2, p3, p4], 1*u.s, fov=140,
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
    sc = sd.LinearThing(y=-.25*u.au, vy=1*u.m/u.s)
    parcels = []
    for i in range(1, 6):
        parcels.append(sd.LinearThing(x=i*10*u.R_sun))

    for theta in np.linspace(np.pi/4, 7/4*np.pi, 6):
        r = 20 * u.R_sun
        parcels.append(sd.LinearThing(
            x=r*np.cos(theta), z=r*np.sin(theta)))
    
    return sc, parcels


@pytest.mark.mpl_image_compare
def test_synthesize_image_physics_less():
    sc, parcels = setup_parcels()
    image, wcs = sd.synthesize_image(
        sc, parcels, 0*u.s, output_size_x=200, output_size_y=100,
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
        sc, parcels, 0*u.s, output_size_x=200, output_size_y=100,
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
        sc, parcels, 0*u.s, output_size_x=200, output_size_y=100,
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
        sc, parcels, 0*u.s, output_size_x=200, output_size_y=100,
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
    sc = sd.LinearThing(y=-10*u.m, vy=10*u.m/u.s)
    
    p = sd.LinearThing(x=10*u.m)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx.to_value(u.deg) == approx(45)
    assert Ty.to_value(u.deg) == approx(0)
    
    p = sd.LinearThing(x=-10*u.m)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx.to_value(u.deg) == approx(-45)
    assert Ty.to_value(u.deg) == approx(0)
    
    p = sd.LinearThing(z=10*u.m)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx.to_value(u.deg) == approx(0)
    assert Ty.to_value(u.deg) == approx(45)
    
    p = sd.LinearThing(z=-10*u.m)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx.to_value(u.deg) == approx(0)
    assert Ty.to_value(u.deg) == approx(-45)
    
    p = sd.LinearThing(x=10/np.sqrt(2)*u.m, z=-10/np.sqrt(2)*u.m)
    Tx, Ty = sd.calc_hpc(sc, p)
    
    pa = 225 * np.pi / 180
    diff = (p - sc)
    el = utils.angle_between_vectors(
        -sc.x, -sc.y, -sc.z, diff.x, diff.y, diff.z)
    Tx_expected = np.arctan2(-np.sin(el) * np.sin(pa), np.cos(el))
    Ty_expected = np.arcsin(np.sin(el) * np.cos(pa))
    
    assert Tx.to_value(u.deg) == approx(Tx_expected[0].to_value(u.deg))
    assert Ty.to_value(u.deg) == approx(Ty_expected[0].to_value(u.deg))


def test_calc_hpc_from_x():
    sc = sd.LinearThing(x=10*u.m, vx=10*u.m/u.s)
    
    p = sd.LinearThing(y=10*u.m)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx.to_value(u.deg) == approx(45)
    assert Ty.to_value(u.deg) == approx(0)
    
    p = sd.LinearThing(y=-10*u.m)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx.to_value(u.deg) == approx(-45)
    assert Ty.to_value(u.deg) == approx(0)
    
    p = sd.LinearThing(z=10*u.m)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx.to_value(u.deg) == approx(0)
    assert Ty.to_value(u.deg) == approx(45)
    
    p = sd.LinearThing(z=-10*u.m)
    Tx, Ty = sd.calc_hpc(sc, p)
    assert Tx.to_value(u.deg) == approx(0)
    assert Ty.to_value(u.deg) == approx(-45)
    
    p = sd.LinearThing(y=10/np.sqrt(2)*u.m, z=-10/np.sqrt(2)*u.m)
    Tx, Ty = sd.calc_hpc(sc, p)
    
    pa = 225 * np.pi / 180
    diff = (p - sc)
    el = utils.angle_between_vectors(
        -sc.x, -sc.y, -sc.z, diff.x, diff.y, diff.z)
    Tx_expected = np.arctan2(-np.sin(el) * np.sin(pa), np.cos(el))
    Ty_expected = np.arcsin(np.sin(el) * np.cos(pa))
    
    assert Tx.to_value(u.deg) == approx(Tx_expected[0].to_value(u.deg))
    assert Ty.to_value(u.deg) == approx(Ty_expected[0].to_value(u.deg))


def test_calculate_radiant():
    # Create two objects which will collide
    sc = sd.LinearThing(x=0*u.m, y=-100*u.m, vx=1*u.m/u.s, vy=0*u.m/u.s)
    p = sc.copy()
    p.vy = -1*u.m/u.s
    p.vx = 0*u.m/u.s
    p = p.offset_by_time(-1*u.s)
    sc = sc.offset_by_time(-1*u.s)
    
    ts = np.linspace(0, .9) * u.s
    rads = sd.calculate_radiant(sc, p, t0=ts)
    # Convert to an FOV position so the values are nearly constant and easy to
    # check.
    rads = sd.elongation_to_FOV(sc.at(ts), rads)
    np.testing.assert_allclose(rads, -np.pi/4*u.rad, atol=0.006)
    
    np.testing.assert_array_equal(
            sd.calculate_radiant(sc.offset_by_time(1e10*u.s), p),
            np.nan)


def test_elongation_to_FOV():
    sc = sd.LinearThing(x=0*u.m, y=-1*u.m, vx=0*u.m/u.s, vy=1*u.m/u.s)
    # Elongation is FOV
    elongations = np.linspace(0, 180) * u.deg
    fovs = sd.elongation_to_FOV(sc, elongations)
    np.testing.assert_allclose(elongations, fovs)
    
    sc.vy = 0*u.m/u.s
    sc.vx = 1*u.m/u.s
    # FOV runs from -90 to 90 degrees
    fovs = sd.elongation_to_FOV(sc, elongations)
    np.testing.assert_allclose(np.linspace(-np.pi/2, np.pi/2)*u.rad, fovs)
    
    # Test vectorizing over time
    sc = sd.LinearThing(x=0*u.m, y=-1*u.m, vx=1*u.m/u.s).at(np.array([-1, 0, 1])*u.s)
    elongations = np.array([0, 0, 0])
    fovs = sd.elongation_to_FOV(sc, elongations)
    np.testing.assert_allclose(
            [-np.pi/4, -np.pi/2, -np.pi/2 - np.pi/4]*u.rad, fovs)