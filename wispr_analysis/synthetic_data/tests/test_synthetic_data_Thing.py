from .. import synthetic_data as sd


import itertools

import astropy.units as u
import numpy as np
import pytest
from pytest import approx


def test_is_same_time():
    t1 = sd.LinearThing(t=0)
    t2 = sd.LinearThing(t=1)
    t3 = t2.at(0)
    t4 = sd.ArrayThing([0, 1], t=1)
    t5 = t4.at(0)
    
    assert t1.is_same_time(t3)
    assert not t1.is_same_time(t2)
    
    assert t1.is_same_time(t5)
    assert not t1.is_same_time(t4)


def test_LinearThing_at():
    for x, y, z in itertools.product([-20, 0, 20], [-20, 0, 20], [-20, 0, 20]):
        for vx, vy, vz in itertools.product(
                [-10, 0, 20], [-10, 0, 20], [-10, 0, 20]):
            for t in np.linspace(-3, 3, 5):
                # Create a Thing and ensure it propagates correctly in time
                thing = sd.LinearThing(
                        x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
                thing = thing.at(t)
                assert thing.vx == vx
                assert thing.vy == vy
                assert thing.vz == vz
                assert thing.x == x + vx * t
                assert thing.y == y + vy * t
                assert thing.z == z + vz * t
                
                # Create a Thing at a non-zero point in time, ensure it's
                # correct
                thing2 = sd.LinearThing(
                        x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, t=t)
                thing3 = thing2.at(t)
                assert thing2 == thing3
                for thing in (thing2, thing3):
                    assert thing.vx == vx
                    assert thing.vy == vy
                    assert thing.vz == vz
                    assert thing.x == approx(x)
                    assert thing.y == approx(y)
                    assert thing.z == approx(z)
                
                thing4 = thing2.at(0)
                assert thing4.vx == vx
                assert thing4.vy == vy
                assert thing4.vz == vz
                assert thing4.x == x - vx * t
                assert thing4.y == y - vy * t
                assert thing4.z == z - vz * t


def test_LinearThing_density():
    thing = sd.LinearThing(rho=2, density_r2=False)
    assert thing.rho == 2
    thing = sd.LinearThing(x=10, rho=2, density_r2=True)
    assert thing.rho == 2 / 10**2


def test_LinearThing_rperp_rpar():
    thing = sd.LinearThing(rperp=1, rpar=2)
    assert thing.rperp == 1
    assert thing.rpar == 2


def test_linear_ArrayThing_at():
    t_eval = np.array([-10, 10])
    for x, y, z in itertools.product([-20, 0, 20], [-20, 0, 20], [-20, 0, 20]):
        for vx, vy, vz in itertools.product(
                [-10, 0, 20], [-10, 0, 20], [-10, 0, 20]):
            # Create a Thing and ensure it propagates correctly in time
            thing = sd.ArrayThing(
                tlist=t_eval,
                xlist=x + vx * t_eval,
                ylist=y + vy * t_eval,
                zlist=z + vz * t_eval,
                rholist=2 * t_eval)
            for t in np.linspace(-3, 3, 5):
                thing = thing.at(t)
                assert thing.vx == approx(vx)
                assert thing.vy == approx(vy)
                assert thing.vz == approx(vz)
                assert thing.x == x + vx * t
                assert thing.y == y + vy * t
                assert thing.z == z + vz * t
                assert thing.rho == 2 * t


def test_ArrayThing_default_density():
    thing = sd.ArrayThing([1, 5], rholist=3)
    assert thing.at(2).rho == 3
    thing = sd.ArrayThing([1, 5], [1, 5], rholist=3, default_density_r2=True)
    assert thing.at(2).rho == 3 / 2**2


def test_ArrayThing_rperp_rpar():
    thing = sd.ArrayThing([1, 5], rperplist=3, rparlist=4)
    assert thing.at(2).rperp == 3
    assert thing.at(2).rpar == 4
    thing = sd.ArrayThing([1, 5], rperplist=[3, 4], rparlist=[4, 5])
    assert thing.at(3).rperp == 3.5
    assert thing.at(3).rpar == 4.5


@pytest.mark.parametrize('type1', ['linear', 'array'])
@pytest.mark.parametrize('type2', ['linear', 'array'])
def test_in_front_of(type1, type2):
    t_eval = np.array([-20, 20])
    # Run a grid of velocity directions
    for vx, vy, vz in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
        # Things should not change over time, the way they're set up here
        for t in np.linspace(-10, 10, 3):
            if vx == 0 and vy == 0 and vz == 0:
                # No velocity direction
                continue
            if type1 == 'linear':
                sc = sd.LinearThing(x=0, y=0, z=0, vx=vx, vy=vy, vz=vz)
            else:
                sc = sd.ArrayThing(
                        t_eval,
                        xlist=vx * t_eval,
                        ylist=vy * t_eval,
                        zlist=vz * t_eval)
            # Make in-front and behind copies of our object
            if type2 == 'linear':
                sc_infront = sd.LinearThing(
                        x=vx, y=vy, z=vz, vx=vx, vy=vy, vz=vz)
                sc_behind = sd.LinearThing(
                        x=-vx, y=-vy, z=-vz, vx=vx, vy=vy, vz=vz)
            else:
                sc_infront = sd.ArrayThing(
                        t_eval,
                        xlist=vx * (t_eval + 1),
                        ylist=vy * (t_eval + 1),
                        zlist=vz * (t_eval + 1))
                sc_behind = sd.ArrayThing(
                        t_eval,
                        xlist=vx * (t_eval - 1),
                        ylist=vy * (t_eval - 1),
                        zlist=vz * (t_eval - 1))
            assert sc_infront.in_front_of(sc, t)
            assert not sc_behind.in_front_of(sc, t)
            
            # Offset the in-front and behind objects perpendicular to the main
            # object's velocity vector---should not change in-front-ness
            # Find a vector perpendicular to the velocity vector by
            # crossing it with any other vector
            vel_vector = np.array((vx, vy, vz), dtype=float)
            vel_vector /= np.sqrt(np.sum(vel_vector**2))
            another_vector = np.array((18, -12, 3), dtype=float)
            another_vector /= np.sqrt(np.sum(another_vector**2))
            perp_vector = np.cross(vel_vector, another_vector)
            for delta in np.logspace(-2, 2, 11):
                offset = perp_vector * delta
                if type2 == 'linear':
                    sc_offset1 = sd.LinearThing(
                            x=sc_infront.x + offset[0],
                            y=sc_infront.y + offset[1],
                            z=sc_infront.z + offset[2])
                    sc_offset2 = sd.LinearThing(
                            x=sc_infront.x - offset[0],
                            y=sc_infront.y - offset[1],
                            z=sc_infront.z - offset[2])
                    
                    sc_offset3 = sd.LinearThing(
                            x=sc_behind.x + offset[0],
                            y=sc_behind.y + offset[1],
                            z=sc_behind.z + offset[2])
                    sc_offset4 = sd.LinearThing(
                            x=sc_behind.x - offset[0],
                            y=sc_behind.y - offset[1],
                            z=sc_behind.z - offset[2])
                else:
                    sc_offset1 = sd.ArrayThing(
                            t_eval,
                            xlist=vx * (t_eval + 1) + offset[0],
                            ylist=vy * (t_eval + 1) + offset[1],
                            zlist=vz * (t_eval + 1) + offset[2])
                    
                    sc_offset2 = sd.ArrayThing(
                            t_eval,
                            xlist=vx * (t_eval + 1) - offset[0],
                            ylist=vy * (t_eval + 1) - offset[1],
                            zlist=vz * (t_eval + 1) - offset[2])
                    
                    sc_offset3 = sd.ArrayThing(
                            t_eval,
                            xlist=vx * (t_eval - 1) + offset[0],
                            ylist=vy * (t_eval - 1) + offset[1],
                            zlist=vz * (t_eval - 1) + offset[2])
                    
                    sc_offset4 = sd.ArrayThing(
                            t_eval,
                            xlist=vx * (t_eval - 1) - offset[0],
                            ylist=vy * (t_eval - 1) - offset[1],
                            zlist=vz * (t_eval - 1) - offset[2])
                
                assert sc_offset1.in_front_of(sc, 0)
                assert sc_offset2.in_front_of(sc, 0)
                
                assert not sc_offset3.in_front_of(sc, 0)
                assert not sc_offset4.in_front_of(sc, 0)
            
            # Have the behind object catch up, test passing t0 value
            if type2 == 'linear':
                sc_behind = sd.LinearThing(
                        x=-vx, y=-vy, z=-vz, vx=2*vx, vy=2*vy, vz=2*vz)
            else:
                sc_behind = sd.ArrayThing(
                        t_eval,
                        xlist=2 * vx * (t_eval - 1),
                        ylist=2 * vy * (t_eval - 1),
                        zlist=2 * vz * (t_eval - 1))
            assert not sc_behind.in_front_of(sc)
            assert sc_behind.in_front_of(sc, 4)
            
            # Fix the objects in place, continue testing passing t0 value
            if type2 == 'linear':
                sc_infront = sd.LinearThing(x=vx, y=vy, z=vz, vx=0, vy=0, vz=0)
                sc_behind = sd.LinearThing(
                        x=-vx, y=-vy, z=-vz, vx=0, vy=0, vz=0)
            else:
                sc_infront = sd.ArrayThing(
                        t_eval,
                        xlist=vx * 1,
                        ylist=vy * 1,
                        zlist=vz * 1)
                sc_behind = sd.ArrayThing(
                        t_eval,
                        xlist=vx * -1,
                        ylist=vy * -1,
                        zlist=vz * -1)
            
            assert sc_infront.in_front_of(sc, 0)
            assert sc_infront.in_front_of(sc, -1)
            assert not sc_infront.in_front_of(sc, 2)
            
            assert not sc_behind.in_front_of(sc, 0)
            assert sc_behind.in_front_of(sc, -2)
            assert not sc_behind.in_front_of(sc, 2)


@pytest.mark.parametrize('type1', ['linear', 'array'])
@pytest.mark.parametrize('type2', ['linear', 'array'])
def test_subtract(type1, type2):
    if type1 == 'linear':
        t1 = sd.LinearThing()
    else:
        t1 = sd.ArrayThing(tlist=[-1, 1])
    if type2 == 'linear':
        t2 = sd.LinearThing(x=1, y=1, z=1, vx=10, vy=10, vz=10)
    else:
        t2 = sd.ArrayThing(
                xlist=[-9, 11],
                ylist=[-9, 11],
                zlist=[-9, 11],
                tlist=[-1, 1])
    diff = t2 - t1
    assert diff.r == np.sqrt(3)
    assert diff.at(1).r == np.sqrt(11**2 + 11**2 + 11**2)
    assert diff.at(-1).r == np.sqrt(9**2 + 9**2 + 9**2)
    assert diff.x == 1
    assert diff.y == 1
    assert diff.z == 1
    assert diff.vx == approx(10)
    assert diff.vy == approx(10)
    assert diff.vz == approx(10)
    
    if type1 == 'linear':
        t1 = sd.LinearThing(vx=10, vy=-10, vz=3)
    else:
        t1 = sd.ArrayThing(
                xlist=[-10, 10],
                ylist=[10, -10],
                zlist=[-3, 3],
                tlist=[-1, 1])
    diff = t2 - t1
    assert diff.r == np.sqrt(3)
    assert diff.at(1).x == 1
    assert diff.at(1).y == 21
    assert diff.at(1).z == 8


def test_tbounds():
    thing1 = sd.LinearThing(t_min=-2, t_max=2)
    thing2 = sd.ArrayThing([-20, 20], t_min=-2, t_max=2)
    
    for thing in [thing1, thing2]:
        for t in [-2.1, 2.1]:
            assert np.isnan(thing.at(t).x)
            assert np.isnan(thing.at(t).y)
            assert np.isnan(thing.at(t).z)
            assert np.isnan(thing.at(t).vx)
            assert np.isnan(thing.at(t).vy)
            assert np.isnan(thing.at(t).vz)
        
        for t in [-1.9, 0, 1.9]:
            assert not np.isnan(thing.at(t).x)
            assert not np.isnan(thing.at(t).y)
            assert not np.isnan(thing.at(t).z)
            assert not np.isnan(thing.at(t).vx)
            assert not np.isnan(thing.at(t).vy)
            assert not np.isnan(thing.at(t).vz)
        
        t = np.array([-2.1, -1.9, 0, 1.9, 2.1])
        for result in [thing.at(t).x,
                       thing.at(t).y,
                       thing.at(t).z,
                       thing.at(t).vx,
                       thing.at(t).vy,
                       thing.at(t).vz]:
            assert np.isnan(result[0])
            assert np.isnan(result[-1])
            assert not np.any(np.isnan(result[1:-1]))


def test_tbounds_array_range():
    thing = sd.ArrayThing([-20, 20], t_min=-2, t_max=2)
    for t in -10, 10:
        for attr in 'x', 'y', 'z', 'vx', 'vy', 'vz':
            res = getattr(thing.at(t), attr)
            assert res.shape == (1,)
            assert np.isnan(res[0])
    
    for t in [-10, -9], [9, 10]:
        for attr in 'x', 'y', 'z', 'vx', 'vy', 'vz':
            res = getattr(thing.at(t), attr)
            assert res.shape == (2,)
            assert np.all(np.isnan(res))
    
    thing = sd.ArrayThing([-20, 20])
    
    for t in -25, 25:
        for attr in 'x', 'y', 'z', 'vx', 'vy', 'vz':
            res = getattr(thing.at(t), attr)
            assert res.shape == (1,)
            assert np.isnan(res[0])
    
    for t in [-25, -24], [24, 25]:
        for attr in 'x', 'y', 'z', 'vx', 'vy', 'vz':
            res = getattr(thing.at(t), attr)
            assert res.shape == (2,)
            assert np.all(np.isnan(res))


def test_LinearThing_units():
    thing = sd.LinearThing(
        x=1*u.km, y=1*u.km, z=1*u.km,
        vx=1*u.km/u.s, vy=1*u.km/u.s, vz=1*u.km/u.s,
        t=0*u.s, t_min=1*u.s, t_max=9*u.s,
        rperp=1*u.m, rpar=1*u.m,
        rho=1*u.g/u.m**3)
    
    thing = thing.at(1*u.s)
    assert thing.x == 2 * u.km
    assert thing.y == 2 * u.km
    assert thing.z == 2 * u.km
    
    assert thing.vx == 1*u.km/u.s
    assert thing.vy == 1*u.km/u.s
    assert thing.vz == 1*u.km/u.s
    
    assert thing.rperp == 1*u.m
    assert thing.rpar == 1*u.m
    assert thing.rho == 1*u.g/u.m**3
    
    thing = thing.strip_units()
    
    assert thing.x == 2000
    assert thing.y == 2000
    assert thing.z == 2000
    
    assert np.isclose(thing.vx, 1000)
    assert np.isclose(thing.vy, 1000)
    assert np.isclose(thing.vz, 1000)
    
    assert thing.rperp == 1
    assert thing.rpar == 1
    assert thing.rho == 0.001


def test_ArrayThing_units():
    thing = sd.ArrayThing(
        tlist=[0, 10] * u.s,
        xlist=[1, 2] * u.km,
        ylist=[1, 2] * u.km,
        zlist=[1, 2] * u.km,
        t_min=1*u.s, t_max=9*u.s,
        rperplist=1*u.m, rparlist=1*u.m,
        rholist=1*u.g/u.m**3)
    
    thing = thing.at(5*u.s)
    assert thing.x == 1.5 * u.km
    assert thing.y == 1.5 * u.km
    assert thing.z == 1.5 * u.km
    
    assert np.isclose(thing.vx, 1*u.km / (10*u.s))
    assert np.isclose(thing.vy, 1*u.km / (10*u.s))
    assert np.isclose(thing.vz, 1*u.km / (10*u.s))
    
    assert thing.rperp == 1*u.m
    assert thing.rpar == 1*u.m
    assert thing.rho == 1*u.g/u.m**3
    
    thing = thing.strip_units()
    
    assert thing.x == 1500
    assert thing.y == 1500
    assert thing.z == 1500
    
    assert np.isclose(thing.vx, 100)
    assert np.isclose(thing.vy, 100)
    assert np.isclose(thing.vz, 100)
    
    assert thing.rperp == 1
    assert thing.rpar == 1
    assert thing.rho == 0.001