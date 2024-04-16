import astropy.coordinates
import astropy.units as u
import astropy.wcs
import numpy as np
import pytest
import sunpy.coordinates

from .. import orbital_frame


def test_orbital_plane_wcs_frame_mapping():
    nx = 4000
    ny = nx * 45 // 360
    wcs = astropy.wcs.WCS(naxis=2)
    wcs.wcs.ctype = 'PSLN-CAR', 'PSLT-CAR'
    wcs.wcs.crpix = nx/2 + .5, ny/2 + .5 + 55
    wcs.wcs.crval = 180, 0
    wcs.wcs.cdelt = -360 / nx, 45 / ny
    wcs.wcs.cunit = 'deg', 'deg'
    wcs.wcs.dateobs = '2021-01-01 12:12:12'
    
    frame = orbital_frame.orbital_plane_wcs_frame_mapping(wcs)
    assert isinstance(frame, orbital_frame.PSPFrame)
    assert frame.obstime == wcs.wcs.dateobs
    
    wcs.wcs.ctype = 'POLN-ARC', 'POLT-ARC'
    
    frame = orbital_frame.orbital_plane_wcs_frame_mapping(wcs)
    assert isinstance(frame, orbital_frame.PSPOrbitalFrame)
    assert frame.obstime == wcs.wcs.dateobs
    
    wcs.wcs.ctype = 'RA---CAR', 'DEC--CAR'
    frame = orbital_frame.orbital_plane_wcs_frame_mapping(wcs)
    assert frame is None


def test_coord_from_wcs():
    nx = 4000
    ny = nx * 45 // 360
    wcs = astropy.wcs.WCS(naxis=2)
    wcs.wcs.ctype = 'PSLN-CAR', 'PSLT-CAR'
    wcs.wcs.crpix = nx/2 + .5, ny/2 + .5 + 55
    wcs.wcs.crval = 180, 0
    wcs.wcs.cdelt = -360 / nx, 45 / ny
    wcs.wcs.cunit = 'deg', 'deg'
    wcs.wcs.dateobs = '2021-01-01 12:12:12'
    
    coord = wcs.pixel_to_world(1,1)
    assert isinstance(coord.frame, orbital_frame.PSPFrame)
    assert coord.frame.obstime == wcs.wcs.dateobs
    
    wcs.wcs.ctype = 'POLN-ARC', 'POLT-ARC'
    
    coord = wcs.pixel_to_world(1,1)
    assert isinstance(coord.frame, orbital_frame.PSPOrbitalFrame)
    assert coord.frame.obstime == wcs.wcs.dateobs


def test_orbital_plane_frame_wcs_mapping():
    obstime = '2022-02-03T12:12:12.000'
    frame = orbital_frame.PSPOrbitalFrame(obstime=obstime)
    wcs = orbital_frame.orbital_plane_frame_wcs_mapping(frame, 'CAR')
    assert list(wcs.wcs.ctype) == ['POLN-CAR', 'POLT-CAR']
    assert wcs.wcs.dateobs == obstime
    assert wcs.world_axis_physical_types == [
        'custom:pos.psporbitalframe.lon',
        'custom:pos.psporbitalframe.lat']


def test_psp_frame_wcs_mapping():
    obstime = '2022-02-03T12:12:12.000'
    observer = astropy.coordinates.SkyCoord(10*u.deg, 20*u.deg, 30*u.R_sun,
                                            frame='heliographic_stonyhurst')
    frame = orbital_frame.PSPFrame(obstime=obstime, observer=observer)
    wcs = orbital_frame.orbital_plane_frame_wcs_mapping(frame, 'CAR')
    assert list(wcs.wcs.ctype) == ['PSLN-CAR', 'PSLT-CAR']
    assert wcs.wcs.dateobs == obstime
    assert wcs.world_axis_physical_types == [
        'custom:pos.pspframe.lon',
        'custom:pos.pspframe.lat']
    assert wcs.wcs.aux.hgln_obs == observer.lon.to_value(u.deg)
    assert wcs.wcs.aux.hglt_obs == observer.lat.to_value(u.deg)
    assert wcs.wcs.aux.dsun_obs == observer.radius.to_value(u.m)


def test_wcs_from_psporbital_coord():
    obstime = '2022-02-03T12:12:12.000'
    frame = orbital_frame.PSPOrbitalFrame(obstime=obstime)
    wcs = astropy.wcs.utils.celestial_frame_to_wcs(frame)
    assert list(wcs.wcs.ctype) == ['POLN-TAN', 'POLT-TAN']
    assert wcs.wcs.dateobs == obstime
    assert wcs.world_axis_physical_types == [
        'custom:pos.psporbitalframe.lon',
        'custom:pos.psporbitalframe.lat']


def test_wcs_from_psp_coord():
    obstime = '2022-02-03T12:12:12.000'
    observer = astropy.coordinates.SkyCoord(10*u.deg, 20*u.deg, 30*u.R_sun,
                                            frame='heliographic_stonyhurst')
    frame = orbital_frame.PSPFrame(obstime=obstime, observer=observer)
    wcs = astropy.wcs.utils.celestial_frame_to_wcs(frame)
    assert list(wcs.wcs.ctype) == ['PSLN-TAN', 'PSLT-TAN']
    assert wcs.wcs.dateobs == obstime
    assert wcs.world_axis_physical_types == [
        'custom:pos.pspframe.lon',
        'custom:pos.pspframe.lat']
    assert wcs.wcs.aux.hgln_obs == observer.lon.to_value(u.deg)
    assert wcs.wcs.aux.hglt_obs == observer.lat.to_value(u.deg)
    assert wcs.wcs.aux.dsun_obs == observer.radius.to_value(u.m)


def test_psporbital_hci_roundtrip(mocker):
    # The PSP kernels are very large and not included in this repo
    mocker.patch('wispr_analysis.orbital_frame.planets.get_orbital_elements',
                return_value=(
                    19390313.93764631 * u.km,
                    0.741835928443799,
                    0.42649992 * u.rad,
                    0.13972456 * u.rad,
                    2.56225543 * u.rad,
                    ))
    
    obstime = '2023-01-02 03:04:05'
    c = astropy.coordinates.SkyCoord(
        10*u.deg, 30*u.deg, 5*u.au, frame='psporbitalframe',
        obstime=obstime)
    
    c2 = c.transform_to('heliocentricinertial')
    
    c3 = c2.transform_to('psporbitalframe')
    
    assert np.isclose(c.lon, c3.lon)
    assert np.isclose(c.lat, c3.lat)
    assert np.isclose(c.distance, c3.distance)
    assert c3.obstime == obstime
    
    c4 = c3.transform_to('heliocentricinertial')
    
    assert np.isclose(c2.lon, c4.lon)
    assert np.isclose(c2.lat, c4.lat)
    assert np.isclose(c2.distance, c4.distance)


def test_psp_hpc_roundtrip(mocker):
    # The PSP kernels are very large and not included in this repo
    mocker.patch('wispr_analysis.orbital_frame.planets.get_orbital_elements',
                return_value=(
                    19390313.93764631 * u.km,
                    0.741835928443799,
                    0.42649992 * u.rad,
                    0.13972456 * u.rad,
                    2.56225543 * u.rad,
                    ))
    
    obstime = '2023-01-02 03:04:05'
    observer = astropy.coordinates.SkyCoord(10*u.deg, 20*u.deg, 30*u.R_sun,
                                            frame='heliographic_stonyhurst')
    c = astropy.coordinates.SkyCoord(
        10*u.deg, 30*u.deg, 5*u.au, frame='pspframe',
        obstime=obstime, observer=observer)
    
    c2 = c.transform_to('helioprojective')
    
    c3 = c2.transform_to('pspframe')
    
    assert np.isclose(c.lon, c3.lon)
    assert np.isclose(c.lat, c3.lat)
    assert np.isclose(c.distance, c3.distance)
    assert c3.obstime == obstime
    
    c4 = c3.transform_to('helioprojective')
    
    assert np.isclose(c2.Tx, c4.Tx)
    assert np.isclose(c2.Ty, c4.Ty)
    assert np.isclose(c2.distance, c4.distance)


def test_hci_psporbital_conversion(mocker):
    # The PSP kernels are very large and not included in this repo
    mocker.patch('wispr_analysis.orbital_frame.planets.get_orbital_elements',
                return_value=(
                    24817701.61920938 * u.km,
                    0.6994548150099811,
                    0.41291772 * u.rad,
                    0.14661706 * u.rad,
                    2.42928699 * u.rad,
                    ))
    
    obstime = '2019-09-09 01:01:01'
    c = astropy.coordinates.SkyCoord(80*u.deg, -10*u.deg, 33*u.au,
                 frame='heliocentricinertial', obstime=obstime)
    c = c.transform_to('psporbitalframe')
    
    assert np.isclose(c.lon.to_value(), 66.82770747896976)
    assert np.isclose(c.lat.to_value(), -32.27856834)
    assert np.isclose(c.distance.to_value(), 33)
    assert c.obstime == obstime


def test_psporbital_hci_conversion(mocker):
    # The PSP kernels are very large and not included in this repo
    mocker.patch('wispr_analysis.orbital_frame.planets.get_orbital_elements',
                return_value=(
                    24817701.61920938 * u.km,
                    0.6994548150099811,
                    0.41291772 * u.rad,
                    0.14661706 * u.rad,
                    2.42928699 * u.rad,
                    ))
    
    obstime = '2019-09-09 01:01:01'
    c = astropy.coordinates.SkyCoord(80*u.deg, -10*u.deg, 33*u.au,
                 frame='psporbitalframe', obstime=obstime)
    c = c.transform_to('heliocentricinertial')
    
    assert np.isclose(c.lon.to_value(), 89.89502647497844)
    assert np.isclose(c.lat.to_value(), 13.408571443919827)
    assert np.isclose(c.distance.to_value(), 33)
    assert c.obstime == obstime


def test_hpc_psp_conversion(mocker):
    # The PSP kernels are very large and not included in this repo
    mocker.patch('wispr_analysis.orbital_frame.planets.get_orbital_elements',
                return_value=(
                    24817701.61920938 * u.km,
                    0.6994548150099811,
                    0.41291772 * u.rad,
                    0.14661706 * u.rad,
                    2.42928699 * u.rad,
                    ))
    
    obstime = '2019-09-09 01:01:01'
    observer = astropy.coordinates.SkyCoord(10*u.deg, 20*u.deg, 30*u.R_sun,
                                            frame='heliographic_stonyhurst')
    c = astropy.coordinates.SkyCoord(80*u.deg, -10*u.deg, 33*u.au,
                 frame='helioprojective', obstime=obstime, observer=observer)
    c = c.transform_to('pspframe')
    
    assert np.isclose(c.lon.to_value(), 339.4265894441424)
    assert np.isclose(c.lat.to_value(), -10.901241566609368)
    assert np.isclose(c.distance.to_value(), 33)
    assert c.obstime == obstime


def test_psp_hpc_conversion(mocker):
    # The PSP kernels are very large and not included in this repo
    mocker.patch('wispr_analysis.orbital_frame.planets.get_orbital_elements',
                return_value=(
                    24817701.61920938 * u.km,
                    0.6994548150099811,
                    0.41291772 * u.rad,
                    0.14661706 * u.rad,
                    2.42928699 * u.rad,
                    ))
    
    obstime = '2019-09-09 01:01:01'
    observer = astropy.coordinates.SkyCoord(10*u.deg, 20*u.deg, 30*u.R_sun,
                                            frame='heliographic_stonyhurst')
    c = astropy.coordinates.SkyCoord(80*u.deg, -10*u.deg, 33*u.au,
                 frame='pspframe', obstime=obstime, observer=observer)
    c = c.transform_to('helioprojective')
    
    assert np.isclose(c.Tx.to_value(), -646622.5542439144)
    assert np.isclose(c.Ty.to_value(), -36026.64360996204)
    assert np.isclose(c.distance.to_value(), 33)
    assert c.obstime == obstime


def test_psp_hpc_mismatched_observers():
    obstime = '2019-09-09 01:01:01'
    observer1 = astropy.coordinates.SkyCoord(10*u.deg, 20*u.deg, 30*u.R_sun,
                                            frame='heliographic_stonyhurst')
    observer2 = astropy.coordinates.SkyCoord(20*u.deg, 20*u.deg, 30*u.R_sun,
                                            frame='heliographic_stonyhurst')
    c = astropy.coordinates.SkyCoord(80*u.deg, -10*u.deg, 33*u.au,
                 frame='pspframe', obstime=obstime, observer=observer1)
    with pytest.raises(ValueError, match="Observers are not equal"):
        c.transform_to(sunpy.coordinates.Helioprojective(
            obstime=obstime, observer=observer2))
    
    c = astropy.coordinates.SkyCoord(80*u.deg, -10*u.deg, 33*u.au,
                 frame='helioprojective', obstime=obstime, observer=observer1)
    with pytest.raises(ValueError, match="Observers are not equal"):
        c.transform_to(orbital_frame.PSPFrame(
            obstime=obstime, observer=observer2))
