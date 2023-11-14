import os

import astropy.coordinates
import astropy.units as u
import astropy.wcs
import numpy as np

from .. import planets, orbital_frame, utils


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
    assert isinstance(coord.frame, orbital_frame.PSPOrbitalFrame)
    assert coord.frame.obstime == wcs.wcs.dateobs


def test_orbital_plane_frame_wcs_mapping():
    obstime = '2022-02-03T12:12:12.000'
    frame = orbital_frame.PSPOrbitalFrame(obstime=obstime)
    wcs = orbital_frame.orbital_plane_frame_wcs_mapping(frame, 'CAR')
    assert list(wcs.wcs.ctype) == ['PSLN-CAR', 'PSLT-CAR']
    assert wcs.wcs.dateobs == obstime
    assert wcs.world_axis_physical_types == [
        'custom:pos.psporbitalplane.lon',
        'custom:pos.psporbitalplane.lat']


def test_wcs_from_coord():
    obstime = '2022-02-03T12:12:12.000'
    frame = orbital_frame.PSPOrbitalFrame(obstime=obstime)
    wcs = astropy.wcs.utils.celestial_frame_to_wcs(frame)
    assert list(wcs.wcs.ctype) == ['PSLN-TAN', 'PSLT-TAN']
    assert wcs.wcs.dateobs == obstime


def test_psp_icrs_roundtrip(mocker):
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
    
    c2 = c.transform_to('icrs')
    
    c3 = c2.transform_to('psporbitalframe')
    
    assert np.isclose(c.lon, c3.lon)
    assert np.isclose(c.lat, c3.lat)
    assert np.isclose(c.distance, c3.distance)
    assert c3.obstime == obstime
    
    c4 = c3.transform_to('icrs')
    
    assert np.isclose(c2.ra, c4.ra)
    assert np.isclose(c2.dec, c4.dec)
    assert np.isclose(c2.distance, c4.distance)


def test_icrs_psp_conversion(mocker):
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
                 frame='icrs', obstime=obstime)
    c = c.transform_to('psporbitalframe')
    
    assert np.isclose(c.lon.to_value(), 349.3277074789698)
    assert np.isclose(c.lat.to_value(), -32.27856834)
    assert np.isclose(c.distance.to_value(), 33)
    assert c.obstime == obstime


def test_psp_icrs_conversion(mocker):
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
    c = c.transform_to('icrs')
    
    assert np.isclose(c.ra.to_value(), 165.3498440482125)
    assert np.isclose(c.dec.to_value(), -1.035504901881233)
    assert np.isclose(c.distance.to_value(), 33)
    assert c.obstime == obstime