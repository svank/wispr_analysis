from .. import composites, plot_utils, projections, utils

from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
from pytest import approx


@pytest.fixture
def hpr_wcs():
    ref_pa = 90
    ref_elongation = 40
    ref_x = 50
    ref_y = 100
    dpa = .2
    delongation = .3
    
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.ctype = 'HPLN-CAR', 'HPLT-CAR'
    wcs_in.wcs.crpix = 100, 200
    wcs_in.wcs.crval = 10, 0
    wcs_in.wcs.cdelt = 2, 1.5
    
    wcs = projections.HprWcs(wcs_in,
            ref_pa=ref_pa, ref_y=ref_y, dpa=dpa,
            ref_elongation=ref_elongation, ref_x=ref_x, delongation=delongation)
    
    return wcs


def test_radial_transformer(hpr_wcs):
    # Test one known coordinate, the reference pixel for the output. It's on
    # the equator, so it's easy to know what HPLN and HPLT it maps to.
    lon, lat = hpr_wcs.pixel_to_world_values(hpr_wcs.ref_x, hpr_wcs.ref_y)
    assert lon == approx(hpr_wcs.ref_elongation)
    assert lat == approx(0)
    
    # From there, add a grid of dx values and compute what they should map to.
    # Still on the equator, so an easy computation to check
    dx = np.arange(0, 200, 10).reshape((10, 2))
    dy = np.zeros_like(dx)
    px = hpr_wcs.ref_x + dx
    py = hpr_wcs.ref_y + dy
    
    lon, lat = hpr_wcs.pixel_to_world_values(px, py)
    
    np.testing.assert_allclose(
        lon,
        hpr_wcs.ref_elongation
            + (dx * hpr_wcs.delongation))
    np.testing.assert_allclose(lat, 0, atol=1e-8)
    
    # Now, offset in position angle. At this point, we need to compute
    # great-circle distances and azimuths to check the result.
    dx = np.array([10, 10])
    dy = np.array([-10, 10])
    px = hpr_wcs.ref_x + dx
    py = hpr_wcs.ref_y + dy
    
    lon, lat = hpr_wcs.pixel_to_world_values(px, py)
    lon *= np.pi/180
    lat *= np.pi/180
    
    computed_elongation = np.arccos(np.sin(lat) * np.sin(0)
            + np.cos(lat) * np.cos(0) * np.cos(lon))
    computed_pa = np.arctan2(np.cos(lat) * np.sin(lon), np.sin(lat))
    
    np.testing.assert_allclose(
            computed_elongation * 180 / np.pi,
            hpr_wcs.ref_elongation + dx * hpr_wcs.delongation)
    np.testing.assert_allclose(
            computed_pa * 180 / np.pi,
            hpr_wcs.ref_pa + dy * hpr_wcs.dpa)


def test_radial_wcs_hpr_to_pix(hpr_wcs):
    x, y = hpr_wcs.hpr_to_pix(hpr_wcs.ref_elongation, hpr_wcs.ref_pa)
    assert x == approx(hpr_wcs.ref_x)
    assert y == approx(hpr_wcs.ref_y)
    
    x, y = hpr_wcs.hpr_to_pix(
        hpr_wcs.ref_elongation + hpr_wcs.delongation,
        hpr_wcs.ref_pa + hpr_wcs.dpa)
    assert x == approx(hpr_wcs.ref_x + 1)
    assert y == approx(hpr_wcs.ref_y + 1)


def test_radial_wcs_pix_to_hpr(hpr_wcs):
    el, pa = hpr_wcs.pix_to_hpr(hpr_wcs.ref_x, hpr_wcs.ref_y)
    assert el == approx(hpr_wcs.ref_elongation)
    assert pa == approx(hpr_wcs.ref_pa)
    
    el, pa = hpr_wcs.pix_to_hpr(hpr_wcs.ref_x + 1, hpr_wcs.ref_y + 1)
    assert el == approx(hpr_wcs.ref_elongation + hpr_wcs.delongation)
    assert pa == approx(hpr_wcs.ref_pa + hpr_wcs.dpa)


@pytest.mark.parametrize('pa_of_ecliptic', [0, 90, 100, 180])
def test_radial_wcs_hpr_hpc_roundtrip(pa_of_ecliptic, mocker, hpr_wcs):
    mocker.patch.object(
        projections.HprWcs, 'pa_of_ecliptic', pa_of_ecliptic)
    
    lat = np.linspace(-90, 90)
    lon = np.linspace(-100, 100)
    
    LN, LT = np.meshgrid(lon, lat)
    
    LN2, LT2 = hpr_wcs.hpr_to_hpc(*hpr_wcs.hpc_to_hpr(LN, LT))
    
    # A longitude isn't defined for the poles, so don't compare those.
    np.testing.assert_allclose(LN[1:-1], LN2[1:-1])
    np.testing.assert_allclose(LT, LT2)


@pytest.mark.parametrize('pa_of_ecliptic', [0, 90, 100, 180])
def test_radial_wcs_hpc_hpr_roundtrip(pa_of_ecliptic, mocker, hpr_wcs):
    mocker.patch.object(
        projections.HprWcs, 'pa_of_ecliptic', pa_of_ecliptic)
    
    elongation = np.linspace(0, 89)
    pa = np.linspace(-100, 100)
    
    E, PA = np.meshgrid(elongation, pa)
    
    E2, PA2 = hpr_wcs.hpc_to_hpr(*hpr_wcs.hpr_to_hpc(E, PA))
    
    # A position angle isn't defined for zero elongation, so don't compare there
    np.testing.assert_allclose(PA[:, 1:] % 360, PA2[:, 1:] % 360)
    np.testing.assert_allclose(E, E2)


@pytest.mark.parametrize('pa_of_ecliptic', [0, 90, 100, 180])
def test_radial_wcs_pix_world_roundtrip(pa_of_ecliptic, mocker, hpr_wcs):
    mocker.patch.object(
        projections.HprWcs, 'pa_of_ecliptic', pa_of_ecliptic)
    
    x = np.linspace(0, 100)
    y = np.linspace(0, 100)
    
    X, Y = np.meshgrid(x, y)
    
    X2, Y2 = hpr_wcs.world_to_pixel_values(
        *hpr_wcs.pixel_to_world_values(X, Y))
    
    np.testing.assert_allclose(X, X2, atol=1e-8)
    np.testing.assert_allclose(Y, Y2, atol=1e-8)


@pytest.mark.parametrize('pa_of_ecliptic', [0, 90, 100, 180])
def test_radial_wcs_world_pix_roundtrip(pa_of_ecliptic, mocker, hpr_wcs):
    mocker.patch.object(
        projections.HprWcs, 'pa_of_ecliptic', pa_of_ecliptic)
    
    lon = np.linspace(-100, 100)
    lat = np.linspace(-85, 85)
    
    LON, LAT = np.meshgrid(lon, lat)
    
    LON2, LAT2 = hpr_wcs.pixel_to_world_values(
        *hpr_wcs.world_to_pixel_values(LON, LAT))
    
    np.testing.assert_allclose(LON, LON2, atol=1e-8)
    np.testing.assert_allclose(LAT, LAT2, atol=1e-8)


def test_radial_wcs_pa_of_ecliptic(mocker, hpr_wcs):
    for pa_of_ecliptic in [-90, -50, -10, 0, 10, 50, 90]:
        mocker.patch.object(
            projections.HprWcs, 'pa_of_ecliptic', pa_of_ecliptic)
        
        e, pa = hpr_wcs.hpc_to_hpr(50, 0)
        assert e == approx(50)
        assert pa == approx(pa_of_ecliptic, rel=1e-4)
        
        e, pa = hpr_wcs.hpc_to_hpr(.1, .1)
        assert e == approx(np.sqrt(.1**2 + .1**2))
        assert pa == approx(pa_of_ecliptic - 45, rel=1e-4)
        
        e, pa = hpr_wcs.hpc_to_hpr(.1, -.1)
        assert e == approx(np.sqrt(.1**2 + .1**2))
        assert pa == approx(pa_of_ecliptic + 45, rel=1e-4)


@pytest.mark.mpl_image_compare
def test_reproject_to_radial():
    data = np.zeros((500, 500))
    x = np.arange(-data.shape[0]//2, data.shape[0]//2)
    y = np.arange(-data.shape[0]//2, data.shape[0]//2)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    T = np.arctan2(Y, X) * 180 / np.pi

    for r in range(10, data.shape[0], 50):
        data[(R > r-10) * (R < r+10)] = 1

    for t in range(-210, 210, 30):
        data[(T > t-4) * (T < t+4)] = 1
    
    data *= (1 + T / 365)

    wcs_in = WCS(naxis=2)
    wcs_in.wcs.ctype = 'HPLN-ARC', 'HPLT-ARC'
    wcs_in.wcs.crpix = data.shape[0]//2, data.shape[1]//2
    wcs_in.wcs.crval = 0, 0
    wcs_in.wcs.cdelt = .3, .3

    out, trans = projections.reproject_to_radial(
            data, wcs_in, out_shape=(433, 500), delongation=.39,
            ref_elongation=0, ref_x=250, ref_pa=0, dpa=.84, ref_y=217)
    
    plt.imshow(out, cmap='viridis', vmin=0)
    
    projections.label_radial_axes(trans)
    
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_reproject_from_radial():
    data = np.zeros((500, 500))
    x = np.arange(0, data.shape[0])
    y = np.arange(0, data.shape[1])
    X, Y = np.meshgrid(x, y)

    for x in range(0, data.shape[1]+1, int(30/.39)):
        data[(X > x-3) * (X < x+3)] = 1

    for y in range(0, data.shape[0]+1, int(30/.84)):
        data[(Y > y-3) * (Y < y+3)] = 1

    data *= (1 + X / 100)
    
    # Negative elongations should not be used
    data[:, :250] = 9

    wcs_in = WCS(naxis=2)
    wcs_in.wcs.ctype = 'HPLN-ARC', 'HPLT-ARC'
    wcs_in.wcs.crpix = data.shape[0]//2, data.shape[1]//2
    wcs_in.wcs.crval = 0, 0
    wcs_in.wcs.cdelt = .3, .3

    
    out, itrans = projections.reproject_from_radial(data, wcs_in,
            out_shape=data.shape, delongation=.39, ref_elongation=0,
            ref_x=250, ref_pa=0, dpa=.84, ref_y=217)
    
    plt.imshow(out, cmap='viridis', vmin=0)
    
    return plt.gcf()


@pytest.mark.parametrize('use_inner_as_ref', [True, False])
@pytest.mark.parametrize('pass_wcs', [True, False])
def test_produce_radec_for_hp_wcs(pass_wcs, use_inner_as_ref):
    dir = os.path.join(
            utils.test_data_path(),
            'WISPR_files_headers_only', '20181101')
    ifile = os.path.join(dir, 'psp_L3_wispr_20181101T051548_V3_1221.fits')
    ofile = os.path.join(dir, 'psp_L3_wispr_20181101T060030_V3_2222.fits')
    
    with utils.ignore_fits_warnings():
        ihdr = fits.getheader(ifile)
        ohdr = fits.getheader(ofile)
    for hdr in [ihdr, ohdr]:
        hdr['NAXIS'] = 2
        hdr['NAXIS1'] = 960
        hdr['NAXIS2'] = 1024
    
    wcs_hp, _, _ = composites.gen_header(ihdr, ohdr)
    wcs_hp = wcs_hp[250:, 350:]
    
    if use_inner_as_ref:
        ref_hdr = ihdr
    else:
        ref_hdr = ohdr
    
    with utils.ignore_fits_warnings():
        ref_wcs_hp = WCS(ref_hdr)
        ref_wcs_ra = WCS(ref_hdr, key='A')
    
    if pass_wcs:
        wcs_ra = projections.produce_radec_for_hp_wcs(
                wcs_hp, ref_wcs_hp, ref_wcs_ra)
    else:
        wcs_ra = projections.produce_radec_for_hp_wcs(wcs_hp, ref_hdr=ref_hdr)
        
    pts_x = np.linspace(0, 1400, 6)
    pts_y = np.linspace(0, 2100, 6)
    pts_x, pts_y = np.meshgrid(pts_x, pts_y)
    
    computed_ra = wcs_ra.all_pix2world(pts_x, pts_y, 0)
    
    pts_hp = wcs_hp.all_pix2world(pts_x, pts_y, 0)
    pts_ref_xy = ref_wcs_hp.all_world2pix(*pts_hp, 0)
    real_ra = ref_wcs_ra.all_pix2world(*pts_ref_xy, 0)
    
    # We may end up with NaNs (when computing pts_ref_xy) if the HP coordinates
    # are outside the bounds of the projection of the reference image.
    ok = np.isfinite(real_ra)
    np.testing.assert_allclose(
            np.array(computed_ra)[ok], np.array(real_ra)[ok], atol=.2)


@pytest.mark.mpl_image_compare
def test_overlay_radial_grid():
    file = os.path.join(utils.test_data_path(),
                        'WISPR_files_with_data_half_size_L3',
                        '20190405',
                        'psp_L3_wispr_20190405T011515_V3_1221.fits')
    with utils.ignore_fits_warnings():
        image = fits.getdata(file)
        wcs = WCS(fits.getheader(file))
    plot_utils.plot_WISPR(image, wcs=wcs)
    projections.overlay_radial_grid(image, wcs)
    return plt.gcf()