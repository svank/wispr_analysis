from .. import composites, projections, utils

from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
from pytest import approx


def get_transformer(inverse=False):
    ref_pa = 90
    ref_elongation = 40
    ref_x = 500
    ref_y = 1000
    dpa = .2
    delongation = .3
    
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.ctype = 'HPLN-CAR', 'HPLT-CAR'
    wcs_in.wcs.crpix = 100, 200
    wcs_in.wcs.crval = 10, 0
    wcs_in.wcs.cdelt = 2, 1.5
    
    constructor = (projections.InverseRadialTransformer
            if inverse else projections.RadialTransformer)
    transformer = constructor(
            ref_pa=ref_pa, ref_y=ref_y, dpa=dpa,
            ref_elongation=ref_elongation, ref_x=ref_x, delongation=delongation,
            wcs_in=wcs_in)
    
    return transformer


def test_radial_transformer():
    transformer = get_transformer()
    # Test one known coordinate, the reference pixel for the output. It's on
    # the equator, so it's easy to know what HPLN and HPLT it maps to.
    pixel_out = np.array(
            [transformer.ref_x, transformer.ref_y]).reshape((1, 1, 2))
    pixel_in = transformer(pixel_out)
    # The resulting pixel coordinates should be those corresponding to a
    # longitude of 40 and latitude of 0.
    np.testing.assert_allclose(
            pixel_in.flatten(),
            transformer.wcs_in.all_world2pix(40, 0, 0))
    
    # From there, add a grid of dx values and compute what they should map to.
    # Still on the equator, so an easy computation to check
    dx = np.arange(0, 200, 10).reshape((10, 2))
    dy = np.zeros_like(dx)
    output_x = transformer.ref_x + dx
    output_y = transformer.ref_y + dy
    pixel_out = np.stack((output_x, output_y), axis=-1)
    
    pixel_in = transformer(pixel_out)
    
    expected_x = transformer.wcs_in.all_world2pix(40, 0, 0)[0]
    expected_x = (expected_x
            + dx * transformer.delongation / transformer.wcs_in.wcs.cdelt[0])
    np.testing.assert_allclose(pixel_in[..., 0], expected_x)
    np.testing.assert_allclose(pixel_in[..., 1],
            transformer.wcs_in.wcs.crpix[1] - 1)
    
    # Now, offset in position angle. At this point, we need to compute
    # great-circle distances and azimuths to check the result.
    dx = np.array([50, 50]).reshape((1, 2))
    dy = np.array([-50, 50]).reshape((1, 2))
    output_x = transformer.ref_x + dx
    output_y = transformer.ref_y + dy
    pixel_out = np.stack((output_x, output_y), axis=-1)
    
    pixel_in = transformer(pixel_out)
    
    lon_in, lat_in = transformer.wcs_in.all_pix2world(
            pixel_in[..., 0], pixel_in[..., 1], 0)
    lon_in *= np.pi / 180
    lat_in *= np.pi / 180
    computed_elongation = np.arccos(np.sin(lat_in) * np.sin(0)
            + np.cos(lat_in) * np.cos(0) * np.cos(lon_in))
    computed_pa = np.arctan2(np.cos(lat_in) * np.sin(lon_in), np.sin(lat_in))
    
    np.testing.assert_allclose(
            computed_elongation * 180 / np.pi,
            transformer.ref_elongation + dx * transformer.delongation)
    np.testing.assert_allclose(
            computed_pa * 180 / np.pi,
            transformer.ref_pa + dy * transformer.dpa)


def test_radial_transformer_all_world2pix():
    transformer = get_transformer()
    assert (transformer.all_world2pix(
                transformer.ref_elongation,
                transformer.ref_pa)
            == (transformer.ref_x, transformer.ref_y))
    
    assert (transformer.all_world2pix(
                transformer.ref_elongation,
                transformer.ref_pa, 1)
            == (transformer.ref_x + 1, transformer.ref_y + 1))
    
    assert (transformer.all_world2pix(
                transformer.ref_elongation + transformer.delongation,
                transformer.ref_pa + transformer.dpa)
            == (transformer.ref_x + 1, transformer.ref_y + 1))
    
    assert (transformer.all_world2pix(
                transformer.ref_elongation + transformer.delongation,
                transformer.ref_pa + transformer.dpa, 1)
            == (transformer.ref_x + 2, transformer.ref_y + 2))


def test_radial_transformer_all_pix2world():
    transformer = get_transformer()
    assert (transformer.all_pix2world(
                transformer.ref_x,
                transformer.ref_y)
            == (transformer.ref_elongation, transformer.ref_pa))
    
    assert (transformer.all_pix2world(
                transformer.ref_x + 1,
                transformer.ref_y + 1, 1)
            == (transformer.ref_elongation, transformer.ref_pa))
    
    assert (transformer.all_pix2world(
                transformer.ref_x + 1,
                transformer.ref_y + 1)
            == (transformer.ref_elongation + transformer.delongation,
                transformer.ref_pa + transformer.dpa))
    
    assert (transformer.all_pix2world(
                transformer.ref_x + 2,
                transformer.ref_y + 2, 1)
            == (transformer.ref_elongation + transformer.delongation,
                transformer.ref_pa + transformer.dpa))


@pytest.mark.parametrize('pa_of_ecliptic', [0, 90, 100, 180])
def test_radial_transformer_hp_elon_roundtrip(pa_of_ecliptic):
    transformer = get_transformer()
    
    transformer.pa_of_ecliptic = pa_of_ecliptic
    
    lat = np.linspace(-90, 90)
    lon = np.linspace(-100, 100)
    
    LN, LT = np.meshgrid(lon, lat)
    
    LN2, LT2 = transformer.elongation_to_hp(
            *transformer.hp_to_elongation(LN, LT))
    
    # A longitude isn't defined for the poles, so don't compare those.
    np.testing.assert_allclose(LN[1:-1], LN2[1:-1])
    np.testing.assert_allclose(LT, LT2)


@pytest.mark.parametrize('pa_of_ecliptic', [0, 90, 100, 180])
def test_radial_transformer_elon_hp_roundtrip(pa_of_ecliptic):
    transformer = get_transformer()
    transformer.pa_of_ecliptic = pa_of_ecliptic
    
    elongation = np.linspace(0, 89)
    pa = np.linspace(-100, 100)
    
    E, PA = np.meshgrid(elongation, pa)
    
    E2, PA2 = transformer.hp_to_elongation(
            *transformer.elongation_to_hp(E, PA))
    
    # A position angle isn't defined for zero elongation, so don't compare there
    np.testing.assert_allclose(PA[:, 1:] % 360, PA2[:, 1:] % 360)
    np.testing.assert_allclose(E, E2)


@pytest.mark.parametrize('pa_of_ecliptic', [0, 90, 100, 180])
def test_radial_transformer_pix_world_roundtrip(pa_of_ecliptic):
    transformer = get_transformer()
    transformer.pa_of_ecliptic = pa_of_ecliptic
    
    x = np.linspace(0, 100)
    y = np.linspace(0, 100)
    
    X, Y = np.meshgrid(x, y)
    
    X2, Y2 = transformer.all_world2pix(
            *transformer.all_pix2world(X, Y, 0), 0)
    
    np.testing.assert_allclose(X, X2)
    np.testing.assert_allclose(Y, Y2)


@pytest.mark.parametrize('pa_of_ecliptic', [0, 90, 100, 180])
def test_radial_transformer_world_pix_roundtrip(pa_of_ecliptic):
    transformer = get_transformer()
    transformer.pa_of_ecliptic = pa_of_ecliptic
    
    pa = np.linspace(-100, 100)
    e = np.linspace(0, 90)
    
    E, PA = np.meshgrid(e, pa)
    
    E2, PA2 = transformer.all_pix2world(
            *transformer.all_world2pix(E, PA, 0), 0)
    
    np.testing.assert_allclose(E, E2, atol=1e-8)
    np.testing.assert_allclose(PA, PA2)


def test_radial_transformer_pa_of_ecliptic():
    transformer = get_transformer()
    for pa_of_ecliptic in [-90, -50, -10, 0, 10, 50, 90]:
        transformer.pa_of_ecliptic = pa_of_ecliptic
        
        e, pa = transformer.hp_to_elongation(50, 0)
        assert e == approx(50)
        assert pa == approx(pa_of_ecliptic, rel=1e-4)
        
        e, pa = transformer.hp_to_elongation(.1, .1)
        assert e == approx(np.sqrt(.1**2 + .1**2))
        assert pa == approx(pa_of_ecliptic - 45, rel=1e-4)
        
        e, pa = transformer.hp_to_elongation(.1, -.1)
        assert e == approx(np.sqrt(.1**2 + .1**2))
        assert pa == approx(pa_of_ecliptic + 45, rel=1e-4)


def test_radial_transformer_roundtrip():
    transformer = get_transformer(inverse=False)
    inv_transformer = get_transformer(inverse=True)
    
    x = np.arange(400, 600, 20)
    y = np.arange(800, 1200, 20)
    
    X, Y = np.meshgrid(x, y)
    
    pixel_out = np.stack((X, Y), axis=-1)
    
    pixel_in = inv_transformer(transformer(pixel_out))
    
    np.testing.assert_allclose(pixel_out, pixel_in)


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
