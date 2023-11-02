from .. import orbital_plane_slices as ops
from ... import utils, planets

import os

from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
import spiceypy as spice


def test_load_files():
    dir_path = os.path.join(utils.test_data_path(),
                            'WISPR_files_with_data_half_size', '20181101')
    ifiles, ofiles = utils.collect_files(dir_path)
    
    bundle = ops.load_files(ifiles)
    
    assert bundle.images.shape[0] == len(ifiles)
    assert len(bundle.wcses) == len(ifiles)
    assert len(bundle.times) == len(ifiles)
    assert bundle.is_inner
    
    bundle = ops.load_files(ofiles)
    
    assert bundle.images.shape[0] == len(ofiles)
    assert len(bundle.wcses) == len(ofiles)
    assert len(bundle.times) == len(ofiles)
    assert not bundle.is_inner


jmap_cache = None

@pytest.fixture
def jmap(mocker):
    # To speed up the tests we generate a Jmap only once
    global jmap_cache
    if jmap_cache is None:
        planets.load_kernels(os.path.join(utils.data_path(), 'spice_kernels'))
        
        dir_path = os.path.join(utils.test_data_path(),
                                'WISPR_files_with_data_half_size_L3')
        ifiles, _ = utils.collect_files(dir_path)
        
        bundle = ops.load_files(ifiles)
        
        # The PSP SPICE kernels are huge and I don't want them in this repo, so
        # here we mock up a lot of SPICE calculations.
        
        # Load in PSP data from the test files
        date_to_sc = {}
        et_to_sc = {}
        with utils.ignore_fits_warnings():
            for ifile in ifiles:
                header = fits.getheader(ifile)
                date = header['date-avg'].replace('T', ' ')
                date_to_sc[date] = (
                    header['hcix_obs'],
                    header['hciy_obs'],
                    header['hciz_obs'],
                    header['hcix_vob'],
                    header['hciy_vob'],
                    header['hciz_vob'],
                    )
                et = spice.str2et(date)
                et_to_sc[et] = date_to_sc[date]
        
        # The code uses SPICE to load the s/c position and velocity (so that
        # the synthetic data doesn't have to generate a full header).
        def locate_psp(date, *args, **kwargs):
            x, y, z, vx, vy, vz = date_to_sc[date]
            return SkyCoord(
                        x=x*u.m,
                        y=y*u.m,
                        z=z*u.m,
                        v_x=vx*u.m/u.s,
                        v_y=vy*u.m/u.s,
                        v_z=vz*u.m/u.s,
                        frame='heliocentricinertial', obstime=date,
                        representation_type='cartesian')
        mocker.patch(
            ops.__name__+'.planets.locate_psp',
            wraps=locate_psp)
        
        # This function needs to be mocked when it's run for PSP, but allowed
        # to go to SPICE for planets.
        real_spkezr = spice.spkezr
        def spkezr(id, et, *args, **kwargs):
            if id == '-96':
                if et == spice.str2et('2020-01-01 12:12:12'):
                    # This is used for determining orbital north
                    return (
                        np.array([2.66925117e+07, -9.53139787e+07, 6.40231055e+06,
                                1.48270646e+01, 2.61921828e+01, -1.77890244e+00]),
                        330.85513358725734)
                x, y, z, vx, vy, vz = et_to_sc[et]
                # Convert to km
                return (x/1000, y/1000, z/1000, vx/1000, vy/1000, vz/1000), 0
            # Pass through for planets
            return real_spkezr(id, et, *args, **kwargs)
        
        mocker.patch(
            ops.__name__+'.planets.spice.spkezr',
            wraps=spkezr)
        
        jmap_cache = ops.extract_slices(bundle, 100, 'test')
        
    return jmap_cache.deepcopy()


@pytest.mark.array_compare
def test_extract_slices(jmap):
    assert jmap.title_stub == 'test'
    return jmap.slices


@pytest.mark.array_compare
def test_extract_slices_angles(jmap):
    fixed_angles = [
        ops.elongation_to_fixed_angle(jmap.angles, jmap.fas_of_sun[i]) 
        for i in range(len(jmap.fas_of_sun))]
    return np.vstack((jmap.angles, fixed_angles))


@pytest.mark.array_compare
def test_extract_slices_venus(jmap):
    venus_angles = [ops.elongation_to_fixed_angle(ve, fa)
                    for ve, fa in zip(jmap.venus_elongations, jmap.fas_of_sun)]
    return np.stack((jmap.venus_elongations, venus_angles))


def test_jmap_trim_nans(jmap):
    slices_orig = jmap.slices.copy()
    jmap.slices[:2] = np.nan
    jmap.slices[-1] = np.nan
    jmap.trim_nans()
    np.testing.assert_array_equal(jmap.slices, slices_orig[2:-1])


@pytest.mark.array_compare
def test_unsharp_mask(jmap):
    jmap.unsharp_mask(3, 1.7)
    assert 'unsharp(3, 1.7)' in jmap.title
    return jmap.slices


@pytest.mark.array_compare
def test_minsmooth(jmap):
    jmap.minsmooth(3, 7)
    assert 'minsmooth(3px, 7)' in jmap.title
    return jmap.slices


@pytest.mark.array_compare
def test_remove_gaussian_blurred(jmap):
    jmap.remove_gaussian_blurred(3)
    assert 'sub_gaussian(3px)' in jmap.title
    return jmap.slices


@pytest.mark.array_compare
def test_per_row_detrend(jmap):
    jmap.per_row_detrend(1)
    assert '1th-order row detrending' in jmap.title
    return jmap.slices


@pytest.mark.array_compare
def test_per_col_detrend(jmap):
    jmap.per_col_detrend(1)
    assert '1th-order col detrending' in jmap.title
    return jmap.slices


@pytest.mark.array_compare
def test_local_col_detrend(jmap):
    jmap.local_col_detrend(1, 3)
    assert 'lcoldet(1, 3px)' in jmap.title
    return jmap.slices


@pytest.mark.array_compare
def test_per_row_normalize(jmap):
    jmap.per_row_normalize()
    assert 'row-normalized' in jmap.title
    return jmap.slices


@pytest.mark.array_compare
def test_fourier_filter(jmap):
    # Ensure there are enough points for the apodization rolloffs
    dt = jmap.times[1] - jmap.times[0]
    jmap.resample_time(dt/3.5)
    jmap.fourier_filter()
    assert 'ffiltered' in jmap.title
    return jmap.slices


@pytest.mark.array_compare(filename='test_resample_time.txt')
@pytest.mark.parametrize('explicit_times', [True, False])
def test_resample_time(jmap, explicit_times):
    original_times = jmap.times
    if explicit_times:
        jmap.resample_time(250, jmap.times[0], jmap.times[-1])
    else:
        jmap.resample_time(250)
        
    assert "resamp dt=250" in jmap.title
    np.testing.assert_allclose(
        jmap.times,
        np.arange(original_times[0], original_times[-1]+1, 250))
    return jmap.slices


@pytest.mark.array_compare
def test_resample_time_venus(jmap):
    jmap.resample_time(250)
    venus_angles = [ops.elongation_to_fixed_angle(ve, fa)
                    for ve, fa in zip(jmap.venus_elongations, jmap.fas_of_sun)]
    return np.stack((jmap.venus_elongations, venus_angles))


def test_clamp(jmap):
    min, max = np.nanpercentile(jmap.slices, [20, 80])
    jmap.clamp(min, max)
    np.nan_to_num(jmap.slices, copy=False, nan=min)
    assert np.all(jmap.slices >= min)
    assert np.all(jmap.slices <= max)


def test_pclamp(jmap):
    min, max = np.nanpercentile(jmap.slices, [20, 80])
    jmap.pclamp(20, 80)
    np.nan_to_num(jmap.slices, copy=False, nan=min)
    assert np.all(jmap.slices >= min)
    assert np.all(jmap.slices <= max)


@pytest.mark.array_compare
def test_percentile_normalize(jmap):
    jmap.percentile_normalize(10, 90)
    assert 'normalized to (10, 90) percentile' in jmap.title
    return jmap.slices


@pytest.mark.array_compare
def test_median_filter(jmap):
    jmap.median_filter(3)
    assert 'med_filt(3)' in jmap.title
    return jmap.slices


@pytest.mark.array_compare
def test_gaussian_filter(jmap):
    jmap.gaussian_filter(3, True)
    assert 'gauss_filt(3)' in jmap.title
    return jmap.slices


@pytest.mark.array_compare
def test_mask_venus(jmap):
    jmap.mask_venus(3)
    return jmap.slices


@pytest.mark.mpl_image_compare
def test_plot(jmap):
    jmap.plot()
    return plt.gcf()


@pytest.mark.array_compare
def test_jmap_derotate(jmap):
    derotated = jmap.derotate(101)
    assert derotated is not jmap
    assert "derot" in derotated.title
    assert jmap.title in derotated.title
    np.testing.assert_array_equal(
        derotated.times, jmap.times)
    np.testing.assert_array_equal(
        derotated.venus_elongations, jmap.venus_elongations)
    return derotated.slices


@pytest.mark.array_compare
def test_jmap_rerotate(jmap):
    derotated = jmap.derotate(101)
    re_rotated = derotated.rotate()
    assert derotated is not re_rotated
    assert derotated is not jmap
    assert "derot" in re_rotated.title
    assert "re-rot" in re_rotated.title
    assert jmap.title in re_rotated.title
    assert derotated.title in re_rotated.title
    np.testing.assert_array_equal(
        re_rotated.times, jmap.times)
    np.testing.assert_array_equal(
        re_rotated.venus_elongations, jmap.venus_elongations)
    return np.vstack((re_rotated.angles, re_rotated.slices))


def test_DerotatedFixedAngleWCS_roundtrip():
    wcs = ops.DerotatedFixedAngleWCS(100, 238, 127)
    input_pixels = np.arange(-2, 130)
    world = wcs.pixel_to_world_values(input_pixels, np.ones_like(input_pixels))
    output_pixels, _ = wcs.world_to_pixel_values(*world)
    np.testing.assert_allclose(input_pixels, output_pixels, equal_nan=True)


def test_RotatedFixedAngleWCS_roundtrip():
    x = np.linspace(0, 1, 121)
    # Make something non-linear
    fixed_angles = 85 + x * 50 + 0.1 * x**2
    fa_of_sun = 25
    elongations = ops.fixed_angle_to_elongation(fixed_angles, fa_of_sun)
    wcs = ops.RotatedFixedAngleWCS(fa_of_sun, elongations, 80)
    input_pixels = np.arange(-2, 130)
    world = wcs.pixel_to_world_values(input_pixels, np.ones_like(input_pixels))
    output_pixels, _ = wcs.world_to_pixel_values(*world)
    
    # Prep comparison to handle the out-of-bounds pixels that come out as nan
    input_pixels = input_pixels.astype(float)
    input_pixels[input_pixels < 0] = np.nan
    input_pixels[input_pixels >= len(x)] = np.nan
    np.testing.assert_allclose(input_pixels, output_pixels, equal_nan=True)