from .. import orbital_plane_slices as ops
from ... import utils, planets

import os

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
    global jmap_cache
    if jmap_cache is None:
        mocker.patch(
            ops.__name__+'.planets.spice.str2et',
            return_value=631152801.1839219)
        mocker.patch(
            ops.__name__+'.planets.spice.spkezr',
            return_value=(
                np.array([2.66925117e+07, -9.53139787e+07, 6.40231055e+06,
                        1.48270646e+01, 2.61921828e+01, -1.77890244e+00]),
                330.85513358725734))

        dir_path = os.path.join(utils.test_data_path(),
                                'WISPR_files_with_data_half_size_L3')
        ifiles, _ = utils.collect_files(dir_path)
        
        bundle = ops.load_files(ifiles)
        
        jmap_cache = ops.extract_slices(bundle, 100, 'test')
        
    return jmap_cache.deepcopy()


@pytest.mark.array_compare
def test_extract_slices(jmap):
    assert jmap.title_stub == 'test'
    return jmap.slices


@pytest.mark.array_compare
def test_extract_slices_angles(jmap):
    return np.vstack((jmap.angles, jmap.fixed_angles))


@pytest.mark.array_compare
def test_extract_slices_venus(mocker):
    try:
        # For this test we need enough kernels loaded to locate Venus, but
        # still need to mock things out when loading PSP's position, since
        # those kernels are very large.
        planets.load_kernels(
            os.path.join(utils.data_path(), 'spice_kernels'),
            force=True)
        spkezr = ops.planets.spice.spkezr
        str2et = ops.planets.spice.str2et
        mocker.patch(
            ops.__name__+'.planets.spice.str2et',
            return_value=631152801.1839219)
        mocker.patch(
            ops.__name__+'.planets.spice.spkezr',
            return_value=(
                np.array([2.66925117e+07, -9.53139787e+07, 6.40231055e+06,
                        1.48270646e+01, 2.61921828e+01, -1.77890244e+00]),
                330.85513358725734))
        # Do the part that requires PSP's position
        ops.OrbitalSliceWCS._init_orbital_north()
        # Un-mock the functions
        ops.planets.spice.spkezr = spkezr
        ops.planets.spice.str2et = str2et
        
        dir_path = os.path.join(utils.test_data_path(),
                                'WISPR_files_with_data_half_size_L3')
        ifiles, _ = utils.collect_files(dir_path)
        
        bundle = ops.load_files(ifiles)
        
        jmap = ops.extract_slices(bundle, 100, 'test')
        assert jmap.title_stub == 'test'
        
        return np.stack((jmap.venus_elongations, jmap.venus_angles))
    finally:
        spice.kclear()


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
    assert 'local col detrend(1, 3px)' in jmap.title
    return jmap.slices


@pytest.mark.array_compare
def test_per_row_normalize(jmap):
    jmap.per_row_normalize()
    assert 'row-normalized' in jmap.title
    return jmap.slices


@pytest.mark.array_compare(filename='test_resample_time.txt')
@pytest.mark.parametrize('explicit_times', [True, False])
def test_resample_time(jmap, explicit_times):
    original_times = jmap.times
    if explicit_times:
        jmap.resample_time(250, jmap.times[0], jmap.times[-1])
    else:
        jmap.resample_time(250)
        
    assert "resampled dt=250" in jmap.title
    np.testing.assert_allclose(
        jmap.times,
        np.arange(original_times[0], original_times[-1]+1, 250))
    return jmap.slices


@pytest.mark.array_compare
def test_resample_time_venus(mocker):
    try:
        # For this test we need enough kernels loaded to locate Venus, but
        # still need to mock things out when loading PSP's position, since
        # those kernels are very large.
        planets.load_kernels(
            os.path.join(utils.data_path(), 'spice_kernels'),
            force=True)
        spkezr = ops.planets.spice.spkezr
        str2et = ops.planets.spice.str2et
        mocker.patch(
            ops.__name__+'.planets.spice.str2et',
            return_value=631152801.1839219)
        mocker.patch(
            ops.__name__+'.planets.spice.spkezr',
            return_value=(
                np.array([2.66925117e+07, -9.53139787e+07, 6.40231055e+06,
                        1.48270646e+01, 2.61921828e+01, -1.77890244e+00]),
                330.85513358725734))
        # Do the part that requires PSP's position
        ops.OrbitalSliceWCS._init_orbital_north()
        # Un-mock the functions
        ops.planets.spice.spkezr = spkezr
        ops.planets.spice.str2et = str2et
        
        dir_path = os.path.join(utils.test_data_path(),
                                'WISPR_files_with_data_half_size_L3')
        ifiles, _ = utils.collect_files(dir_path)
        
        bundle = ops.load_files(ifiles)
        
        jmap = ops.extract_slices(bundle, 100, 'test')
        
        jmap.resample_time(250)
        
        return np.stack((jmap.venus_elongations, jmap.venus_angles))
    finally:
        spice.kclear()


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
def test_mask_venus(mocker):
    try:
        # For this test we need enough kernels loaded to locate Venus, but
        # still need to mock things out when loading PSP's position, since
        # those kernels are very large.
        planets.load_kernels(
            os.path.join(utils.data_path(), 'spice_kernels'),
            force=True)
        spkezr = ops.planets.spice.spkezr
        str2et = ops.planets.spice.str2et
        mocker.patch(
            ops.__name__+'.planets.spice.str2et',
            return_value=631152801.1839219)
        mocker.patch(
            ops.__name__+'.planets.spice.spkezr',
            return_value=(
                np.array([2.66925117e+07, -9.53139787e+07, 6.40231055e+06,
                        1.48270646e+01, 2.61921828e+01, -1.77890244e+00]),
                330.85513358725734))
        # Do the part that requires PSP's position
        ops.OrbitalSliceWCS._init_orbital_north()
        # Un-mock the functions
        ops.planets.spice.spkezr = spkezr
        ops.planets.spice.str2et = str2et
        
        dir_path = os.path.join(utils.test_data_path(),
                                'WISPR_files_with_data_half_size_L3')
        ifiles, _ = utils.collect_files(dir_path)
        
        bundle = ops.load_files(ifiles)
        
        jmap = ops.extract_slices(bundle, 100, 'test')
        
        jmap.mask_venus(3)
        
        return jmap.slices
    finally:
        spice.kclear()


@pytest.mark.mpl_image_compare
def test_plot(jmap):
    jmap.plot()
    return plt.gcf()


@pytest.mark.array_compare
def test_jmap_derotate(jmap):
    derotated = jmap.derotate(101)
    assert derotated is not jmap
    assert "derotated" in derotated.title
    assert jmap.title in derotated.title
    np.testing.assert_array_equal(
        derotated.times, jmap.times)
    np.testing.assert_array_equal(
        derotated.venus_elongations, jmap.venus_elongations)
    np.testing.assert_array_equal(
        derotated.venus_angles, jmap.venus_angles)
    return derotated.slices