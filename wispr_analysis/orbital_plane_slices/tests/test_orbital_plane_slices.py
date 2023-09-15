from .. import orbital_plane_slices as ops
from ... import utils, planets

import os

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


@pytest.mark.array_compare
def test_extract_slices(mocker):
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
    
    jmap = ops.extract_slices(bundle, 100, 'test')
    assert jmap.title_stub == 'test'
    
    return jmap.slices


@pytest.mark.array_compare
def test_extract_slices_angles(mocker):
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
    
    jmap = ops.extract_slices(bundle, 100, 'test')
    assert jmap.title_stub == 'test'
    
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