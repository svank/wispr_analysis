from .. import data_wrangling
from .. import utils

import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import pytest


def test_convert_to_compressed_hdul(tmp_path):
    path = os.path.join(os.path.dirname(__file__),
            'test_data',
            'WISPR_files_with_distortion_map',
            'psp_L2_wispr_20220527T230018_V1_1221.fits')
    with utils.ignore_fits_warnings():
        hdul_orig = fits.open(path)
    hdul_compressed = data_wrangling.convert_to_compressed_hdul(hdul_orig)
    
    output_path = tmp_path / 'compressed_file.fits'
    hdul_compressed.writeto(output_path)
    
    with utils.ignore_fits_warnings():
        hdul_compressed_loaded = fits.open(output_path)
    
    assert isinstance(hdul_compressed_loaded[0], fits.PrimaryHDU)
    
    orig_data = hdul_orig[0].data
    orig_data[np.isnan(orig_data)] = 0
    np.testing.assert_allclose(
            orig_data, hdul_compressed_loaded[1].data, atol=3e-15)
    
    np.testing.assert_array_equal(
            hdul_orig[1].data, hdul_compressed_loaded[2].data)
    np.testing.assert_array_equal(
            hdul_orig[2].data, hdul_compressed_loaded[3].data)
    
    with utils.ignore_fits_warnings():
        wcs_orig = WCS(hdul_orig[0].header, hdul_orig)
        wcs_comp = WCS(hdul_compressed_loaded[1].header, hdul_compressed_loaded)
    
    for pix in [(10, 30), (520, 380), (480, 920)]:
        coord_orig = wcs_orig.pixel_to_world_values(*pix)
        coord_comp = wcs_comp.pixel_to_world_values(*pix)
        assert coord_orig == coord_comp

