from .. import composites, plot_utils, utils

from copy import deepcopy
import os

from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import pytest
import warnings


@pytest.mark.parametrize('wcs_key', [' ', 'A'])
def test_find_bounds(wcs_key):
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = 1, 1
    wcs_out.wcs.crval = 0, 0
    wcs_out.wcs.cdelt = 1, 1
    wcs_out.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 1, 1
    # Add an offset to avoid anything landing right at pixel boundaries and so
    # having to care about floating-point error
    wcs_in.wcs.crval = 0.1, 0.1
    wcs_in.wcs.cdelt = 1, 1
    wcs_in.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    header = wcs_in.to_header(key=wcs_key)
    header['NAXIS1'] = 10
    header['NAXIS2'] = 12
    
    bounds = composites.find_bounds(header, wcs_out, key=wcs_key)
    
    assert bounds == (0, 10, 0, 12)
    
    bounds = composites.find_bounds(header, wcs_out, trim=(1,2,4,5),
            key=wcs_key)
    
    assert bounds == (1, 8, 4, 7)


def test_find_bounds_coord_bounds():
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = 1, 1
    wcs_out.wcs.crval = 0, 0
    wcs_out.wcs.cdelt = 1, 1
    wcs_out.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 1, 1
    # Add an offset to avoid anything landing right at pixel boundaries and so
    # having to care about floating-point error
    wcs_in.wcs.crval = 0.1, 0.1
    wcs_in.wcs.cdelt = 1, 1
    wcs_in.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    header = wcs_in.to_header()
    header['NAXIS1'] = 10
    header['NAXIS2'] = 12
    
    bounds = composites.find_bounds(
            header, wcs_out, world_coord_bounds=[1, 5, 2, 6])
    
    assert bounds is None
    
    bounds = composites.find_bounds(
            header, wcs_out, world_coord_bounds=[1, 5, None, None])
    
    assert bounds == (1, 5, 0, 12)
    
    bounds = composites.find_bounds(
            header, wcs_out, world_coord_bounds=[None, None, 2, 6])
    
    assert bounds == (0, 10, 2, 6)
    
    bounds = composites.find_bounds(
            header, wcs_out, world_coord_bounds=[None, 4, None, 6])
    
    assert bounds == (0, 4, 0, 6)


@pytest.mark.parametrize('wcs_key', [' ', 'A'])
@pytest.mark.parametrize('use_wrapper', [True, False])
def test_find_bounds_wrap_aware(wcs_key, use_wrapper):
    shape = (180, 360)
    wcs_out = WCS(naxis=2)
    crpix = [shape[1]/2+1, shape[0]/2+1]
    wcs_out.wcs.crpix = crpix
    wcs_out.wcs.crval = 180, 0
    wcs_out.wcs.cdelt = 1, 1
    wcs_out.wcs.ctype = 'RA---CAR', 'DEC--CAR'
    wcs_out.wcs.cunit = 'deg', 'deg'
    
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 10, 10
    # Add an offset to avoid anything landing right at pixel boundaries and so
    # having to care about floating-point error
    wcs_in.wcs.crval = 0.1, 0.1
    wcs_in.wcs.cdelt = 1, 1
    wcs_in.wcs.ctype = "RA---CAR", "DEC--CAR"
    
    header = wcs_in.to_header(key=wcs_key)
    header['NAXIS1'] = 40
    header['NAXIS2'] = 20
    
    if use_wrapper:
        bounds = composites.find_bounds_wrap_aware(header, wcs_out, key=wcs_key)
    else:
        bounds = composites.find_bounds(
                header, wcs_out, key=wcs_key, wrap_aware=True)
    
    assert bounds == [(0, 31, 81, 101), (351, 360, 81, 101)]
    
    if use_wrapper:
        bounds = composites.find_bounds_wrap_aware(
                header, wcs_out, trim=(1,2,4,5), key=wcs_key)
    else:
        bounds = composites.find_bounds(
                header, wcs_out, trim=(1,2,4,5), key=wcs_key, wrap_aware=True)
    
    assert bounds == [(0, 29, 85, 96), (352, 360, 85, 96)]
    
    wcs_in.wcs.pc = [
            [np.cos(45*np.pi/180), -np.sin(45*np.pi/180)],
            [np.sin(45*np.pi/180), np.cos(45*np.pi/180)]]
    header = wcs_in.to_header(key=wcs_key)
    header['NAXIS1'] = 40
    header['NAXIS2'] = 20
    
    if use_wrapper:
        bounds = composites.find_bounds_wrap_aware(header, wcs_out, key=wcs_key)
    else:
        bounds = composites.find_bounds(
                header, wcs_out, key=wcs_key, wrap_aware=True)
    
    assert bounds == [(0, 28, 77, 119), (346, 360, 78, 104)]


@pytest.mark.parametrize('wcs_key', [' ', 'A'])
def test_find_collective_bounds(wcs_key):
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = 1, 1
    wcs_out.wcs.crval = 0, 0
    wcs_out.wcs.cdelt = 1, 1
    wcs_out.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 1, 1
    # Add an offset to avoid anything landing right at pixel boundaries and so
    # having to care about floating-point error
    wcs_in.wcs.crval = 0.1, 0.1
    wcs_in.wcs.cdelt = 1, 1
    wcs_in.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    header1 = wcs_in.to_header(key=wcs_key)
    header1['NAXIS1'] = 10
    header1['NAXIS2'] = 12
    
    header2 = deepcopy(header1)
    header2['NAXIS2'] = 24
    
    bounds = composites.find_collective_bounds([header1, header2], wcs_out,
            key=wcs_key)
    # Bounds are (0, 10, 0, 12) and (0, 10, 0, 24)
    assert bounds == (0, 10, 0, 24)
    
    header2['CRPIX1' + wcs_key] = 2
    header2['CRPIX2' + wcs_key] = 3
    
    bounds = composites.find_collective_bounds([header1, header2], wcs_out,
            key=wcs_key)
    # Bounds are (0, 10, 0, 12) and (-1, 9, -2, 22)
    assert bounds == (-1, 10, -2, 22)
    
    # Test `trim` values
    bounds = composites.find_collective_bounds([header1, header2], wcs_out,
            trim=(1, 2, 4, 5), key=wcs_key)
    # Bounds are (1, 8, 4, 7) and (0, 7, 2, 17)
    assert bounds == (0, 8, 2, 17)
    
    # Test multiple sub-lists and only one trim value to apply to each
    bounds = composites.find_collective_bounds([[header1], [header2]], wcs_out,
            trim=(1, 2, 4, 5), key=wcs_key)
    # Bounds are (1, 8, 4, 7) and (0, 7, 2, 17)
    assert bounds == (0, 8, 2, 17)
    
    # Test multiple sub-lists and a separate trim value to apply to each
    bounds = composites.find_collective_bounds([[header1], [header2]], wcs_out,
            trim=[(0, 0, 0, 0), (1, 2, 4, 5)], key=wcs_key)
    # Bounds are (0, 10, 0, 12) and (0, 7, 2, 17)
    assert bounds == (0, 10, 0, 17)
    
    bounds = composites.find_collective_bounds([[header1], [header2]], wcs_out,
            trim=[(1, 2, 4, 5), (0, 0, 0, 0)], key=wcs_key)
    # Bounds are (1, 8, 4, 7) and (-1, 9, -2, 22)
    assert bounds == (-1, 9, -2, 22)
    
    # Finally test just one header
    bounds = composites.find_collective_bounds([header1], wcs_out,
            trim=(1, 2, 4, 5), key=wcs_key)
    assert bounds == (1, 8, 4, 7)
    bounds = composites.find_collective_bounds(header1, wcs_out,
            trim=(1, 2, 4, 5), key=wcs_key)
    assert bounds == (1, 8, 4, 7)


def test_gen_header():
    dir_path = utils.test_data_path('WISPR_files_with_data_half_size')
    ifiles, ofiles = utils.collect_files(dir_path)
    with utils.ignore_fits_warnings():
        hdr_i = fits.getheader(ifiles[0])
        hdr_o = fits.getheader(ofiles[0])
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                '.*has more axes \\(2\\) than the image.*')
        wcs, naxis1, naxis2 = composites.gen_header(hdr_i, hdr_o)
    assert list(wcs.wcs.ctype) == ['HPLN-ARC', 'HPLT-ARC']
    assert wcs.wcs.crval[0] == pytest.approx(60.013157)
    assert wcs.wcs.crval[1] == pytest.approx(-10.51129)
    assert list(wcs.wcs.crpix) == [720, 512]
    assert list(wcs.wcs.cdelt) == [0.08461, 0.08461]
    assert (naxis1, naxis2) == (1440, 1024)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                '.*has more axes \\(2\\) than the image.*')
        wcs, naxis1, naxis2 = composites.gen_header(hdr_i, hdr_o, key='A',
                proj='CAR')
    assert list(wcs.wcs.ctype) == ['RA---CAR', 'DEC--CAR']
    assert wcs.wcs.crval[0] == pytest.approx(207.78723)
    assert wcs.wcs.crval[1] == pytest.approx(-10.59083)
    assert list(wcs.wcs.crpix) == [720, 512]
    assert list(wcs.wcs.cdelt) == [-0.08461, 0.08461]
    assert (naxis1, naxis2) == (1440, 1024)


def setup_gen_composite_test():
    dir_path = utils.test_data_path('WISPR_files_with_data_half_size')
    ifiles, ofiles = utils.collect_files(dir_path)
    
    with utils.ignore_fits_warnings():
        hdr_i = fits.getheader(ifiles[0])
        hdr_o = fits.getheader(ofiles[0])
    
    return ifiles[0], ofiles[0], hdr_i, hdr_o


@pytest.mark.mpl_image_compare
def test_gen_composite():
    ifile, ofile, hdr_i, hdr_o = setup_gen_composite_test()
    
    # Disable any trimming or cropping to make it easy to validate the WCS
    composite, wcs = composites.gen_composite(ifile, ofile,
            image_trim=False, bounds=False)
    
    assert (wcs.to_header_string()
            == composites.gen_header(hdr_i, hdr_o)[0].to_header_string())
    
    fig = plt.figure()
    plot_utils.plot_WISPR(composite, wcs=wcs, grid=1)
    return fig


@pytest.mark.mpl_image_compare
def test_gen_composite_with_auto_trim_crop():
    ifile, ofile, hdr_i, hdr_o = setup_gen_composite_test()
    
    composite, wcs = composites.gen_composite(ifile, ofile)
    
    fig = plt.figure()
    plot_utils.plot_WISPR(composite, wcs=wcs, grid=1)
    return fig


@pytest.mark.mpl_image_compare
def test_gen_composite_blank_i():
    ifile, ofile, hdr_i, hdr_o = setup_gen_composite_test()
    
    composite, wcs = composites.gen_composite(ifile, ofile, blank_i=True)
    
    fig = plt.figure()
    plot_utils.plot_WISPR(composite, wcs=wcs, grid=1)
    return fig


@pytest.mark.mpl_image_compare
def test_gen_composite_blank_o():
    ifile, ofile, hdr_i, hdr_o = setup_gen_composite_test()
    
    composite, wcs = composites.gen_composite(ifile, ofile, blank_o=True)
    
    fig = plt.figure()
    plot_utils.plot_WISPR(composite, wcs=wcs, grid=1)
    return fig


@pytest.mark.mpl_image_compare
def test_gen_composite_ra_dec():
    ifile, ofile, hdr_i, hdr_o = setup_gen_composite_test()
    
    composite, wcs = composites.gen_composite(ifile, ofile, key='A')
    
    fig = plt.figure()
    plot_utils.plot_WISPR(composite, wcs=wcs, grid=1)
    return fig


@pytest.mark.mpl_image_compare
def test_gen_composite_custom_trim():
    ifile, ofile, hdr_i, hdr_o = setup_gen_composite_test()
    
    composite, wcs = composites.gen_composite(ifile, ofile,
            image_trim=[[80, 70, 110, 105], [50, 80, 65, 85]])
    
    fig = plt.figure()
    plot_utils.plot_WISPR(composite, wcs=wcs, grid=1)
    return fig


def test_gen_composite_trim_defaults():
    ifile, ofile, hdr_i, hdr_o = setup_gen_composite_test()
    
    composite, wcs = composites.gen_composite(ifile, ofile)
    composite2, wcs2 = composites.gen_composite(ifile, ofile,
            image_trim=[[None, None, None, None], [None, None, None, None]])
    
    np.testing.assert_equal(composite, composite2)
    
    assert wcs.to_header_string() == wcs2.to_header_string()
