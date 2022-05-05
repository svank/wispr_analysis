from .. import composites

from astropy.wcs import WCS


def test_find_bounds():
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 1, 1
    # Add an offset to avoid anything landing right at pixel boundaries and so
    # having to care about floating-point error
    wcs_in.wcs.crval = 0.1, 0.1
    wcs_in.wcs.cdelt = 1, 1
    wcs_in.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = 1, 1
    wcs_out.wcs.crval = 0, 0
    wcs_out.wcs.cdelt = 1, 1
    wcs_out.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    header = wcs_in.to_header()
    header['NAXIS1'] = 10
    header['NAXIS2'] = 12
    
    bounds = composites.find_bounds(header, wcs_out)
    
    assert bounds == (0, 10, 0, 12)
    
    bounds = composites.find_bounds(header, wcs_out, trim=(1,2,4,5))
    
    assert bounds == (1, 8, 4, 7)
