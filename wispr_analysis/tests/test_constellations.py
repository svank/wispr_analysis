from .. import plot_utils, utils

import os

import matplotlib.pyplot as plt
import pytest


@pytest.mark.mpl_image_compare
def test_plot_constellations():
    dir_path = utils.test_data_path(
        'WISPR_files_with_data_half_size', '20181101',
        'psp_L2_wispr_20181101T004530_V3_2222.fits')
    plot_utils.plot_WISPR(dir_path, draw_constellations=True)
    
    # Note: the testing environment seems to be expanding the axis ranges, but
    # that doesn't happen in actual use. Go figure.
    return plt.gcf()