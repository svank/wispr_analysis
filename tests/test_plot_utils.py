from .. import plot_utils

from datetime import datetime
import os

from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.testing.conftest import mpl_test_settings
from matplotlib.testing.decorators import image_comparison
import pytest


def test_parse_level_preset():
    assert plot_utils.parse_level_preset('L1') == '1'
    assert plot_utils.parse_level_preset('L2') == '2'
    assert plot_utils.parse_level_preset('L3') == '3'
    assert plot_utils.parse_level_preset('1') == '1'
    assert plot_utils.parse_level_preset('2') == '2'
    assert plot_utils.parse_level_preset('3') == '3'
    assert plot_utils.parse_level_preset(1) == '1'
    assert plot_utils.parse_level_preset(2) == '2'
    assert plot_utils.parse_level_preset(3) == '3'
    assert plot_utils.parse_level_preset(None) == '3'
    
    with pytest.raises(ValueError):
        plot_utils.parse_level_preset(4)
    with pytest.raises(ValueError):
        plot_utils.parse_level_preset('L0')
    with pytest.raises(ValueError):
        plot_utils.parse_level_preset('j')
    
    header = fits.Header()
    header['level'] = 'L3'
    
    assert plot_utils.parse_level_preset(None, header) == '3'
    assert plot_utils.parse_level_preset(2, header) == '2'
    
    header['level'] = 'L2'
    assert plot_utils.parse_level_preset(None, header) == '2'
    assert plot_utils.parse_level_preset(1, header) == '1'
    
    header['level'] = 'bad'
    assert plot_utils.parse_level_preset(None, header) == '3'
    assert plot_utils.parse_level_preset(1, header) == '1'


@image_comparison(baseline_images=['test_plot_orbit'], extensions=['pdf'],
        tol=10)
def test_plot_orbit():
    dir_path = os.path.join(os.path.dirname(__file__),
                'test_data', 'WISPR_files_headers_only')
    plot_utils.plot_orbit(dir_path)
    
    return plt.gcf()


@image_comparison(baseline_images=['test_x_axis_dates'], extensions=['pdf'])
def test_x_axis_dates():
    dates = ['20200101T000000', '20200105T120000', '20200110T010000']
    y = [1, 0, 2]
    
    x = plot_utils.x_axis_dates(dates)
    plt.plot(x, y)
    
    return plt.gcf()


@image_comparison(baseline_images=['test_y_axis_dates'], extensions=['pdf'])
def test_y_axis_dates():
    dates = ['20200101T000000', '20200105T120000', '20200110T010000']
    y = [1, 0, 2]
    
    x = plot_utils.y_axis_dates(dates)
    plt.plot(y, x)
    
    return plt.gcf()


@image_comparison(baseline_images=['test_x_axis_dates_setting_fig_ax'],
        extensions=['pdf'])
def test_x_axis_dates_setting_fig_ax():
    fig1 = plt.figure()
    fig2 = plt.figure()
    plt.figure(fig2.number)
    dates = ['20200101T000000', '20200105T120000', '20200110T010000']
    y = [1, 0, 2]
    
    ax = fig1.add_subplot(1, 1, 1)
    x = plot_utils.x_axis_dates(dates, ax=ax, fig=fig1)
    ax.plot(x, y)
    
    plt.close(fig2)
    return fig1


def test_date_to_mdate():
    assert plot_utils.date_to_mdate('19700101T000000') == 0
    assert plot_utils.date_to_mdate('19700101T060000') == .25
    
    timestamps = ['2021-02-03T12:13:14.5',
                  '2022-02-12T12:14:14.5',
                  '2023-02-01T12:15:14.5',
                  '2023-04-02T12:16:14.5']
    
    assert (plot_utils.date_to_mdate(timestamps)
            == [plot_utils.date_to_mdate(t) for t in timestamps])
