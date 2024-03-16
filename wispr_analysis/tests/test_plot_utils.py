from .. import plot_utils, utils
import os

from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_parse_level_preset():
    assert plot_utils.parse_level_preset('L1') == '1'
    assert plot_utils.parse_level_preset('L2') == '2'
    assert plot_utils.parse_level_preset('L3') == '3'
    assert plot_utils.parse_level_preset('LW') == 'W'
    assert plot_utils.parse_level_preset('1') == '1'
    assert plot_utils.parse_level_preset('2') == '2'
    assert plot_utils.parse_level_preset('3') == '3'
    assert plot_utils.parse_level_preset('W') == 'W'
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
    
    header['level'] = 'LW'
    assert plot_utils.parse_level_preset(None, header) == 'W'
    assert plot_utils.parse_level_preset(1, header) == '1'


@pytest.mark.mpl_image_compare
def test_plot_orbit(mocker):
    times = np.array([
        1.54102122e+09, 1.54108602e+09, 1.54115082e+09, 1.54121562e+09,
        1.54128042e+09, 1.54134522e+09, 1.54141002e+09, 1.54147482e+09])
    poses = SkyCoord(
        np.array([
            35950077.11211421, 33633933.9862569 , 30826470.80720564,
            27463317.06427251, 23492294.88034739, 18893588.41288763,
            13705306.10463386,  8042152.2194448 ]) * u.km,
        np.array([
            -3134222.36970821,  1312105.23274597,  5736926.49596144,
            10055181.02164345, 14146735.13435697, 17855169.61531039,
            21000776.17243724, 23414024.55029698]) * u.km,
        np.array([
            -306250.77413144,  -582840.95529877,  -850753.76931808,
            -1103110.62721629, -1330762.40654031, -1522499.36320502,
            -1666331.83426047, -1752099.43801478]) * u.km,
        representation_type='cartesian',
        frame='heliocentricinertial',
        obstime='2018-11-06T03:27:00.000')
    mocker.patch('wispr_analysis.planets.trace_psp_orbit',
                 return_value=(poses, times))
    mocker.patch('wispr_analysis.orbital_frame.planets.get_orbital_elements',
                 return_value=(
                    24818507.23190053 * u.km,
                    0.6994392466036299,
                    0.07120535 * u.rad,
                    2.93536417 * u.rad,
                    4.58828605 * u.rad,
                    ))
    dir_path = utils.test_data_path('WISPR_files_headers_only')
    plot_utils.plot_orbit(dir_path)
    
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_x_axis_dates():
    dates = ['20200101T000000', '20200105T120000', '20200110T010000']
    y = [1, 0, 2]
    
    x = plot_utils.x_axis_dates(dates)
    plt.plot(x, y)
    
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_y_axis_dates():
    dates = ['20200101T000000', '20200105T120000', '20200110T010000']
    y = [1, 0, 2]
    
    x = plot_utils.y_axis_dates(dates)
    plt.plot(y, x)
    
    return plt.gcf()


@pytest.mark.mpl_image_compare
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