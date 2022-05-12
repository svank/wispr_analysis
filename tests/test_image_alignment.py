from .. import image_alignment, utils


from itertools import product


from astropy.wcs import WCS
import numpy as np
import pytest


@pytest.mark.parametrize('x', [4.55, 5, 5.45])
@pytest.mark.parametrize('y', [3.55, 4, 4.45])
def test_make_cutout(x, y):
    data = np.arange(100).reshape((10, 10))
    
    cutout, start_x, start_y = image_alignment.make_cutout(
            x, y, data, 3, normalize=False)
    
    np.testing.assert_equal(
            cutout,
            np.array([[34, 35, 36], [44, 45, 46], [54, 55, 56]]))
    
    assert start_x == 4
    assert start_y == 3


def test_make_cutout_normalize():
    data = np.arange(100).reshape((10, 10))
    
    cutout, start_x, start_y = image_alignment.make_cutout(
            5, 4, data, 3)
    
    assert np.all(cutout >= 0)
    assert np.all(cutout <= 1)


def test_make_cutout_out_of_bounds():
    data = np.arange(100).reshape((10, 10))
    
    with pytest.raises(AssertionError):
        cutout, start_x, start_y = image_alignment.make_cutout(
                5, 0, data, 3)
    
    with pytest.raises(AssertionError):
        cutout, start_x, start_y = image_alignment.make_cutout(
                0, 4, data, 3)
    
    with pytest.raises(AssertionError):
        cutout, start_x, start_y = image_alignment.make_cutout(
                9, 4, data, 3)
    
    with pytest.raises(AssertionError):
        cutout, start_x, start_y = image_alignment.make_cutout(
                5, 9, data, 3)
    
    # No exception for these, right at the edge
    cutout, start_x, start_y = image_alignment.make_cutout(
            8, 5, data, 3)
    cutout, start_x, start_y = image_alignment.make_cutout(
            5, 8, data, 3)


def test_fit_star():
    x = np.arange(20)
    y = np.arange(20)
    xx, yy = np.meshgrid(x, y)
    
    x_star = 13.2
    y_star = 9.1
    sig_star = 0.8
    
    data = 2 + .1 * xx - .2 * yy
    data += 4 * np.exp(-( (xx - x_star)**2 + (yy - y_star)**2 )
                        / 2 / sig_star**2)
    np.random.seed(1)
    data += np.random.random(data.shape)
    
    all_stars_x = [x_star, 20]
    all_stars_y = [y_star, 1]
    
    fx, fy, err = image_alignment.fit_star(10, 10, data,
            all_stars_x, all_stars_y)
    
    assert len(err) == 0
    assert fx == pytest.approx(x_star, abs=.1)
    assert fy == pytest.approx(y_star, abs=.1)


@pytest.mark.parametrize('start_at_max', [True, False])
def test_fit_star_multiple_peaks(start_at_max):
    x = np.arange(20)
    y = np.arange(20)
    xx, yy = np.meshgrid(x, y)
    
    x_star = 13.2
    y_star = 8.1
    x_star_2 = 9.2
    y_star_2 = 10.5
    sig_star = 0.8
    
    data = 2 + .1 * xx - .2 * yy
    data += 6 * np.exp(-( (xx - x_star)**2 + (yy - y_star)**2 )
                        / 2 / sig_star**2)
    data += 5 * np.exp(-( (xx - x_star_2)**2 + (yy - y_star_2)**2 )
                        / 2 / sig_star**2)
    np.random.seed(2)
    data += np.random.random(data.shape)
    
    all_stars_x = [x_star, 20]
    all_stars_y = [y_star, 1]
    
    fx, fy, err = image_alignment.fit_star(10, 10, data,
            all_stars_x, all_stars_y, start_at_max=start_at_max)
    
    assert len(err) == 0
    if start_at_max:
        assert fx == pytest.approx(x_star, abs=.1)
        assert fy == pytest.approx(y_star, abs=.1)
    else:
        assert fx == pytest.approx(x_star_2, abs=.1)
        assert fy == pytest.approx(y_star_2, abs=.1)


def test_fit_star_crowded():
    x = np.arange(20)
    y = np.arange(20)
    xx, yy = np.meshgrid(x, y)
    
    x_star = 13.2
    y_star = 9.1
    sig_star = 0.8
    
    data = 2 + .1 * xx - .2 * yy
    data += 4 * np.exp(-( (xx - x_star)**2 + (yy - y_star)**2 )
                        / 2 / sig_star**2)
    
    second_star_x = [
            (6.49, False),
            (6.51, True),
            (10, True),
            (13.49, True),
            (13.51, False),
            ]
    
    second_star_y = [
            (6.49, False),
            (6.51, True),
            (10, True),
            (13.49, True),
            (13.51, False),
            ]
    
    for (x, x_is_crowded), (y, y_is_crowded) in product(
            second_star_x, second_star_y):
        all_stars_x = [x_star, x]
        all_stars_y = [y_star, y]
    
        fx, fy, err = image_alignment.fit_star(10, 10, data,
                all_stars_x, all_stars_y, cutout_size=7)
        
        if x_is_crowded and y_is_crowded:
            assert 'Crowded frame' in err
        else:
            assert 'Crowded frame' not in err
            assert fx == pytest.approx(x_star, abs=.1)
            assert fy == pytest.approx(y_star, abs=.1)


def test_fit_star_no_peak():
    x = np.arange(20)
    y = np.arange(20)
    xx, yy = np.meshgrid(x, y)
    
    data = 2 + .1 * xx - .2 * yy
    np.random.seed(1)
    data += np.random.random(data.shape)
    
    all_stars_x = [10, 20]
    all_stars_y = [10, 1]
    
    fx, fy, err = image_alignment.fit_star(10, 10, data,
            all_stars_x, all_stars_y)
    
    assert 'No peak found' in err


def test_fit_star_too_wide():
    x = np.arange(20)
    y = np.arange(20)
    xx, yy = np.meshgrid(x, y)
    
    x_star = 13.2
    y_star = 9.1
    sig_star = 2.8
    
    data = 2 + .1 * xx - .2 * yy
    data += 4 * np.exp(-( (xx - x_star)**2 + (yy - y_star)**2 )
                        / 2 / sig_star**2)
    np.random.seed(1)
    data += np.random.random(data.shape)
    
    all_stars_x = [x_star, 20]
    all_stars_y = [y_star, 1]
    
    fx, fy, err = image_alignment.fit_star(10, 10, data,
            all_stars_x, all_stars_y)
    
    assert 'Fit too wide' in err
    

def test_fit_star_too_narrow():
    x = np.arange(20)
    y = np.arange(20)
    xx, yy = np.meshgrid(x, y)
    
    x_star = 13.2
    y_star = 9.1
    sig_star = 0.1
    
    data = 2 + .1 * xx - .2 * yy
    data += 10 * np.exp(-( (xx - x_star)**2 + (yy - y_star)**2 )
                        / 2 / sig_star**2)
    np.random.seed(1)
    data += np.random.random(data.shape)
    
    all_stars_x = [x_star, 20]
    all_stars_y = [y_star, 1]
    
    fx, fy, err = image_alignment.fit_star(10, 10, data,
            all_stars_x, all_stars_y)
    
    assert 'Fit too narrow' in err


def test_fit_star_too_close_to_edge():
    x = np.arange(20)
    y = np.arange(20)
    xx, yy = np.meshgrid(x, y)
    
    x_stars = [
            (5.5, True),
            (10, False),
            (14.5, True),
        ]
    y_stars = [
            (5.5, True),
            (10, False),
            (14.5, True),
        ]
    
    for (x_star, x_at_edge), (y_star, y_at_edge) in product(
            x_stars, y_stars):
        sig_star = 0.8
        
        data = 2 + .1 * xx - .2 * yy
        data += 10 * np.exp(-( (xx - x_star)**2 + (yy - y_star)**2 )
                            / 2 / sig_star**2)
        
        all_stars_x = [x_star, 20]
        all_stars_y = [y_star, 1]
        
        fx, fy, err = image_alignment.fit_star(10, 10, data,
                all_stars_x, all_stars_y)
        
        if x_at_edge or y_at_edge:
            assert 'Fitted peak too close to edge' in err
            assert fx == pytest.approx(x_star, abs=.5)
            assert fy == pytest.approx(y_star, abs=.5)
        else:
            assert 'Fitted peak too close to edge' not in err
            assert fx == pytest.approx(x_star, abs=.1)
            assert fy == pytest.approx(y_star, abs=.1)


@pytest.mark.parametrize('binning', [2, 1])
def test_find_stars_in_frame(mocker, binning):
    bin_factor = 2 / binning
    x = np.arange(200 * bin_factor)
    y = np.arange(200 * bin_factor)
    xx, yy = np.meshgrid(x, y)
    
    data = 2 + .1 / bin_factor * xx - .2 / bin_factor * yy
    
    good_stars_x = np.array([20, 30, 50, 20, 80, 60]) * bin_factor
    good_stars_y = np.array([80, 50, 10, 30, 80, 50]) * bin_factor
    for x, y in zip(good_stars_x, good_stars_y):
        sig_star = 0.8 * bin_factor
        data += 5 * np.exp(-( (xx - x)**2 + (yy - y)**2 ) / 2 / sig_star**2)
    
    x_wide, y_wide = 10 * bin_factor, 10 * bin_factor
    sig_star = 3.8 * bin_factor
    data += 5 * np.exp(-( (xx - x_wide)**2 + (yy - y_wide)**2 )
                        / 2 / sig_star**2)
    
    x_wide_2, y_wide_2 = 90 * bin_factor, 10 * bin_factor
    sig_star = 3.8 * bin_factor
    data += 5 * np.exp(-( (xx - x_wide_2)**2 + (yy - y_wide_2)**2 )
                        / 2 / sig_star**2)
    
    x_narrow, y_narrow = 90 * bin_factor, 90 * bin_factor
    sig_star = .01
    data += 15 * np.exp(-( (xx - x_narrow)**2 + (yy - y_narrow)**2 )
                        / 2 / sig_star**2)
    
    # Getting the fit to come out as too narrow is difficult, since we can only
    # control things at the one-pixel level. Cheese up the threshold to make it
    # easier.
    mocker.patch(image_alignment.__name__+'.MIN_SIGMA', 0.3)
    
    
    x_edge, y_edge = 10 * bin_factor, 90 * bin_factor
    x_edge_expected, y_edge_expected = 15 * bin_factor, 90 * bin_factor
    sig_star = .8 * bin_factor
    data += 5 * np.exp(-( (xx - x_edge - 4.8 * bin_factor)**2 + (yy - y_edge)**2 )
                        / 2 / sig_star**2)
    
    x_crowded, y_crowded = 50 * bin_factor, 50 * bin_factor
    
    stars_x = np.concatenate(
            (good_stars_x, [x_wide], [x_wide_2], [x_narrow], [x_edge],
                [x_crowded]))
    stars_y = np.concatenate(
            (good_stars_y, [y_wide], [y_wide_2], [y_narrow], [y_edge],
                [y_crowded]))
    
    all_stars_x = np.concatenate((stars_x, [x_crowded + 1]))
    all_stars_y = np.concatenate((stars_y, [y_crowded + 1]))
    
    ras = 2 * all_stars_x
    decs = 0.5 * all_stars_y
    mocker.patch(image_alignment.__name__+'.prep_frame_for_star_finding',
            return_value=(
                stars_x, stars_y, None,
                ras, decs,
                all_stars_x, all_stars_y, data, binning
            ))
    
    fname = 'psp_L3_wispr_20181101T210030_V3_2222.fits'
    t = utils.to_timestamp(fname)
    good, bad, crowded_out, codes, mapping = image_alignment.find_stars_in_frame(
            (fname, True))
    
    good_rounded = [(round(x), round(y)) for x, y in good]
    assert sorted(zip(good_stars_x, good_stars_y)) == sorted(good_rounded)
    
    bad_rounded = [(round(x), round(y)) for x, y in bad]
    assert sorted(bad_rounded) == sorted([
            (x_wide, y_wide),
            (x_wide_2, y_wide_2),
            (x_narrow, y_narrow),
            (x_edge_expected, y_edge_expected)])
    
    crowded_rounded = [(round(x), round(y))
            for x, y in crowded_out]
    assert crowded_rounded == [(x_crowded, y_crowded)]
    
    for code in codes:
        if 'wide' in code:
            assert codes[code] == 2
        else:
            assert codes[code] == 1
    
    for x, y in good:
        assert mapping[(2*round(x), 0.5*round(y))] == (x, y, t)
    assert len(mapping) == len(good)


def test_load_stars(mocker):
    star_data = [''] * 43
    star_data.append('1;2 30 00;+10 30 00;2')
    star_data.append('2;2 30 30;+10 30 30;5')
    star_data.append('3;12 30 30;-10 30 30;10')
    star_data.append('4;8 00 00;-01 00 00;10')
    star_data.append('')
    
    class MockFile():
        def readlines(self):
            return star_data
    
    mocker.patch(image_alignment.__name__+'.open',
            return_value=MockFile())
    
    stars = image_alignment.load_stars()
    
    data = stars.get_stars(2.5/24*360, 10.5)
    assert len(data) == 2
    assert data[0] == (2.5/24*360, 10.5, 2)
    assert data[1] == ((2.5 + .5/60)/24*360, 10.5 + .5/60, 5)
    
    data = stars.get_stars(12.4/24*360, -10.6)
    assert len(data) == 1
    assert data[0] == ((12.5 + .5/60)/24*360, -10.5 - .5/60, 10)
    
    data = stars.get_stars(0, 0)
    assert len(data) == 0
    
    data = list(stars.stars_between([(2/24*360, 3/24*360)], 10, 11))
    assert len(data) == 2
    assert data[0] == (2.5/24*360, 10.5, 2)
    assert data[1] == ((2.5 + .5/60)/24*360, 10.5 + .5/60, 5)
    
    data = list(stars.stars_between(
        [(2/24*360, 3/24*360), (11/24*360, 13/24*360)], -11, 11))
    assert len(data) == 3
    assert data[0] == (2.5/24*360, 10.5, 2)
    assert data[1] == ((2.5 + .5/60)/24*360, 10.5 + .5/60, 5)
    assert data[2] == ((12.5 + .5/60)/24*360, -10.5 - .5/60, 10)
    
    data = list(stars.stars_between(
        [(2/24*360, 13/24*360)], -11, 11))
    assert len(data) == 4
    assert data[0] == (2.5/24*360, 10.5, 2)
    assert data[1] == ((2.5 + .5/60)/24*360, 10.5 + .5/60, 5)
    assert data[2] == (8/24*360, -1, 10)
    assert data[3] == ((12.5 + .5/60)/24*360, -10.5 - .5/60, 10)


def test_do_iteration_no_crpix():
    w = WCS(naxis=2)
    w.wcs.crpix = 50, 50
    w.wcs.crval = 0, 0
    w.wcs.cdelt = 1, 1
    
    xs = np.arange(-100, 100)
    ys = np.arange(-100, 100)
    np.random.seed(1)
    np.random.shuffle(xs)
    np.random.shuffle(ys)
    
    ras, decs = w.all_pix2world(xs, ys, 0)
    
    w.wcs.crval = 1.2, -0.8
    angle = 0.1
    w.wcs.pc = np.array(
            [[np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]])
    
    _, fitted_angle, fitted_dra, fitted_ddec, fitted_dx, fitted_dy = \
            image_alignment.do_iteration_no_crpix(
            ras, decs, xs, ys, w)
    
    assert fitted_dra == pytest.approx(-w.wcs.crval[0])
    assert fitted_ddec == pytest.approx(-w.wcs.crval[1])
    assert fitted_angle == pytest.approx(-angle)
    assert fitted_dx == 0
    assert fitted_dy == 0


def test_smooth_curve_constant():
    x = np.arange(100)
    y = np.full(100, 10)
    
    y_sm = image_alignment.smooth_curve(x, y, sig=10)
    
    np.testing.assert_allclose(y, y_sm)


def test_smooth_curve_linear():
    x = np.arange(100)
    y = np.arange(100)
    
    y_sm = image_alignment.smooth_curve(x, y, sig=2, n_sig=3)
    
    np.testing.assert_allclose(y[2*3:-2*3], y_sm[2*3:-2*3])
    
    assert np.all(y_sm[:2*3] > y[:2*3])
    assert np.all(y_sm[-2*3:] < y[-2*3:])


def test_smooth_curve_outlier_rejection():
    x = np.arange(100)
    y = np.full(100, 1)
    
    # Add an outlier
    y[50] = 1000
    
    # Find the threshold value of outlier_sig that will reject this outlier
    window = y[50-6:50+7]
    std = np.std(window)
    mean = np.mean(window)
    
    sig_thresh = (y[50] - mean) / std
    
    # If we set outlier_sig below the threshold, the outlier should be rejected
    y_sm = image_alignment.smooth_curve(x, y, sig=6, n_sig=1,
            outlier_sig=sig_thresh * 0.99)
    
    np.testing.assert_allclose(y_sm, 1)
    
    # With outlier_sig above the threshold, the outlier should not be rejected
    y_sm = image_alignment.smooth_curve(x, y, sig=6, n_sig=1,
            outlier_sig=sig_thresh * 1.01)
    
    assert np.all(y_sm[50-6:50+7] > 1)
    np.testing.assert_allclose(y_sm[:50-6], 1)
    np.testing.assert_allclose(y_sm[50+7:], 1)
