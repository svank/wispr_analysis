from .. import data_cleaning

import numpy as np
from astropy.io.fits import Header
from astropy.wcs import WCS

import pytest


streak_val = 1000


@pytest.fixture
def random_data():
    # Make random data
    np.random.seed(5555)
    img1 = np.random.random((100, 100))
    img2 = img1 + np.random.normal(.5, .1, (100, 100))
    img3 = img1 + np.random.normal(.5, .2, (100, 100))
    return img1, img2, img3


def assert_no_streak(mask, filtered_image, orig_image):
    # Check that nothing was found
    np.testing.assert_equal(mask, 0)
    # Check that nothing was erased
    np.testing.assert_equal(filtered_image, orig_image)


def assert_streak(mask, streak, filtered_image, orig_image):
    # Check that the streak was found
    expected_streak = np.zeros_like(orig_image)
    expected_streak[streak] = 1
    np.testing.assert_equal(mask, expected_streak)
    
    # Check that the streak was erased
    np.testing.assert_array_less(filtered_image[streak], streak_val)
    # Check that non-streak pixels are unaffected
    orig_image[streak] = 0
    filtered_image[streak] = 0
    np.testing.assert_equal(filtered_image, orig_image)


def test_dust_streak_filter_single_streak(random_data):
    img1, img2, img3 = random_data
    
    # Insert a streak
    streak = (slice(50, 52), slice(30, 60))
    img2[streak] = streak_val
    
    filtered, mask = data_cleaning.dust_streak_filter(
            (img1, None), (img2, None), (img3, None), radec=False,
            return_mask='also', greatest_allowed_gap=False)
    
    assert_streak(mask, streak, filtered, img2)


def test_dust_streak_filter_too_small_streak(random_data):
    img1, img2, img3 = random_data
    
    # Insert a small streak
    min_size = 40
    streak = (slice(50, 51), slice(20, 20 + min_size - 1))
    img2[streak] = streak_val
    
    filtered, mask = data_cleaning.dust_streak_filter(
            (img1, None), (img2, None), (img3, None), radec=False,
            return_mask='also', greatest_allowed_gap=False)
    
    assert_no_streak(mask, filtered, img2)
    

@pytest.mark.parametrize('delta_sig', [-.1, .1])
def test_dust_streak_filter_marginal_significance(random_data, delta_sig):
    img1, img2, img3 = random_data
    
    sliding_window_stride = 1
    window_width = 9
    mean, std = data_cleaning.gen_diffs_distribution(
            img1, img3, (0, 0, 0, 0), sliding_window_stride, window_width)
    
    # Insert a marginal streak
    streak = (slice(50, 54), slice(30, 60))
    larger = np.max((img1, img3), axis=0)
    required_sigma = 5
    img2[streak] = larger[streak] + mean[streak] + (5 + delta_sig) * std[streak]
    
    filtered, mask = data_cleaning.dust_streak_filter(
            (img1, None), (img2, None), (img3, None), radec=False,
            return_mask='also', greatest_allowed_gap=False,
            sliding_window_stride=sliding_window_stride,
            window_width=window_width)
    
    if delta_sig < 0:
        assert_no_streak(mask, filtered, img2)
    else:
        assert_streak(mask, streak, filtered, img2)
    

def test_dust_streak_filter_significance_peak(random_data):
    img1, img2, img3 = random_data
    
    sliding_window_stride = 1
    window_width = 9
    mean, std = data_cleaning.gen_diffs_distribution(
            img1, img3, (0, 0, 0, 0), sliding_window_stride, window_width)
    
    # Insert a faint streak
    streak = (slice(50, 54), slice(30, 60))
    larger = np.max((img1, img3), axis=0)
    img2[streak] = larger[streak] + mean[streak] + 2 * std[streak]
    
    # Add one bright peak
    img2[50, 50] = larger[50, 50] + mean[50, 50] + 5 * std[50, 50]
    
    filtered, mask = data_cleaning.dust_streak_filter(
            (img1, None), (img2, None), (img3, None), radec=False,
            return_mask='also', greatest_allowed_gap=False,
            sliding_window_stride=sliding_window_stride,
            window_width=window_width)
    
    assert_streak(mask, streak, filtered, img2)


def test_dust_streak_filter_separated_segments(random_data):
    img1, img2, img3 = random_data
    
    # Insert a streak
    streak = (slice(50, 52), slice(30, 50))
    img2[streak] = streak_val
    # Break it into two segments by filling in values that are weak spikes
    split = (slice(50, 52), slice(40, 44))
    img2[split] = (img1 + 2*img3)[split]
    
    filtered, mask = data_cleaning.dust_streak_filter(
            (img1, None), (img2, None), (img3, None), radec=False,
            return_mask='also', greatest_allowed_gap=False)
    
    assert_streak(mask, streak, filtered, img2)


def test_dust_streak_filter_too_separated_segments(random_data):
    img1, img2, img3 = random_data
    
    # Insert a streak
    streak = (slice(50, 52), slice(30, 50))
    img2[streak] = streak_val
    # Break it into two segments by filling in values that are weak spikes
    split = (slice(50, 52), slice(40, 45))
    img2[split] = (img1 + 2*img3)[split]
    
    filtered, mask = data_cleaning.dust_streak_filter(
            (img1, None), (img2, None), (img3, None), radec=False,
            return_mask='also', greatest_allowed_gap=False)
    
    assert_no_streak(mask, filtered, img2)


def test_dust_streak_filter_greatest_allowable_gap(random_data):
    img1, img2, img3 = random_data
    
    # Insert a streak
    streak = (slice(50, 52), slice(30, 60))
    img2[streak] = streak_val
    
    # Create headers with a 1-day difference in timestamps
    h1 = Header()
    h1['date-avg'] = '20210101T000000'
    h3 = Header()
    h3['date-avg'] = '20210102T000000'
    
    # Set the 'greatest gap' threshold to 1 day _plus_ 1 second
    filtered, mask = data_cleaning.dust_streak_filter(
            (img1, h1), (img2, None), (img3, h3), radec=False,
            return_mask='also', greatest_allowed_gap=24*60*60+1)
    
    # Streak should be filtered like normal
    assert_streak(mask, streak, filtered, img2)
    
    # Set the 'greatest gap' threshold to 1 day _minus_ 1 second
    filtered, mask = data_cleaning.dust_streak_filter(
            (img1, h1), (img2, None), (img3, h3), radec=False,
            return_mask='also', greatest_allowed_gap=24*60*60-1)
    
    # Streak should not be filtered
    assert_no_streak(mask, filtered, img2)
    

@pytest.mark.parametrize('window_width', [3, 5])
def test_gen_diffs_distribution(window_width):
    img1 = np.arange(10)
    img1 = np.vstack([img1] * 10)
    
    img3 = np.zeros_like(img1)
    
    mean, std = data_cleaning.gen_diffs_distribution(img1, img3,
            (0, 0, 0, 0), sliding_window_stride=1, window_width=window_width)
    
    # Outside of the edges, the mean differences should be the same as the
    # input values, since the averaging will be over something like
    # (x-1, x, x+1)
    edge = window_width // 2
    expected = img1[edge:-edge, edge:-edge]
    expected = np.pad(expected, edge, mode='edge')
    np.testing.assert_equal(mean, expected)
    
    # And that makes the standard deviation constant throughout the image
    np.testing.assert_equal(std, np.std(np.arange(window_width) - edge))
    

def test_gen_diffs_distribution_stride():
    img1 = np.arange(10)
    img1 = np.vstack([img1] * 10)
    
    img3 = np.zeros_like(img1)
    
    mean, std = data_cleaning.gen_diffs_distribution(img1, img3,
            (0, 0, 0, 0), sliding_window_stride=2, window_width=3)
    
    # Alternating rows should be equal (and, in fact, all rows should be equal
    # for this setup)
    np.testing.assert_equal(mean[::2], mean[1::2])
    # Outside the padded edges, alternating columns should be equal
    np.testing.assert_equal(mean[:, 2::2], mean[:, 1:-1:2])
    
    # Outside of the edges, the mean differences should be the same as the
    # input values for every other row, since the averaging will be over
    # something like (x-1, x, x+1)
    expected = img1[1:-1, 1:-1]
    expected = np.pad(expected, 1, mode='edge')
    np.testing.assert_equal(mean[1:-1:2, 1:-1:2], expected[1:-1:2, 1:-1:2])
    
    # And that makes the standard deviation constant throughout the image
    np.testing.assert_equal(std, np.std([-1, 0, 1]))
    

def test_gen_diffs_distribution_stride_2():
    img1 = np.arange(100).reshape((10, 10))
    
    img3 = np.zeros_like(img1)
    
    mean, std = data_cleaning.gen_diffs_distribution(img1, img3,
            (0, 0, 0, 0), sliding_window_stride=2, window_width=3)
    
    # Outside the padded edges, alternating rows should be equal
    np.testing.assert_equal(mean[2::2], mean[1:-1:2])
    # Outside the padded edges, alternating columns should be equal
    np.testing.assert_equal(mean[:, 2::2], mean[:, 1:-1:2])

