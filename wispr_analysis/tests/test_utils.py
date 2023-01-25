from .. import utils

from datetime import datetime, timezone
import os
import tempfile

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pytest
import warnings


def test_to_timestamp():
    assert (utils.to_timestamp('2021-02-03T12:13:14.5')
            == datetime(
                2021, 2, 3, 12, 13, 14, 500000, timezone.utc).timestamp())
    assert (utils.to_timestamp('20210203T121314')
            == datetime(2021, 2, 3, 12, 13, 14, 0, timezone.utc).timestamp())
    assert (utils.to_timestamp('path/psp_L3_wispr_20210111T083017_V1_1221.fits')
            == datetime(2021, 1, 11, 8, 30, 17, 0, timezone.utc).timestamp())
    assert (utils.to_timestamp(
        'path/psp_L3_wispr_20210111T083017_V1_1221.fits', as_datetime=True)
            == datetime(2021, 1, 11, 8, 30, 17, 0, timezone.utc))
    assert isinstance(utils.to_timestamp(
        'path/psp_L3_wispr_20210111T083017_V1_1221.fits', as_datetime=True),
            datetime)


def test_to_timestamp_list():
    timestamps = ['2021-02-03T12:13:14.5',
                  '2022-02-12T12:14:14.5',
                  '2023-02-01T12:15:14.5',
                  '2023-04-02T12:16:14.5']
    
    assert (utils.to_timestamp(timestamps)
            == [utils.to_timestamp(x) for x in timestamps])
    
    timestamps = ['path/psp_L3_wispr_20210111T083017_V1_1221.fits',
                  'path/psp_L3_wispr_20210211T083017_V1_1221.fits',
                  'path/psp_L3_wispr_20210311T083017_V1_1221.fits']
    
    assert (utils.to_timestamp(timestamps)
            == [utils.to_timestamp(x) for x in timestamps])


def test_get_PSP_path():
    dir_path = os.path.join(os.path.dirname(__file__),
                'test_data', 'WISPR_files_headers_only')
    times, positions, vs = utils.get_PSP_path(dir_path)
    assert times.size == positions.shape[0] == vs.shape[0]
    assert np.all(times[1:] > times[:-1])
    assert positions.shape[1] == 3
    assert vs.shape[1] == 3


def test_collect_files():
    dir_path = os.path.join(os.path.dirname(__file__),
                'test_data', 'WISPR_files_headers_only')
    files = utils.collect_files(dir_path, separate_detectors=True)
    files_avg = utils.collect_files(dir_path, separate_detectors=True,
            order='DATE-AVG')
    files_together = utils.collect_files(dir_path, separate_detectors=False)
    files_together_avg = utils.collect_files(dir_path,
            separate_detectors=False, order='DATE-AVG')
    
    for file_list in (files, files_avg):
        assert len(file_list) == 2
        assert len(file_list[0]) == 100
        assert len(file_list[1]) == 80
        for file in file_list[0]:
            assert 'V3_1' in file
        for file in file_list[1]:
            assert 'V3_2' in file
    
    for file_list in (files_together, files_together_avg):
        assert len(file_list) == 180
    
    for file_list, key in (
            (files[0], 'DATE-BEG'),
            (files[1], 'DATE-BEG'),
            (files_together, 'DATE-BEG'),
            (files_avg[0], 'DATE-AVG'),
            (files_avg[1], 'DATE-AVG'),
            (files_together_avg, 'DATE-AVG')):
        last_timestamp = -1
        for file in file_list:
            header = fits.getheader(file)
            timestamp = datetime.strptime(
                    header[key], "%Y-%m-%dT%H:%M:%S.%f").timestamp()
            assert timestamp > last_timestamp
            last_timestamp = timestamp


def test_collect_files_with_headers():
    dir_path = os.path.join(os.path.dirname(__file__),
                'test_data', 'WISPR_files_headers_only')
    file_list = utils.collect_files(os.path.join(dir_path, '20181101'),
            include_headers=True, separate_detectors=False)
    assert len(file_list) == 58
    assert len(file_list[0]) == 2
    assert isinstance(file_list[0][1], fits.Header)
    
    file_list = utils.collect_files(os.path.join(dir_path, '20181101'),
            include_headers=True, include_sortkey=True,
            separate_detectors=False, order='date-avg')
    assert len(file_list) == 58
    assert len(file_list[0]) == 3
    assert isinstance(file_list[0][2], fits.Header)
    for sortkey, file, header in file_list:
        assert header['DATE-AVG'] == sortkey


def test_collect_files_between():
    dir_path = os.path.join(os.path.dirname(__file__),
                'test_data', 'WISPR_files_headers_only')
    file_list = utils.collect_files(dir_path, separate_detectors=False,
            between=('20181102T000000', None))
    assert len(file_list) == 122
    
    file_list = utils.collect_files(dir_path, separate_detectors=False,
            between=(None, '20181102T000000'))
    assert len(file_list) == 58
    
    file_list = utils.collect_files(dir_path, separate_detectors=False,
            between=('20181101T103000', '20181102T000000'))
    assert len(file_list) == 34


def test_collect_files_between_timestamp_parsing():
    dir_path = os.path.join(os.path.dirname(__file__),
                'test_data', 'WISPR_files_headers_only')
    file_list = utils.collect_files(dir_path, separate_detectors=False,
            between=('20181102T000000', None), order='DATE-AVG')
    assert len(file_list) == 122
    
    file_list = utils.collect_files(dir_path, separate_detectors=False,
            between=(None, '20181102T000000'), order='DATE-AVG')
    assert len(file_list) == 58
    
    file_list = utils.collect_files(dir_path, separate_detectors=False,
            between=('20181101T103000', '20181102T000000'), order='DATE-AVG')
    assert len(file_list) == 34


def test_collect_files_filters():
    dir_path = os.path.join(os.path.dirname(__file__),
                'test_data', 'WISPR_files_headers_only')
    file_list = utils.collect_files(dir_path, separate_detectors=False,
            include_headers=True)
    all_values = np.array([f[1]['dsun_obs'] for f in file_list])
    
    for lower in [32067077000, None]:
        for upper in [34213000000, None]:
            for as_list in [True, False]:
                filters = [('dsun_obs', lower, upper)]
                if not as_list:
                    # A single filter can be passed as one tuple or as a list
                    # of one tuple
                    filters = filters[0]
                file_list = utils.collect_files(dir_path,
                        separate_detectors=False,
                        filters=filters,
                        include_headers=True)
                
                expected = np.ones_like(all_values)
                if lower is not None:
                    expected *= all_values >= lower
                if upper is not None:
                    expected *= all_values <= upper
                
                assert len(file_list) == np.sum(expected)
                headers = [f[1] for f in file_list]
                for h in headers:
                    if lower is not None:
                        assert float(h['dsun_obs']) >= lower
                    if upper is not None:
                        assert float(h['dsun_obs']) <= upper


def test_collect_files_two_filters():
    dir_path = os.path.join(os.path.dirname(__file__),
                'test_data', 'WISPR_files_headers_only')
    file_list = utils.collect_files(dir_path, separate_detectors=False,
            include_headers=True)
    all_values1 = np.array([f[1]['dsun_obs'] for f in file_list])
    all_values2 = np.array([f[1]['xposure'] for f in file_list])
    
    file_list = utils.collect_files(dir_path, separate_detectors=False,
            filters=[('dsun_obs', 32067077000, 34213000000),
                     ('xposure', 100, None)],
            include_headers=True)
    
    e = (all_values1 >= 32067077000) * (all_values1 <= 34213000000)
    expected = e * (all_values2 >= 100)
    # Ensure that the values chosen for the second filter actually have an effect
    assert np.sum(e) != np.sum(expected)
    
    assert len(file_list) > 0
    assert len(file_list) == np.sum(expected)
    
    headers = [f[1] for f in file_list]
    for h in headers:
        assert float(h['dsun_obs']) >= 32067077000
        assert float(h['dsun_obs']) <= 34213000000
        assert float(h['xposure']) >= 100


def test_ensure_data():
    data = np.arange(100).reshape((10, 10))
    wcs = WCS(naxis=2)
    wcs.array_shape = data.shape
    header = wcs.to_header()
    
    data_out, h = utils.ensure_data(data)
    assert data is data_out
    assert h is None
    
    data_out, h = utils.ensure_data((data, header))
    assert data is data_out
    assert h is header
    
    data_out = utils.ensure_data((data, header), header=False)
    assert data is data_out
    
    data_out, h, w = utils.ensure_data((data, header), wcs=True)
    assert data is data_out
    assert h is header
    assert str(w) == str(WCS(h))
    
    data_out, h, w = utils.ensure_data((data, header, wcs),
            header=True, wcs=True)
    assert data is data_out
    assert h is header
    assert w is wcs
    
    data_out, h = utils.ensure_data((data, header, wcs),
            header=True, wcs=False)
    assert data is data_out
    assert h is header
    
    data_out, w = utils.ensure_data((data, header, wcs),
            header=False, wcs=True)
    assert data is data_out
    assert w is wcs
    
    data_out = utils.ensure_data((data, header, wcs),
            header=False, wcs=False)
    assert data is data_out
    
    with tempfile.TemporaryDirectory() as td:
        file = os.path.join(td, 'file.fits')
        fits.writeto(file, data, header=header)
        # Update after io.fits modifies the header
        header = fits.getheader(file)
        
        data_out, h = utils.ensure_data(file)
        assert np.all(data_out == data)
        assert isinstance(h, fits.Header)
        
        data_out, h, w = utils.ensure_data(file, wcs=True)
        assert np.all(data_out == data)
        assert isinstance(h, fits.Header)
        assert str(h) == str(header)
        assert isinstance(w, WCS)
        assert str(w) == str(wcs)


def test_get_hann_rolloff_1d():
    window = utils.get_hann_rolloff(50, 10)
    assert window.shape == (50,)
    np.testing.assert_equal(window[10:-10], 1)
    np.testing.assert_array_less(window[:10], 1)
    np.testing.assert_array_less(window[-10:], 1)
    np.testing.assert_equal(window[:10], window[-10:][::-1])


def test_get_hann_rolloff_2d():
    window = utils.get_hann_rolloff((20, 20), 5)
    assert window.shape == (20, 20)
    np.testing.assert_equal(window[5:-5, 5:-5], 1)
    
    np.testing.assert_array_less(window[:5, :], 1)
    np.testing.assert_array_less(window[:, :5], 1)
    np.testing.assert_array_less(window[-5:, :], 1)
    np.testing.assert_array_less(window[:, -5:], 1)
    
    np.testing.assert_equal(window[:5, :], window[-5:, :][::-1, :])
    np.testing.assert_equal(window[:, :5], window[:, -5:][:, ::-1])
    
    window = utils.get_hann_rolloff((20, 25), (7, 3))
    assert window.shape == (20, 25)
    np.testing.assert_equal(window[7:-7, 3:-3], 1)
    
    np.testing.assert_array_less(window[:7, :], 1)
    np.testing.assert_array_less(window[:, :3], 1)
    np.testing.assert_array_less(window[-7:, :], 1)
    np.testing.assert_array_less(window[:, -3:], 1)
    
    np.testing.assert_equal(window[:7, :], window[-7:, :][::-1, :])
    np.testing.assert_equal(window[:, :3], window[:, -3:][:, ::-1])


def test_get_hann_rolloff_3d():
    window = utils.get_hann_rolloff((20, 20, 20), 5)
    assert window.shape == (20, 20, 20)
    np.testing.assert_equal(window[5:-5, 5:-5, 5:-5], 1)
    
    np.testing.assert_array_less(window[:5, :, :], 1)
    np.testing.assert_array_less(window[:, :5, :], 1)
    np.testing.assert_array_less(window[:, :, :5], 1)
    np.testing.assert_array_less(window[-5:, :, :], 1)
    np.testing.assert_array_less(window[:, -5:, :], 1)
    np.testing.assert_array_less(window[:, :, -5:], 1)
    
    np.testing.assert_equal(window[:5, :, :], window[-5:, :, :][::-1, :, :])
    np.testing.assert_equal(window[:, :5, :], window[:, -5:, :][:, ::-1, :])
    np.testing.assert_equal(window[:, :, :5], window[:, :, -5:][:, :, ::-1])
    
    window = utils.get_hann_rolloff((20, 25, 15), (7, 3, 4))
    assert window.shape == (20, 25, 15)
    np.testing.assert_equal(window[7:-7, 3:-3, 4:-4], 1)
    
    np.testing.assert_array_less(window[:7, :, :], 1)
    np.testing.assert_array_less(window[:, :3, :], 1)
    np.testing.assert_array_less(window[:, :, :4], 1)
    np.testing.assert_array_less(window[-7:, :, :], 1)
    np.testing.assert_array_less(window[:, -3:, :], 1)
    np.testing.assert_array_less(window[:, :, -4:], 1)
    
    np.testing.assert_equal(window[:7, :, :], window[-7:, :, :][::-1, :, :])
    np.testing.assert_equal(window[:, :3, :], window[:, -3:, :][:, ::-1, :])
    np.testing.assert_equal(window[:, :, :4], window[:, :, -4:][:, :, ::-1])


def test_get_hann_rolloff_some_zeros_rolloffs():
    comparison = utils.get_hann_rolloff(10, 4)
    
    test = utils.get_hann_rolloff((10, 10), (4, 0))
    np.testing.assert_equal(np.stack([comparison] * 10).T, test)
    
    test = utils.get_hann_rolloff((10, 10), (0, 4))
    np.testing.assert_equal(np.stack([comparison] * 10), test)


def test_get_hann_rolloff_zero_borders_1d():
    window = utils.get_hann_rolloff(60, 10, 5)
    comp = utils.get_hann_rolloff(50, 10, 0)
    
    np.testing.assert_equal(window[:5], 0)
    np.testing.assert_equal(window[-5:], 0)
    np.testing.assert_equal(window[5:-5], comp)
    

def test_get_hann_rolloff_zero_borders_2d():
    window = utils.get_hann_rolloff((22, 24), 5, (1, 2))
    comp = utils.get_hann_rolloff((20, 20), 5, 0)
    
    np.testing.assert_equal(window[0:1], 0)
    np.testing.assert_equal(window[-1:], 0)
    np.testing.assert_equal(window[:, 0:2], 0)
    np.testing.assert_equal(window[:, -2:], 0)
    
    np.testing.assert_equal(window[1:-1, 2:-2], comp)
    

def test_get_hann_rolloff_zero_borders_4d():
    window = utils.get_hann_rolloff((22, 12, 14, 16), 4, (5, 0, 1, 2))
    comp = utils.get_hann_rolloff((12, 12, 12, 12), 4, 0)
    
    np.testing.assert_equal(window[0:5], 0)
    np.testing.assert_equal(window[-5:], 0)
    np.testing.assert_equal(window[:, :, :1], 0)
    np.testing.assert_equal(window[:, :, -1:], 0)
    np.testing.assert_equal(window[..., :2], 0)
    np.testing.assert_equal(window[..., -2:], 0)
    
    np.testing.assert_equal(window[5:-5, :, 1:-1, 2:-2], comp)
    

def test_get_hann_rolloff_errors():
    # Wrong number of rolloff sizes
    with pytest.raises(ValueError):
        utils.get_hann_rolloff(50, (20, 30))
    with pytest.raises(ValueError):
        utils.get_hann_rolloff((20, 30), (20, 30, 40))
    with pytest.raises(ValueError):
        utils.get_hann_rolloff((20, 30, 10), (20, 30))
    
    # Wrong number of zero-pad sizes
    with pytest.raises(ValueError):
        utils.get_hann_rolloff(50, 5, (20, 30))
    with pytest.raises(ValueError):
        utils.get_hann_rolloff((20, 30), 5, (20, 30, 40))
    with pytest.raises(ValueError):
        utils.get_hann_rolloff((20, 30, 10), 5, (20, 30))
    
    # Rolloffs that don't even fit in the window
    with pytest.raises(ValueError):
        utils.get_hann_rolloff(10, 10)
    with pytest.raises(ValueError):
        utils.get_hann_rolloff((10, 20), (3, 30))
    with pytest.raises(ValueError):
        utils.get_hann_rolloff((10, 20), (12, 3))
    # With zero-padded edges
    with pytest.raises(ValueError):
        utils.get_hann_rolloff((10, 20), (8, 3), (1, 0))
    with pytest.raises(ValueError):
        utils.get_hann_rolloff((10, 20), (4, 6), (1, 7))
    with pytest.raises(ValueError):
        utils.get_hann_rolloff((10, 20), (4, 8), (1, 7))
    
    # Rolloffs for which the two ends overlap
    with pytest.warns(Warning):
        utils.get_hann_rolloff(10, 9)
    with pytest.warns(Warning):
        utils.get_hann_rolloff((10, 20), (3, 12))
    with pytest.warns(Warning):
        utils.get_hann_rolloff((10, 20), (8, 3))
    # With zero-padded edges
    with pytest.warns(Warning):
        utils.get_hann_rolloff((10, 20), (4, 3), (1, 0))
    with pytest.warns(Warning):
        utils.get_hann_rolloff((10, 20), (4, 3), (1, 7))
    with pytest.warns(Warning):
        utils.get_hann_rolloff((10, 20), (4, 4), (1, 7))
    # No warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        utils.get_hann_rolloff(11, 5)
        utils.get_hann_rolloff(15, 5, 2)
    
    # Check that the following does *not* cause a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(Exception):
            utils.get_hann_rolloff((10, 20), (5, 8))
    
    # Too-small rolloff
    with pytest.raises(ValueError):
        utils.get_hann_rolloff(10, 1)
    
    # Non-integer rolloff
    with pytest.raises(ValueError):
        utils.get_hann_rolloff(10, 2.2)
    
    # All-zeros rolloff
    with pytest.raises(ValueError):
        utils.get_hann_rolloff(25, 0)
    with pytest.raises(ValueError):
        utils.get_hann_rolloff((25, 25), 0)
    with pytest.raises(ValueError):
        utils.get_hann_rolloff((25, 25), (0, 0))
    
    # Negative zero padding
    with pytest.raises(ValueError):
        utils.get_hann_rolloff(15, 3, -1)
    with pytest.raises(ValueError):
        utils.get_hann_rolloff((15, 13), 3, (-1, 4))
    with pytest.raises(ValueError):
        utils.get_hann_rolloff((15, 13), 3, (4, -1))


@pytest.mark.array_compare
def test_sliding_window_stats():
    data1 = np.ones((15, 30))
    data2 = data1 + 9
    data = np.vstack((data1, data2))
    
    mean = utils.sliding_window_stats(data, 5, 'mean')
    return mean


def test_sliding_window_stride_2():
    data1 = np.ones((15, 30))
    data2 = np.full((15, 30), 10)
    data = np.vstack((data1, data2))
    
    std, mean = utils.sliding_window_stats(data, 5, ['std', 'mean'])
    std2, mean2 = utils.sliding_window_stats(data, 5, ['std', 'mean'],
            sliding_window_stride=2)
    
    assert std2.shape == std.shape
    assert mean2.shape == mean.shape
    
    # Check the padded columns
    for arr in std, mean, std2, mean2:
        np.testing.assert_array_equal(arr[0], arr[3])
        np.testing.assert_array_equal(arr[1], arr[3])
        np.testing.assert_array_equal(arr[-2], arr[-3])
        np.testing.assert_array_equal(arr[-1], arr[-3])
        np.testing.assert_array_equal(arr[:, 0], arr[:, 3])
        np.testing.assert_array_equal(arr[:, 1], arr[:, 3])
        np.testing.assert_array_equal(arr[:, -2], arr[:, -3])
        np.testing.assert_array_equal(arr[:, -1], arr[:, -3])
    
    # Trim off the padded columns
    std = std[2:-2, 2:-2]
    std2 = std2[2:-2, 2:-2]
    mean = mean[2:-2, 2:-2]
    mean2 = mean2[2:-2, 2:-2]
    
    for arr, arr2 in zip((std, mean), (std2, mean2)):
        np.testing.assert_array_equal(arr[::2, ::2], arr2[::2, ::2])
        np.testing.assert_array_equal(arr2[::2], arr2[1::2])
        np.testing.assert_array_equal(arr2[:, ::2], arr2[:, 1::2])


def test_sliding_window_stride_3():
    data1 = np.ones((15, 31))
    data2 = np.full((15, 31), 10)
    data = np.vstack((data1, data2))
    
    std, mean = utils.sliding_window_stats(data, 5, ['std', 'mean'])
    std2, mean2 = utils.sliding_window_stats(data, 5, ['std', 'mean'],
            sliding_window_stride=3)
    
    assert std2.shape == std.shape
    assert mean2.shape == mean.shape
    
    # Check the padded columns
    for arr in std, mean, std2, mean2:
        np.testing.assert_array_equal(arr[0], arr[3])
        np.testing.assert_array_equal(arr[1], arr[3])
        np.testing.assert_array_equal(arr[-2], arr[-3])
        np.testing.assert_array_equal(arr[-1], arr[-3])
        np.testing.assert_array_equal(arr[:, 0], arr[:, 3])
        np.testing.assert_array_equal(arr[:, 1], arr[:, 3])
        np.testing.assert_array_equal(arr[:, -2], arr[:, -3])
        np.testing.assert_array_equal(arr[:, -1], arr[:, -3])
    
    # Trim off the padded columns
    std = std[2:-2, 2:-2]
    std2 = std2[2:-2, 2:-2]
    mean = mean[2:-2, 2:-2]
    mean2 = mean2[2:-2, 2:-2]
    
    for arr, arr2 in zip((std, mean), (std2, mean2)):
        np.testing.assert_array_equal(arr[::3, ::3], arr2[::3, ::3])
        np.testing.assert_array_equal(arr2[::3], arr2[1::3])
        np.testing.assert_array_equal(arr2[1:-1:3], arr2[2::3])
        np.testing.assert_array_equal(arr2[:, ::3], arr2[:, 1::3])
        np.testing.assert_array_equal(arr2[:, ::3], arr2[:, 2::3])


def test_sliding_window_nans():
    data = np.ones((15, 15))
    data[2, 2] = np.nan
    
    mean_with_nan = utils.sliding_window_stats(data, 5, 'mean',
            check_nans=False)
    mean_wo_nan = utils.sliding_window_stats(data, 5, 'mean',
            check_nans=True)
    
    np.testing.assert_array_equal(mean_wo_nan, 1)
    np.testing.assert_allclose(mean_with_nan[:5, :5], np.nan, equal_nan=True)
    np.testing.assert_array_equal(mean_with_nan[5:, 5:], 1)


@pytest.mark.array_compare
def test_sliding_window_edges():
    np.random.seed(20)
    data = np.random.random(900).reshape((30, 30))
    med = utils.sliding_window_stats(data, 6, ['median'])
    return med[0]


@pytest.mark.array_compare
def test_sliding_window_trim():
    np.random.seed(20)
    data = np.random.random(900).reshape((30, 30))
    med = utils.sliding_window_stats(data, 6, ['median'], trim=(7, 4, 9, 5))
    return med[0]


@pytest.mark.array_compare
def test_sliding_window_1D():
    np.random.seed(40)
    data = np.random.random(40)
    
    mean = utils.sliding_window_stats(data, 5, 'mean')
    mean_strided = utils.sliding_window_stats(data, 5, 'mean',
            sliding_window_stride=2)
    mean_trimmed = utils.sliding_window_stats(data, 5, 'mean', trim=(3, 4))
    
    return np.vstack((mean, mean_strided, mean_trimmed))


def test_sliding_window_where():
    data = np.ones((15, 15))
    data[::2, ::2] = 2
    
    mean = utils.sliding_window_stats(data, 5, ['mean'], where=(data == 1))[0]
    
    np.testing.assert_array_equal(mean, 1)


def test_sliding_window_asymmetric_window():
    data = np.ones((15, 15))
    data[::2] = 2
    
    mean = utils.sliding_window_stats(data, (1, 5), 'mean')
    
    assert mean.shape == data.shape
    
    np.testing.assert_array_equal(mean[::2], 2)
    np.testing.assert_array_equal(mean[1::2], 1)

