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


@pytest.mark.parametrize('stat', ['mean', 'std', 'median'])
@pytest.mark.parametrize('fill', ['repeat', 'interp'])
def test_sliding_window_stride_2(stat, fill):
    np.random.seed(4321)
    data = np.random.random((30, 30))
    
    out = utils.sliding_window_stats(data, 5, stat)
    out2 = utils.sliding_window_stats(data, 5, stat, sliding_window_stride=2,
            stride_fill=fill)
    
    assert out.shape == out2.shape
    
    # Check the padded columns
    for arr in out, out2:
        np.testing.assert_array_equal(arr[0], arr[2])
        np.testing.assert_array_equal(arr[1], arr[2])
        np.testing.assert_array_equal(arr[-2], arr[-3])
        np.testing.assert_array_equal(arr[-1], arr[-3])
        np.testing.assert_array_equal(arr[:, 0], arr[:, 2])
        np.testing.assert_array_equal(arr[:, 1], arr[:, 2])
        np.testing.assert_array_equal(arr[:, -2], arr[:, -3])
        np.testing.assert_array_equal(arr[:, -1], arr[:, -3])
    
    # Trim off the padded columns
    out = out[2:-2, 2:-2]
    out2 = out2[2:-2, 2:-2]
    
    # Check that the computed pixels are the same
    np.testing.assert_allclose(out[::2, ::2], out2[::2, ::2])
    
    if fill == 'interp':
        # Check that the interpolated pixels are interpolated (making sure that
        # for each dimension, we check only those pixels interpolated along
        # that dimension)
        np.testing.assert_allclose(
                (out[:-2:2, ::2] + out[2::2, ::2]) / 2,
                out2[1:-1:2, ::2])
        np.testing.assert_allclose(
                (out[::2, :-2:2] + out[::2, 2::2]) / 2, out2[::2, 1:-1:2])
        
        # Check that the final, duplicated rows are right
        np.testing.assert_array_equal(out2[-1], out2[-2])
        np.testing.assert_array_equal(out2[:, -1], out2[:, -2])
    else:
        # Check that the repeat data is right
        np.testing.assert_array_equal(out2[::2], out2[1::2])
        np.testing.assert_array_equal(out2[:, ::2], out2[:, 1::2])


@pytest.mark.parametrize('stat', ['mean', 'std', 'median'])
@pytest.mark.parametrize('fill', ['repeat', 'interp'])
def test_sliding_window_stride_3(stat, fill):
    np.random.seed(4321)
    data = np.random.random((30, 31))
    
    out = utils.sliding_window_stats(data, 5, stat)
    out2 = utils.sliding_window_stats(data, 5, stat,
            sliding_window_stride=3, stride_fill=fill)
    
    assert out.shape == out2.shape
    
    # Check the padded columns
    for arr in out, out2:
        np.testing.assert_array_equal(arr[0], arr[2])
        np.testing.assert_array_equal(arr[1], arr[2])
        np.testing.assert_array_equal(arr[-2], arr[-3])
        np.testing.assert_array_equal(arr[-1], arr[-3])
        np.testing.assert_array_equal(arr[:, 0], arr[:, 2])
        np.testing.assert_array_equal(arr[:, 1], arr[:, 2])
        np.testing.assert_array_equal(arr[:, -2], arr[:, -3])
        np.testing.assert_array_equal(arr[:, -1], arr[:, -3])
    
    # Trim off the padded columns
    out = out[2:-2, 2:-2]
    out2 = out2[2:-2, 2:-2]
    
    # Check that the computed pixels are the same
    np.testing.assert_allclose(out[1::3, 1::3], out2[1::3, 1::3])
    
    if fill == 'interp':
        # Check that the initial/final duplicated rows are right, and then
        # remove them
        np.testing.assert_array_equal(out2[0], out2[1])
        np.testing.assert_array_equal(out2[:, -1], out2[:, -2])
        np.testing.assert_array_equal(out2[:, 0], out2[:, 1])
        out = out[1:, 1:-1]
        out2 = out2[1:, 1:-1]
        
        # Check that the interpolated pixels are interpolated (making sure that
        # for each dimension, we check only those pixels interpolated along
        # that dimension)
        np.testing.assert_allclose(
                (2/3*out[:-3:3, ::3] + 1/3*out[3::3, ::3]),
                out2[1:-1:3, ::3])
        np.testing.assert_allclose(
                (1/3*out[:-3:3, ::3] + 2/3*out[3::3, ::3]),
                out2[2:-1:3, ::3])
        np.testing.assert_allclose(
                (2/3*out[::3, :-3:3] + 1/3*out[::3, 3::3]),
                out2[::3, 1:-1:3])
        np.testing.assert_allclose(
                (1/3*out[::3, :-3:3] + 2/3*out[::3, 3::3]),
                out2[::3, 2:-1:3])
    else:
        # Check that the repeated values are right
        np.testing.assert_array_equal(out2[::3], out2[1::3])
        np.testing.assert_array_equal(out2[1:-1:3], out2[2::3])
        np.testing.assert_array_equal(out2[:, ::3], out2[:, 1::3])
        np.testing.assert_array_equal(out2[:, ::3], out2[:, 2::3])


@pytest.mark.parametrize('stat', ['mean', 'std', 'median'])
def test_sliding_window_stride_5(stat):
    """
    Regression test---sometimes the duplicated values were offset from where
    they should be, depending on the stride size and data array size. Wasn't
    ever visible with a stride of <= 3
    """
    np.random.seed(4321)
    data = np.random.random((30, 31))
    
    out = utils.sliding_window_stats(data, 5, stat)
    out2 = utils.sliding_window_stats(data, 5, stat,
            sliding_window_stride=5, stride_fill='repeat')
    
    assert out.shape == out2.shape
    
    # Check the padded columns
    for arr in out, out2:
        for i in range(2):
            np.testing.assert_array_equal(arr[i], arr[2])
            np.testing.assert_array_equal(arr[-i-1], arr[-3])
            np.testing.assert_array_equal(arr[:, i], arr[:, 2])
            np.testing.assert_array_equal(arr[:, -i-1], arr[:, -3])
    
    # Trim off the padded columns
    out = out[2:-2, 2:-2]
    out2 = out2[2:-2, 2:-2]
    
    # Check that the computed pixels are the same
    np.testing.assert_allclose(out[2::5, 2::5], out2[2::5, 2::5])
    
    # There should be a few extra repeated rows/columns at the end
    np.testing.assert_array_equal(out2[-1], out2[-2])
    out = out[:-1]
    out2 = out2[:-1]
    np.testing.assert_array_equal(out2[:, -2], out2[:, -3])
    np.testing.assert_array_equal(out2[:, -1], out2[:, -3])
    out = out[:, :-2]
    out2 = out2[:, :-2]
    
    # Check that the repeated values are right
    for i in [1, 2]:
        np.testing.assert_array_equal(out2[2 - i::5], out2[2::5])
        np.testing.assert_array_equal(out2[2 + i::5], out2[2::5])
        np.testing.assert_array_equal(out2[:, 2 - i::5], out2[:, 2::5])
        np.testing.assert_array_equal(out2[:, 2 + i::5], out2[:, 2::5])


@pytest.mark.parametrize('stat', ['mean', 'std', 'median'])
@pytest.mark.parametrize('fill', ['repeat', 'interp'])
def test_sliding_window_stride_partial(stat, fill):
    """ Only stride in one dimension """
    np.random.seed(4321)
    data = np.random.random((30, 31))
    
    out = utils.sliding_window_stats(data, 5, stat)
    out2 = utils.sliding_window_stats(data, 5, stat,
            sliding_window_stride=(2, 0), stride_fill=fill)
    
    assert out.shape == out2.shape
    
    # Check the padded columns
    for arr in out, out2:
        np.testing.assert_array_equal(arr[0], arr[2])
        np.testing.assert_array_equal(arr[1], arr[2])
        np.testing.assert_array_equal(arr[-2], arr[-3])
        np.testing.assert_array_equal(arr[-1], arr[-3])
        np.testing.assert_array_equal(arr[:, 0], arr[:, 2])
        np.testing.assert_array_equal(arr[:, 1], arr[:, 2])
        np.testing.assert_array_equal(arr[:, -2], arr[:, -3])
        np.testing.assert_array_equal(arr[:, -1], arr[:, -3])
    
    # Trim off the padded columns
    out = out[2:-2, 2:-2]
    out2 = out2[2:-2, 2:-2]
    
    # Check that the computed pixels are the same
    np.testing.assert_allclose(out[::2], out2[::2])
    
    if fill == 'interp':
        # Check that the initial/final duplicated rows are right, and then
        # remove them
        np.testing.assert_array_equal(out2[-1], out2[-2])
        out = out[:-1]
        out2 = out2[:-1]
        
        # Check that the interpolated pixels are interpolated (making sure that
        # for each dimension, we check only those pixels interpolated along
        # that dimension)
        np.testing.assert_allclose(
                (out[:-2:2] + out[2::2]) / 2,
                out2[1:-1:2])
    else:
        # Check that the repeat data is right
        np.testing.assert_array_equal(out2[::2], out2[1::2])


@pytest.mark.parametrize('stat', ['mean', 'std', 'median'])
@pytest.mark.parametrize('fill', ['repeat', 'interp'])
def test_sliding_window_stride_varied(stat, fill):
    """ Different strides in each dimension """
    np.random.seed(4321)
    data = np.random.random((30, 31))
    
    out = utils.sliding_window_stats(data, 5, stat)
    out2 = utils.sliding_window_stats(data, 5, stat,
            sliding_window_stride=(2, 3), stride_fill=fill)
    
    assert out.shape == out2.shape
    
    # Check the padded columns
    for arr in out, out2:
        np.testing.assert_array_equal(arr[0], arr[2])
        np.testing.assert_array_equal(arr[1], arr[2])
        np.testing.assert_array_equal(arr[-2], arr[-3])
        np.testing.assert_array_equal(arr[-1], arr[-3])
        np.testing.assert_array_equal(arr[:, 0], arr[:, 2])
        np.testing.assert_array_equal(arr[:, 1], arr[:, 2])
        np.testing.assert_array_equal(arr[:, -2], arr[:, -3])
        np.testing.assert_array_equal(arr[:, -1], arr[:, -3])
    
    # Trim off the padded columns
    out = out[2:-2, 2:-2]
    out2 = out2[2:-2, 2:-2]
    
    # Check that the computed pixels are the same
    np.testing.assert_allclose(out[::2, 1::3], out2[::2, 1::3])
    
    if fill == 'interp':
        # Check that the initial/final duplicated rows are right, and then
        # remove them
        np.testing.assert_array_equal(out2[-1], out2[-2])
        np.testing.assert_array_equal(out2[:, -1], out2[:, -2])
        np.testing.assert_array_equal(out2[:, 0], out2[:, 1])
        out = out[:-1, 1:-1]
        out2 = out2[:-1, 1:-1]
        
        # Check that the interpolated pixels are interpolated (making sure that
        # for each dimension, we check only those pixels interpolated along
        # that dimension)
        np.testing.assert_allclose(
                (out[:-2:2, ::3] + out[2::2, ::3]) / 2,
                out2[1:-1:2, ::3])
        
        np.testing.assert_allclose(
                (2/3*out[::2, :-3:3] + 1/3*out[::2, 3::3]),
                out2[::2, 1:-1:3])
        np.testing.assert_allclose(
                (1/3*out[::2, :-3:3] + 2/3*out[::2, 3::3]),
                out2[::2, 2:-1:3])
    else:
        # Check that the repeat data is right
        np.testing.assert_array_equal(out2[::2], out2[1::2])
        np.testing.assert_array_equal(out2[:, ::3], out2[:, 1::3])
        np.testing.assert_array_equal(out2[:, ::3], out2[:, 2::3])


@pytest.mark.parametrize('stat', ['mean', 'std', 'median'])
def test_sliding_window_nans(stat):
    data = np.ones((15, 15))
    data[2, 2] = np.nan
    
    with_nan = utils.sliding_window_stats(data, 5, stat, check_nans=False)
    wo_nan = utils.sliding_window_stats(data, 5, stat, check_nans=True)
    
    expected = 0 if stat == 'std' else 1
    np.testing.assert_array_equal(wo_nan, expected)
    if stat == 'median':
        np.testing.assert_allclose(with_nan[:5, :5], np.nan, equal_nan=True)
        np.testing.assert_array_equal(with_nan[5:], expected)
        np.testing.assert_array_equal(with_nan[:, 5:], expected)
    else:
        np.testing.assert_allclose(with_nan, np.nan, equal_nan=True)


@pytest.mark.array_compare
def test_sliding_window_trim():
    np.random.seed(20)
    data = np.random.random(900).reshape((30, 30))
    med = utils.sliding_window_stats(data, 6, ['median'], trim=(7, 4, 9, 5))
    return med[0]


@pytest.mark.parametrize('stat', ['mean', 'std', 'median'])
@pytest.mark.array_compare
def test_sliding_window_1D(stat):
    np.random.seed(40)
    data = np.random.random(40)
    
    out = utils.sliding_window_stats(data, 5, stat)
    out_strided = utils.sliding_window_stats(data, 5, stat,
            sliding_window_stride=2, stride_fill='interp')
    out_trimmed = utils.sliding_window_stats(data, 5, stat, trim=(3, 4))
    
    return np.vstack((out, out_strided, out_trimmed))


@pytest.mark.parametrize('stat', ['mean', 'std', 'median'])
def test_sliding_window_4D(stat):
    data = np.full((6, 5, 7, 6), 10)
    
    out = utils.sliding_window_stats(data, 2, stat)
    
    expected = 0 if stat == 'std' else 10
    
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize('stat', ['mean', 'std'])
@pytest.mark.parametrize('stride', [1,2,3])
def test_sliding_window_where(stat, stride):
    data = np.ones((30, 30))
    
    data[15:17] = 2
    data[:, 15:17] = 2
    
    where = data == 1
    
    out = utils.sliding_window_stats(data, 5, stat, where=where,
            sliding_window_stride=stride)
    
    expected = 0 if stat == 'std' else 1
    
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize('stat', ['mean', 'median', 'std'])
def test_sliding_window_asymmetric_window(stat):
    data = np.ones((15, 15))
    data[::2] = 2
    
    out = utils.sliding_window_stats(data, (1, 5), stat)
    
    assert out.shape == data.shape
    
    if stat == 'std':
        expected1 = 0
        expected2 = 0
    else:
        expected1 = 1
        expected2 = 2
    np.testing.assert_array_equal(out[::2], expected2)
    np.testing.assert_array_equal(out[1::2], expected1)


@pytest.mark.parametrize('stat', ['mean', 'std'])
@pytest.mark.parametrize('stride', [1,2,3])
def test_sliding_window_where(stat, stride):
    data = np.ones((30, 30))
    
    data[15:17] = 2
    data[:, 15:17] = 2
    
    where = data == 1
    
    out = utils.sliding_window_stats(data, 5, stat, where=where,
            sliding_window_stride=stride)
    
    expected = 0 if stat == 'std' else 1
    
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize('stat', ['mean', 'std', 'median'])
def test_sliding_window_empty_strips(stat):
    np.random.seed(2)
    data = np.random.random((30, 30))
    
    # Create strips of NaNs so that the rolling calculations experience steps
    # where no pixels are added or none are removed. Ensure that doesn't make
    # the calculations wacky.
    data[15:17] = np.nan
    data[:, 15:17] = np.nan
    # To stress it further, don't provide any data for the initial window
    # position, and ensure the calculations still go on.
    data[:6, :6] = np.nan
    
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore',
                message=".*All-NaN slice.*")
        out = utils.sliding_window_stats(data, 5, stat, check_nans=True)
    
    if stat == 'mean':
        fcn = np.nanmean
    elif stat == 'std':
        fcn = np.nanstd
    elif stat == 'median':
        fcn = np.nanmedian
    
    for i in range(2, data.shape[0]-2):
        for j in range(2, data.shape[1]-2):
            if i <= 3 and j <= 3:
                assert np.isnan(out[i, j])
            else:
                assert out[i, j] == pytest.approx(fcn(data[i-2:i+3, j-2:j+3]))


@pytest.mark.parametrize('stat', ['mean', 'std', 'median'])
@pytest.mark.parametrize('stride', [1, 2, 3, 4, 5])
def test_sliding_window_validity(stat, stride):
    """Manually check that each value we get out is correct"""
    np.random.seed(2)
    data = np.random.random((30, 31))
    
    out = utils.sliding_window_stats(data, 5, stat, check_nans=False,
            sliding_window_stride=stride, stride_fill='interp')
    
    if stat == 'mean':
        fcn = np.mean
    elif stat == 'std':
        fcn = np.std
    elif stat == 'median':
        fcn = np.median
    
    stride_offset = int((stride - 1) // 2)
    for i in range(2 + stride_offset, data.shape[0]-2, stride):
        for j in range(2 + stride_offset, data.shape[1]-2, stride):
            assert out[i, j] == pytest.approx(fcn(data[i-2:i+3, j-2:j+3]))
