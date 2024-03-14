[![tests](https://github.com/svank/wispr_analysis/actions/workflows/main.yml/badge.svg)](https://github.com/svank/wispr_analysis/actions/workflows/main.yml)
[![docs](https://github.com/svank/wispr_analysis/actions/workflows/docs.yml/badge.svg)](https://github.com/svank/wispr_analysis/actions/workflows/docs.yml)

# wispr-analysis
Shared tools for WISPR data analysis

Find the documentation [here](https://wispr-analysis.readthedocs.io/en/latest/).

## Commands

To run the tests, simply run `pytest` within the base directory of this repo.

To generate a coverage report for the tests, run `NUMBA_DISABLE_JIT=1 coverage run; coverage html`.

To update reference arrays for the tests, run `pytest --arraydiff-generate-path=tests/reference`.

To update reference images for the tests, run `pytest --mpl-generate-path=tests/baseline`.

If the image-comparison tests are failing, run `pytest --mpl-generate-summary=html` to generate a summary page showing the generated and reference images. The location of the generated file will be shown at the end of pytest’s command-line output.

## Some highlights

`plot_utils.py`

- `plot_WISPR`: Aims to be a versatile function that does the Right Thing for plotting WISPR images, with colorbar bounds that are adjusted for inner and outer FOV and for L2 or L3 images, a square-root-scaled colorbar, and WCS coordinate support
- `*_axis_dates`: Helper util for labeling a temporal axis with dates.
- `plot_orbit`: Reads a directory (or nested set of directories) of WISPR files and plots a diagram showing the orbital path of PSP and the locations where images were taken, like this:
![image](https://user-images.githubusercontent.com/23462789/174193783-26a7360b-fa92-4a36-8306-b0564eae6aa5.png)


`projections.py`

- `reproject_to_radial`: Proof-of-concept code for reprojecting data into a radial coordinate system (where each row of the output array is a radial line out from the Sun.

`data_cleaning.py`

- `dust_streak_filter`: Code for identifying debris streaks in the WISPR images
- `clean_fits_files`: Function to batch-run `dust_streak_filter` on a directory of images.

`composites.py`

- `gen_composite`: Reprojects an inner- and outer-FOV image into a common coordinate system

`utils.py`

- `to_timestamp`: Parse a timestamp from a handful of formats, including the timestamps inside WISPR headers, or entire WISPR filenames. Returns a numerical timestamp.
- `collect_files`: Walks a directory of WISPR files (or a directory of subdirectories of WISPR images), identifies all the WISPR images, sorts them, and separates them by inner and outer FOVs.
- `ignore_fits_warnings`: Suppresses the warnings Astropy raises when reading WISPR FITS files or parsing WCS data. Use like so:
```python
with utils.ignore_fits_warnings():
    header = fits.getheader(wispr_file_name)
    wcs = WCS(header)
```
