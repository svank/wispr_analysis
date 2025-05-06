from __future__ import annotations

from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import datetime
from itertools import chain, repeat
import multiprocessing
import os
import pickle
import shutil
import warnings

from astropy.io import fits
import astropy.units as u
from astropy.wcs import DistortionLookupTable, WCS
from IPython.core.display import HTML
from IPython.display import display
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.stats
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from . import star_tools, utils, plot_utils


def run(raw_dir, between=None, use_outer=False, corrector=None,
        smooth_second_offsets=False, enable_shear_adjustment=False, max_dsun=.25 * u.au.to(u.m)):
    tasks = TaskSequence()
    print("Running on " + datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

    i = raw_dir.index('_ENC')
    enc_number = int(raw_dir[i + 4:i + 6])
    if '_L2_' in raw_dir:
        L_number = 2
    elif '_L3_' in raw_dir:
        L_number = 3
    else:
        raise ValueError("Could not detect data level")

    title_string = f"Encounter {enc_number}, L{L_number}"
    if use_outer:
        title_string += ', Outer'

    ifiles, ofiles = utils.collect_files(raw_dir, include_sortkey=False, between=between,
                                         filters=('dsun_obs', 0, max_dsun))

    if use_outer:
        ifiles = ofiles

    inputs = [InputFile(f) for f in ifiles]

    star_finder = StarFinder(inputs, corrector=corrector)
    star_finder.execute()
    tasks.append(star_finder)

    found_stars = star_finder.found_stars

    offsetter = PointingOffset(tasks[-1].outputs, found_stars)
    tasks.append(offsetter)
    offsetter.execute()

    tweaker = ProjectionTweak(tasks[-1].outputs, found_stars)
    tweaker.execute()
    tasks.append(tweaker)

    print("\n\n\nStarting second iteration...\n\n\n")


    star_finder = StarFinder(tasks[-1].outputs, start_at_max=False, corrector=corrector)
    star_finder.execute()
    tasks.append(star_finder)

    found_stars = star_finder.found_stars

    offsetter = PointingOffset(tasks[-1].outputs, found_stars, smooth_offsets=smooth_second_offsets)
    offsetter.execute()
    tasks.append(offsetter)

    tweaker = ProjectionTweak(tasks[-1].outputs, found_stars)
    tweaker.execute()
    tasks.append(tweaker)

    mapper = DistortionMapper(tasks[-1].outputs, found_stars)
    mapper.execute()
    tasks.append(mapper)

    print("Computing final errors...")

    with multiprocessing.Pool() as p:
        errors_after_tasks = p.starmap(series_errors,
                                       zip([tasks[0].found_stars] + [t.found_stars for t in tasks],
                                           [tasks[0].inputs] + [t.outputs for t in tasks],
                                           [False] * (len(tasks) + 1)))
    for task, errors_before, errors_after in zip(tasks, errors_after_tasks[:-1], errors_after_tasks[1:]):
        task.errors_before = errors_before
        task.errors_after = errors_after

    return tasks


class RunArtefact:
    def __init__(self):
        pass


class TaskSequence(list):
    def plot_error_histogram(self):
        errors = [self[0].errors_before.errors.ravel()]
        labels = ["Initial"]
        for task in self:
            if isinstance(task, StarFinder):
                continue
            errors.append(task.errors_after.errors.ravel())
            labels.append(f"After {task._plot_label}")
        plt.hist(errors, label=labels,
                 bins=50, range=(0, 1.1), histtype='step')
        plt.xlabel("Position error (px)")
        plt.ylabel("Count")
        plt.legend()
        plt.show()

    def plot_before(self):
        self[0].errors_before.plot_errors()

    def plot_after(self):
        self[-1].errors_after.plot_errors()


class Task:
    pass


class StarFinder(Task):
    _plot_label = "Found stars"

    def __init__(self, inputs, start_at_max=True, corrector=None):
        self.inputs = inputs
        self.start_at_max = start_at_max
        self.corrector = corrector

    def execute(self):
        print("Identifying stars...")
        (self.found_stars, self.good_x, self.good_y, self.crowded_out_x, self.crowded_out_y,
         self.bad_x, self.bad_y, self.codes) = find_all_stars(
            self.inputs, ret_all=True, start_at_max=self.start_at_max, corrector=self.corrector)
        self.outputs = [i for i in self.inputs if i.timestamp in self.found_stars.mapping_by_time]

    def plot_results(self, title_string=""):
        if self.good_x is not None:
            plt.figure(figsize=(14, 4))
            plt.subplot(131)
            plot1 = plt.hexbin(self.good_x, self.good_y, gridsize=30)
            plt.title("Good fits")
            plt.colorbar()
            plt.subplot(132)
            plot2 = plt.hexbin(self.bad_x, self.bad_y, gridsize=30)
            plt.title("Bad fits")
            plt.colorbar()
            plt.subplot(133)
            plot3 = plt.hexbin(self.crowded_out_x, self.crowded_out_y, gridsize=30)
            plt.title("Crowded stars")
            plt.colorbar()
            plt.suptitle(title_string)

            vmax = max(plot1.get_array().max(), plot2.get_array().max(), plot3.get_array().max())
            plot1.norm.vmax = vmax
            plot2.norm.vmax = vmax
            plot3.norm.vmax = vmax

            plt.show()

        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        ra, dec = zip(*self.found_stars.mapping_by_coord.keys())
        plt.scatter(ra, dec, s=1)
        plt.xlabel("RA")
        plt.ylabel("Declination")
        plt.title("Identified stars")

        px_x, px_y = [], []
        for stars in self.found_stars.mapping_by_coord.values():
            for star in stars:
                px_x.append(star.x)
                px_y.append(star.y)
        plt.subplot(132)
        plt.scatter(px_x, px_y, s=.01)
        plt.xlabel("Image frame x")
        plt.ylabel("Image frame y")
        plt.title("Identified stars")

        plt.subplot(133)
        plt.plot([len(self.found_stars[input.timestamp]) for input in self.inputs])
        plt.xlabel("Frame number")
        plt.ylabel("Number of identified stars")
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f"<StarFinder, {len(self.inputs)} inputs>"

    def __str__(self):
        return self.__repr__()


@dataclass
class PointingOffset(Task):
    _plot_label = "Adjusted pointing"

    inputs: list[InputFile]
    found_stars: FoundStars
    smooth_offsets: bool = False
    outputs: list[InputFile] = None
    angle_ts: np.ndarray = None
    rval_ts: np.ndarray = None
    rpix_ts: np.ndarray = None
    t_vals: np.ndarray = None
    alignment_rmse: np.ndarray = None

    def execute(self):
        print("Finding offsets...")
        (self.outputs, self.angle_ts, self.rval_ts, self.rpix_ts, self.t_vals,
         self.alignment_rmse) = iteratively_align_files(
            self.inputs, self.found_stars, smooth_offsets=self.smooth_offsets)
        self.t_vals = np.asarray(self.t_vals)
        self.angle_ts = np.asarray(self.angle_ts)

        return self.outputs

    def plot_results(self, title_string=""):
        plt.figure(figsize=(9, 3))
        plt.subplot(121)
        plot_dates = plot_utils.x_axis_dates(self.t_vals)
        plt.plot(plot_dates, self.rval_ts[:, 0], label='RA offset')
        plt.plot(plot_dates, self.rval_ts[:, 1], label='Dec offset')
        plt.plot(plot_dates, self.angle_ts * 180 / np.pi, color='C2', label='Rotation offset')
        plt.axhline(0, color='.5')
        plt.xlabel("Time")
        plt.ylabel("Offset (deg)")
        plt.legend()

        plt.subplot(122)
        plot_dates = plot_utils.x_axis_dates(self.t_vals)
        plt.plot(plot_dates, self.alignment_rmse)
        plt.xlabel("Time")
        plt.ylabel("RMSE error after offset")

        plt.suptitle(title_string)
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f"<PointingOffset, {len(self.inputs)} inputs>"

    def __str__(self):
        return self.__repr__()


@dataclass
class ProjectionTweak(Task):
    _plot_label = "Projection tweaked"

    inputs: list[InputFile]
    found_stars: FoundStars
    enable_shear_adjustment: bool = False
    outputs: list[InputFile] = None
    ns: np.ndarray = None
    pv_sequence_orig: np.ndarray = None
    bin_centers_orig: np.ndarray = None
    bin_centers: np.ndarray = None
    pv_sequence: np.ndarray = None
    pv_orig: list[float] = None
    lon_bin_spacing: int = 5
    lon_bin_width: float = 20

    def execute(self):
        print("Tweaking projection")

        lons = np.array([f.longitude for f in self.inputs])
        lons[lons < -100] += 360

        dl = self.lon_bin_width
        starts = np.arange(np.min(lons), np.max(lons) - dl + 1, self.lon_bin_spacing)
        bin_centers = starts + dl / 2

        def args():
            for start, end in zip(starts, starts + dl):
                inputs = [i for i, l in zip(self.inputs, lons) if start <= l <= end]
                these_lons = [i.longitude for i in inputs]
                times = [i.timestamp for i in inputs]
                found_stars = self.found_stars.filter_for_time(
                    min(times), max(times))
                yield inputs, these_lons, found_stars

        pvs = []
        ns = []
        for ret, n in process_map(tweak_projection_one_l_band,
                                  args(),
                                  total=len(starts)):
            pvs.append(ret)
            ns.append(n)
        pv_sequence = np.array(pvs)
        self.ns = np.array(ns)

        self.bin_centers_orig = bin_centers.copy()
        self.pv_sequence_orig = pv_sequence.copy()
        f = np.isfinite(pv_sequence[:, 3])
        pv_sequence = pv_sequence[f, :]
        bin_centers = bin_centers[f]
        self.bin_centers = bin_centers

        for i in range(pv_sequence.shape[1]):
            y = pv_sequence[:, i]
            y = utils.time_window_savgol_filter(bin_centers, y, 25 * 5, 4)
            pv_sequence[:, i] = y

        self.outputs = []
        for input, lon in zip(self.inputs, lons):
            pv_values = []
            for i in range(pv_sequence.shape[1]):
                pv_values.append(np.interp(lon, bin_centers, pv_sequence[:, i]))
            refinement = Refinement(pv_values=pv_values)
            output = input.gen_successor()
            output.refinement_applied = refinement
            output.wcs = refinement.apply_to(input.wcs)
            self.outputs.append(output)

        self.pv_orig = Refinement().get_resulting_pv_values(self.inputs[0].wcs, only_ones_we_modify=True)
        self.pv_sequence = pv_sequence
        return self.outputs

    def plot_results(self, title_string=""):
        fig, axs = plt.subplots(1, 7, figsize=(15, 3))
        ax = axs[0]
        ax.plot(self.bin_centers_orig, self.ns)
        ax.set_xlabel("Longitude")
        ax.set_title("# of files")

        for i in range(0, self.pv_sequence.shape[1]):
            ax = axs[i + 1]
            ax.plot(self.bin_centers_orig, self.pv_sequence_orig[:, i])
            ax.plot(self.bin_centers, self.pv_sequence[:, i], ls='--')
            ax.axhline(self.pv_orig[i][2], color='k', linestyle='--')
            ax.set_xlabel("Longitude")
            ax.set_title(f"PV2_{i}")
        plt.suptitle(title_string)
        plt.tight_layout()

    def __repr__(self):
        return f"<ProjectionTweak, {len(self.inputs)} inputs>"

    def __str__(self):
        return self.__repr__()

def tweak_projection_one_l_band(args):
    inputs, lons, found_stars = args
    dlon = np.abs(np.diff(lons))
    dlon_left = np.concatenate(([np.inf], dlon))
    dlon_right = np.concatenate((dlon, [np.inf]))
    dlon = np.min(np.stack((dlon_left, dlon_right)), axis=0)

    if len(inputs) >= 15:
        refinement = iteratively_perturb_projections(inputs, found_stars, False, do_print=False, weights=dlon)
        pvs = refinement.get_resulting_pv_values(inputs[0].wcs, only_ones_we_modify=True)
        pvs = [pv[2] for pv in pvs]
        return pvs, len(inputs)
    else:
        return [np.nan] * 6, len(inputs)


@dataclass
class DistortionMapper(Task):
    _plot_label = "Distortion table added"

    inputs: list[InputFile]
    found_stars: FoundStars
    outputs: list[InputFile] = None
    refinement: Refinement = None
    lon_bin_spacing: int = 5
    lon_bin_width: float = 20
    px_per_bin: int = 16

    def execute(self):
        print("Computing distortion maps")

        lons = np.array([f.longitude for f in self.inputs])
        lons[lons < -100] += 360

        dl = self.lon_bin_width
        starts = np.arange(np.min(lons), np.max(lons) - dl + 1, self.lon_bin_spacing)
        bin_centers = starts + dl / 2

        def args():
            for start, end in zip(starts, starts + dl):
                inputs = [i for i, l in zip(self.inputs, lons) if start <= l <= end]
                times = [i.timestamp for i in inputs]
                found_stars = self.found_stars.filter_for_time(
                    min(times), max(times))
                yield found_stars, inputs, self.px_per_bin

        with multiprocessing.Pool() as p:
            res = p.starmap(_compute_distortion_map_one_l_band,  args())

        refinements, self.xmaps, self.ymaps = zip(*res)

        self.band_starts = starts
        self.band_ends = starts + dl
        self.band_centers = bin_centers

        self.outputs = []
        for input, lon in zip(self.inputs, lons):
            i = 0
            while bin_centers[i] > lon:
                if i == len(bin_centers) - 1:
                    break
                i += 1

            if i == len(bin_centers) - 1:
                xmap = self.xmaps[-1]
                ymap = self.ymaps[-1]
            else:
                f = (lon - bin_centers[i]) / (bin_centers[i+1] - bin_centers[i])
                if f < 0:
                    xmap = self.xmaps[0]
                    ymap = self.ymaps[0]
                elif f > 1:
                    xmap = self.xmaps[-1]
                    ymap = self.ymaps[-1]
                else:
                    xmap = self.xmaps[i] * (1 - f) + self.xmaps[i + 1] * f
                    ymap = self.ymaps[i] * (1 - f) + self.ymaps[i + 1] * f
            refinement = copy(refinements[0])
            refinement.distortion_x.data = -xmap.astype('float32')
            refinement.distortion_y.data = -ymap.astype('float32')
            output = input.gen_successor()
            output.refinement_applied = refinement
            output.wcs = refinement.apply_to(output.wcs)
            self.outputs.append(output)

        return self.outputs

    def plot_results(self, title_string=""):
        for start, end, xmap, ymap in zip(self.band_starts, self.band_ends, self.xmaps, self.ymaps):
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.imshow(xmap, vmin=-0.8, vmax=0.8, cmap='bwr', origin='lower')
            plt.title("X offset table")
            plt.subplot(122)
            plt.imshow(ymap, vmin=-0.8, vmax=0.8, cmap='bwr', origin='lower')
            plt.colorbar(ax=plt.gcf().axes).set_label("Distortion (px)")
            plt.title("Y offset table")
            plt.suptitle(title_string + f" longitude band [{start:.2f} to {end:.2f}]")
            plt.show()

    def __repr__(self):
        return f"<DistortionMapper, {len(self.inputs)} inputs>"

    def __str__(self):
        return self.__repr__()


def _compute_distortion_map_one_l_band(found_stars, inputs, px_per_bin=16):
    errors = series_errors(found_stars, inputs)
    err_x, err_y, err_px, err_py = calc_binned_err_components(
        errors.px_x, errors.px_y, errors.errors_x, errors.errors_y, ret_coords=True,
        px_per_bin=px_per_bin)

    err_x = filter_distortion_table(err_x).astype(np.float32)
    err_y = filter_distortion_table(err_y).astype(np.float32)

    crpix = (1, 1)
    crval = (err_px[0] + 1, err_py[0] + 1)
    cdelt = ((err_px[1] - err_px[0]), (err_py[1] - err_py[0]))
    dx = PickleableDistortionLookupTable(-err_x, crpix, crval, cdelt)
    dy = PickleableDistortionLookupTable(-err_y, crpix, crval, cdelt)

    refinement = Refinement(distortion_x=dx, distortion_y=dy)

    return refinement, err_x, err_y


class PickleableDistortionLookupTable:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def instantiate(self):
        return DistortionLookupTable(*self.args, **self.kwargs)

    @property
    def data(self):
        return self.args[0]

    @data.setter
    def data(self, value):
        self.args = (value, *self.args[1:])


@dataclass
class Refinement:
    rotation: float = 0
    crpix_x: float = 0
    crpix_y: float = 0
    crval_ra: float = 0
    crval_dec: float = 0
    pv_factors: list[float] = None
    pv_values: list[float] = None
    shear_x: float = 0
    shear_y: float = 0
    is_bad: bool = False
    distortion_x: DistortionLookupTable | PickleableDistortionLookupTable = None
    distortion_y: DistortionLookupTable | PickleableDistortionLookupTable = None
    distortions_are_replacement: bool = False

    # def __add__(self, other: Refinement):
    #     if not isinstance(other, Refinement):
    #         return NotImplemented
    #     new_refinement = Refinement(
    #         rotation=self.rotation + other.rotation,
    #         crpix_x=self.crpix_x + other.crpix_x,
    #         crpix_y=self.crpix_y + other.crpix_y,
    #         crval_ra=self.crval_ra + other.crval_ra,
    #         crval_dec=self.crval_dec + other.crval_dec,
    #         is_bad=self.is_bad or other.is_bad,
    #         distortion_x=self._merge_distortions(
    #             self.distortion_x, other.distortion_x, other.distortions_are_replacement),
    #         distortion_y=self._merge_distortions(
    #             self.distortion_y, other.distortion_y, other.distortions_are_replacement),
    #         distortions_are_replacement=True,
    #         pv_factors=...
    #     )
    #     return new_refinement

    @staticmethod
    def _merge_distortions(d1, d2, d2_is_replacement):
        if d1 and not d2:
            return d1
        if d2 and not d1:
            return d2
        if d2_is_replacement:
            return d2
        d = deepcopy(d2)
        d.data += d1.data
        return d

    def apply_to(self, wcs):
        wcs = wcs.deepcopy()
        ra, dec = wcs.wcs.crval
        wcs.wcs.crval = ra + self.crval_ra, dec + self.crval_dec
        x, y = wcs.wcs.crpix
        wcs.wcs.crpix = x + self.crpix_x, y + self.crpix_y
        matrix = np.array([[np.cos(self.rotation), -np.sin(self.rotation)],
                          [np.sin(self.rotation), np.cos(self.rotation)]])
        wcs.wcs.pc = matrix @ wcs.wcs.pc

        if self.pv_factors is not None or self.pv_values is not None:
            wcs.wcs.set_pv(self.get_resulting_pv_values(wcs))

        if self.shear_x or self.shear_y:
            shear = (np.array([[1, self.shear_x], [0, 1]])
                     @ np.array([[1, 0], [self.shear_y, 1]]))
            wcs.wcs.pc = shear @ wcs.wcs.pc

        if distortion_x := self.distortion_x:
            if isinstance(distortion_x, PickleableDistortionLookupTable):
                distortion_x = distortion_x.instantiate()
            wcs.cpdis1 = distortion_x
        if distortion_y := self.distortion_y:
            if isinstance(distortion_y, PickleableDistortionLookupTable):
                distortion_y = distortion_y.instantiate()
            wcs.cpdis2 = distortion_y

        return wcs

    def get_resulting_pv_values(self, wcs, only_ones_we_modify=False):
        pv = wcs.wcs.get_pv()

        if self.pv_factors is not None:
            for i, pv_pert in enumerate(self.pv_factors):
                for j, elem in enumerate(pv):
                    if elem[0] == 2 and elem[1] == i:
                        elem = (elem[0], elem[1], pv_pert * elem[2])
                        pv[j] = elem
                        break
                else:
                    pv.append((2, i, pv_pert))
        elif self.pv_values is not None:
            for i, pv_value in enumerate(self.pv_values):
                for j, elem in enumerate(pv):
                    if elem[0] == 2 and elem[1] == i:
                        elem = (elem[0], elem[1], pv_value)
                        pv[j] = elem
                        break
                else:
                    pv.append((2, i, pv_value))

        if only_ones_we_modify:
            pv = [p for p in pv if p[0] == 2]
        return pv


class InputFile:
    def __init__(self, path):
        self.path = path
        header = fits.getheader(path)
        self.wcs = WCS(header, key='A')
        self.timestamp = utils.to_timestamp(path)
        self.refinement_applied: Refinement = None
        self.previous = None
        _, lon = utils.load_orbit_plane_rtheta(header)
        self.longitude = lon[0] * 180 / np.pi

    def write_updated_file(self, out_path):
        update_dict = self.gen_update_dict()
        update_hdul = self.wcs.to_fits(key='A')

        with fits.open(self.path) as hdul:
            for k, v in update_dict.items():
                hdul[0].header[k] = v

            if len(update_hdul) > 1:
                for key in ('CPDIS1', 'CPDIS2',
                            'DP1.AXIS.1', 'DP1.AXIS.2', 'DP1.EXTVER', 'DP1.NAXES',
                            'DP2.AXIS.1', 'DP2.AXIS.2', 'DP2.EXTVER', 'DP2.NAXES',
                            'EXTEND'):
                    hdul[0].header[key] = update_hdul[0].header[key]
                hdul.extend(update_hdul[-2:])

            hdul.write_to(out_path)

    def gen_update_dict(self):
        update_hdul = self.wcs.to_fits(key='A')
        update_dict = {}
        update_dict['CRVAL1A'] = update_hdul[0].header['CRVAL1A']
        update_dict['CRVAL2A'] = update_hdul[0].header['CRVAL2A']
        update_dict['PC1_1A'] = update_hdul[0].header['PC1_1A']
        update_dict['PC2_1A'] = update_hdul[0].header['PC2_1A']
        update_dict['PC1_2A'] = update_hdul[0].header['PC1_2A']
        update_dict['PC2_2A'] = update_hdul[0].header['PC2_2A']

        node = self
        net_rotation = node.refinement_applied.rotation
        while node.previous:
            node = node.previous
            net_rotation += node.rotation
        origin_node = node
        rot = np.array([[np.cos(net_rotation), -np.sin(net_rotation)],
                        [np.sin(net_rotation), np.cos(net_rotation)]])

        with fits.open(self.path) as hdul:
            # Using the original celestial--helio alignment, get the HP coord that matches our new CRVAL
            ref_px = origin_node.wcs.world_to_pixel_values(*self.wcs.wcs.crval)
            origin_hp_wcs = WCS(hdul[0].header)
            crval = origin_hp_wcs.pixel_to_world_values(ref_px)
            update_dict['CRVAL1'] = crval[0]
            update_dict['CRVAL2'] = crval[1]

            pc = rot @ origin_hp_wcs.wcs.pc
            update_dict['PC1_1'] = pc[0, 0]
            update_dict['PC1_2'] = pc[0, 1]
            update_dict['PC2_1'] = pc[1, 0]
            update_dict['PC2_2'] = pc[1, 1]

            for key in update_hdul[0].header:
                # These updates can be copied directly from the celestial WCS to HP
                if key[:4] in ('CRPI', 'PV2_'):
                    update_dict[key] = update_hdul[0].header[key]
                    # These updates can be copied directly from the celestial WCS to HP (by removing the trailing 'A'
                    # from the key)
                    update_dict[key[-1:]] = update_hdul[0].header[key]

        return update_dict

    def gen_successor(self):
        out = copy(self)
        out.previous = self
        return out


def make_cutout(x, y, data, cutout_size, normalize=True):
    """
    Cuts a section from a data array centered on a coordinate and normalizes it.
    
    Raises an error if the cutout extends beyond the data bounds.
    
    Parameters
    ----------
    x, y : float
        Floating-point array indices around which to center the cutout
    data : array
        The data out of which to take the cutout
    cutout_size : int
        The size of the square cutout, in pixels
    normalize : boolean
        Whether to normalize the data in the cutout
    
    Returns
    -------
    cutout : array
        The cutout
    cutout_start_x, cutout_start_y : int
        The array indices in the full array of the first row/column in the
        cutout
    """
    cutout_size = int(round(cutout_size))
    
    cutout_start_x = int(round(x)) - cutout_size//2
    cutout_start_y = int(round(y)) - cutout_size//2
    
    assert 0 < cutout_start_y < data.shape[0] - cutout_size + 1
    assert 0 < cutout_start_x < data.shape[1] - cutout_size + 1
    
    cutout = data[
            cutout_start_y:cutout_start_y + cutout_size,
            cutout_start_x:cutout_start_x + cutout_size]
    
    if normalize:
        cutout = cutout - np.min(cutout)
        with np.errstate(invalid='raise'):
            cutout = cutout / np.max(cutout)
    
    return cutout, cutout_start_x, cutout_start_y


MIN_SIGMA = 0.05
MAX_SIGMA = 1.5
def fit_star(x, y, data, all_stars_x, all_stars_y, cutout_size=9,
        ret_more=False, ret_star=False, binning=2, start_at_max=True,
        normalize_cutout=True):
    bin_factor = 2 / binning
    cutout_size = int(round(cutout_size * bin_factor))
    if cutout_size % 2 != 1:
        cutout_size += 1
    
    try:
        cutout, cutout_start_x, cutout_start_y = make_cutout(
                x, y, data, cutout_size, normalize_cutout)
    except FloatingPointError:
        err = ["Invalid values encountered"]
        if ret_star:
            return None, None, None, err
        if not ret_more:
            return np.nan, np.nan, np.nan, np.nan, np.nan, err
    
    cutout = cutout.astype(float)
    cutout_size = cutout.shape[0]
    
    err = []
    if np.any(np.isnan(cutout)):
        err.append("NaNs in cutout")
        if ret_star:
            return None, None, None, err
        if not ret_more:
            return x, y, np.nan, np.nan, np.nan, err
    
    if all_stars_x is None:
        all_stars_x = np.array([x])
    else:
        all_stars_x = np.asarray(all_stars_x)
    if all_stars_y is None:
        all_stars_y = np.array([y])
    else:
        all_stars_y = np.asarray(all_stars_y)
    
    n_in_cutout = np.sum(
        (all_stars_x > cutout_start_x - .5)
        * (all_stars_x < cutout_start_x + cutout_size - .5)
        * (all_stars_y > cutout_start_y - .5)
        * (all_stars_y < cutout_start_y + cutout_size - .5))
    
    if n_in_cutout > 1:
        err.append("Crowded frame")
        if ret_star:
            return None, None, None, err
        if not ret_more:
            return x, y, np.nan, np.nan, np.nan, err
    
    if start_at_max:
        i_max = np.argmax(cutout)
        y_start, x_start = np.unravel_index(i_max, cutout.shape)
    else:
        x_start = x - cutout_start_x
        y_start = y - cutout_start_y
    
    with warnings.catch_warnings():
        warnings.filterwarnings(action='error')
        try:
            # We'll follow astropy's example and apply bounds ourself with the
            # lm fitter, which is 10x faster than using the bounds-aware fitter
            # without any real change in outputs.
            bounds=np.array([
                (0,             # amplitude
                 -1,            # x0
                 -1,            # y0
                 -np.inf,       # x_std
                 -np.inf,       # y_std
                 -2*np.pi,      # theta
                 -np.inf,       # intercept
                 -np.inf,       # x slope
                 -np.inf),      # y slope
                (np.inf,        # amplitude
                 cutout_size,   # x0
                 cutout_size,   # y0
                 np.inf,        # x_std
                 np.inf,        # y_std
                 2*np.pi,       # theta
                 np.inf,        # intercept
                 np.inf,        # x slope
                 np.inf),       # y slope
                ])
            x0 = [cutout.max(),      # amplitude
                  x_start,           # x0
                  y_start,           # y0
                  bin_factor,        # x_std
                  bin_factor,        # y_std
                  0,                 # theta
                  np.median(cutout), # intercept
                  0,                 # x slope
                  0,                 # y slope
                 ]
            res = scipy.optimize.least_squares(
                    model_error,
                    x0,
                    args=(cutout, bounds),
                    method='lm'
                )
            A, xc, yc, xstd, ystd, theta, intercept, slope_x, slope_y = res.x
        except RuntimeWarning:
            err.append("No solution found")
            if ret_more:
                return None, cutout, err, cutout_start_x, cutout_start_y
            elif ret_star:
                return None, None, None, err
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan, err
    
    max_std = MAX_SIGMA * bin_factor
    min_std = MIN_SIGMA * bin_factor
    
    if A < 0.5 * (np.max(cutout) - intercept):
        err.append("No peak found")
    if xstd > max_std or ystd > max_std:
        err.append("Fit too wide")
    if xstd < min_std or ystd < min_std:
        err.append("Fit too narrow")
    if (not (0 < xc < cutout_size - 1)
            or not (0 < yc < cutout_size - 1)):
        err.append("Fitted peak too close to edge")
    
    if ret_more:
        return res, cutout, err, cutout_start_x, cutout_start_y
    if ret_star:
        star = model_fcn(
                (A, xc, yc, xstd, ystd, theta, 0, 0, 0),
                cutout)
        return star, cutout_start_x, cutout_start_y, err
    return (xc + cutout_start_x,
            yc + cutout_start_y,
            xstd,
            ystd,
            theta,
            err)


@njit
def model_fcn(params, cutout):
    x = np.empty(cutout.shape, dtype=np.int64)
    y = np.empty(cutout.shape, dtype=np.int64)
    for i in range(cutout.shape[0]):
        for j in range(cutout.shape[1]):
            x[i, j] = j
            y[i, j] = i
    
    A, xc, yc, xstd, ystd, theta, intercept, slope_x, slope_y = params
    
    a = np.cos(theta)**2 / (2 * xstd**2) + np.sin(theta)**2 / (2 * ystd**2)
    b = np.sin(2*theta)  / (2 * xstd**2) - np.sin(2*theta)  / (2 * ystd**2)
    c = np.sin(theta)**2 / (2 * xstd**2) + np.cos(theta)**2 / (2 * ystd**2)
    
    model = (
        A * np.exp(
            - a * (x-xc)**2
            - b * (x-xc) * (y-yc)
            - c * (y-yc)**2
        )
        + intercept + slope_x * x + slope_y * y
    )
    
    return model


@njit
def model_error(params, cutout, bounds):
    for i in range(len(params)):
        if params[i] < bounds[0][i]:
            params[i] = bounds[0][i]
        if params[i] > bounds[1][i]:
            params[i] = bounds[1][i]
    model = model_fcn(params, cutout)
    return (model - cutout).flatten()

DIM_CUTOFF = 8
BRIGHT_CUTOFF = 2

def prep_frame_for_star_finding(input, dim_cutoff=DIM_CUTOFF,
        bright_cutoff=BRIGHT_CUTOFF, corrector=None):
    with utils.ignore_fits_warnings(), fits.open(input.path) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header
        w = input.wcs
    
    if corrector is not None:
        data = corrector(data)
    
    if data.shape not in ((1024, 960), (2048, 1920)):
        raise ValueError(
                f"Strange image shape {data.shape} in {input.path}---skipping")
    
    if hdr['nbin1'] != hdr['nbin2']:
        raise ValueError(f"There's some weird binning going on in {input.path}")
    binning = hdr['nbin1']
    bin_factor = 2 / binning

    trim = (40, 20, 20, 20)
    trim = [int(round(t * bin_factor)) for t in trim]
    
    (stars_x, stars_y, stars_vmag, stars_ra, stars_dec, all_stars_x,
            all_stars_y) = star_tools.find_expected_stars_in_frame(
                    (hdr, w), trim=trim, dim_cutoff=dim_cutoff,
                    bright_cutoff=bright_cutoff)
    
    return (stars_x + trim[0], stars_y + trim[2], stars_vmag, stars_ra, stars_dec,
            all_stars_x, all_stars_y, data, binning)

def find_stars_in_frame(data):
    input, start_at_max, include_shapes, corrector = data
    t = input.timestamp
    try:
        (stars_x, stars_y, stars_vmag, stars_ra, stars_dec,
                all_stars_x, all_stars_y, data,
                binning) = prep_frame_for_star_finding(
                    input, corrector=corrector)
    except ValueError as e:
        print(e)
        return [], [], [], {}

    good = []
    crowded_out = []
    bad = []
    codes = {}
    
    for x, y, ra, dec, vmag in zip(
            stars_x, stars_y, stars_ra, stars_dec, stars_vmag):
        fx, fy, fxstd, fystd, theta, err = fit_star(
                x, y, data, all_stars_x, all_stars_y,
                ret_more=False, binning=binning,
                start_at_max=start_at_max)

        if include_shapes:
            star = FoundStarWithStd(
                x=fx, y=fy,
                ra=ra, dec=dec, vmag=vmag, t = t,
                xstd=fxstd, ystd=fystd, theta=theta,
            )
        else:
            star = FoundStar(
                x = fx, y = fy,
                ra = ra, dec = dec, vmag = vmag, t = t,
            )
        if len(err) == 0:
            good.append(star)
        elif 'Crowded frame' in err:
            crowded_out.append(star)
        else:
            bad.append(star)
        for e in err:
            codes[e] = codes.get(e, 0) + 1
    return good, bad, crowded_out, codes


def find_all_stars(inputs, ret_all=False, start_at_max=True,
                   include_shapes=False, corrector=None):
    res = process_map(
            find_stars_in_frame,
            zip(inputs, repeat(start_at_max), repeat(include_shapes),
                repeat(corrector)),
            total=len(inputs))

    good = []
    crowded_out = []
    bad = []
    codes = {}
    found_stars = FoundStars()

    for input, (good_in_frame, bad_in_frame, crowded_in_frame,
            codes_in_frame) in zip(inputs, res):
        good.extend(good_in_frame)
        bad.extend(bad_in_frame)
        crowded_out.extend(crowded_in_frame)
        for code, count in codes_in_frame.items():
            codes[code] = codes.get(code, 0) + count

        for star_data in good_in_frame:
            found_stars.add_star(star_data)
    
    if ret_all:
        good_x = np.array([star.x for star in good])
        good_y = np.array([star.y for star in good])
        crowded_out_x = np.array([star.x for star in crowded_out])
        crowded_out_y = np.array([star.y for star in crowded_out])
        bad_x = np.array([star.x for star in bad])
        bad_y = np.array([star.y for star in bad])
        return (found_stars, good_x, good_y, crowded_out_x,
                crowded_out_y, bad_x, bad_y, codes)
    return found_stars


@dataclass
class FoundStar:
    ra: float
    dec: float
    x: float
    y: float
    t: float
    vmag: float


@dataclass
class FoundStarWithStd(FoundStar):
    xstd: float
    ystd: float
    theta: float


class FoundStars:
    def __init__(self):
        self.mapping_by_coord = dict()
        self.mapping_by_time = dict()

    def add_star(self, star: FoundStar):
        key = (star.ra, star.dec)
        if key not in self.mapping_by_coord:
            self.mapping_by_coord[key] = []
        self.mapping_by_coord[key].append(star)
        if star.t not in self.mapping_by_time:
            self.mapping_by_time[star.t] = []
        self.mapping_by_time[star.t].append(star)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.mapping_by_coord[item]
        return self.mapping_by_time[item]

    def filter_for_time(self, tmin, tmax):
        out = FoundStars()
        for stars in self.mapping_by_time.values():
            for star in stars:
                if tmin <= star.t <= tmax:
                    out.add_star(star)
        return out


def do_iteration_with_crpix(pts1, pts2, w1, w2, angle_start, dra_start,
        ddec_start, dx_start, dy_start):
    def f(args):
        angle, dra, ddec, dx, dy = args
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        w22 = w2.deepcopy()
        w22.wcs.crval = w22.wcs.crval + np.array([dra, ddec])
        w22.wcs.crpix = w22.wcs.crpix + np.array([dx, dy])
        w22.wcs.pc = rot @ w22.wcs.pc
        pts22 = np.array(w1.world_to_pixel(w22.pixel_to_world(
            pts2[:, 0], pts2[:, 1]))).T
        ex = pts22[:, 0] - pts1[:, 0]
        ey = pts22[:, 1] - pts1[:, 1]
        err = np.sqrt(ex**2 + ey**2)
        return err
    res = scipy.optimize.least_squares(
            f,
            [angle_start, dra_start, ddec_start, dx_start, dy_start],
            bounds=[[-np.pi, -np.inf, -np.inf, -np.inf, -np.inf],
                    [np.pi, np.inf, np.inf, np.inf, np.inf]])
    return res, res.x


def do_iteration_no_crpix(ras, decs, xs_true, ys_true, w):
    def f(args):
        angle, dra, ddec = args
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        w2 = w.deepcopy()
        w2.wcs.crval = w2.wcs.crval + np.array([dra, ddec])
        w2.wcs.pc = rot @ w2.wcs.pc
        xs, ys = np.array(w2.all_world2pix(ras, decs, 0))
        ex = xs_true - xs
        ey = ys_true - ys
        err = np.sqrt(ex**2 + ey**2)
        # err[err > 1] = 1
        return err
    res = scipy.optimize.least_squares(
            f,
            [0, 0, 0],
            bounds=[[-np.pi, -np.inf, -np.inf],
                    [np.pi, np.inf, np.inf]],
                                      # loss='cauchy'
                                      )
    return res, *res.x, 0, 0


do_iteration = do_iteration_no_crpix


def iteratively_align_one_file(data):
    input, series_data = data
    t = input.timestamp
    
    # Check up here, especially in case the frame has *zero* identified stars
    # (i.e. a very bad frame)
    if len(series_data) < 50:
        print(f"Only {len(series_data)} stars found in {input.path}---skipping")
        refinement = Refinement(
            rotation=np.nan,
            crpix_x=np.nan,
            crpix_y=np.nan,
            crval_ra=np.nan,
            crval_dec=np.nan,
            is_bad=True,
        )
        return refinement, t, np.nan
    
    ras, decs, xs, ys = [], [], [], []
    for star in series_data:
        ras.append(star.ra)
        decs.append(star.dec)
        xs.append(star.x)
        ys.append(star.y)
    xs = np.array(xs)
    ys = np.array(ys)
    ras = np.array(ras)
    decs = np.array(decs)
    
    xs_comp, ys_comp = np.array(input.wcs.all_world2pix(ras, decs, 0))
    dx = xs_comp - xs
    dy = ys_comp - ys
    dr = np.sqrt(dx**2 + dy**2)
    outlier_cutoff = dr.mean() + 2 * dr.std()
    inliers = dr < outlier_cutoff
    
    xs = xs[inliers]
    ys = ys[inliers]
    ras = ras[inliers]
    decs = decs[inliers]
    
    # Check down here after outlier removal
    if len(xs) < 50:
        print(f"Only {len(xs)} stars found in {input.path}"
               "after outlier removal---skipping")
        refinement = Refinement(
            rotation=np.nan,
            crpix_x=np.nan,
            crpix_y=np.nan,
            crval_ra=np.nan,
            crval_dec=np.nan,
            is_bad=True,
        )
        return refinement, t, np.nan

    res, angle, dra, ddec, dx, dy = do_iteration(ras, decs, xs, ys, input.wcs)
    rmse = np.sqrt(np.mean(np.sqrt(res.fun)))

    refinement = Refinement(
        rotation=angle,
        crpix_x=dx,
        crpix_y=dy,
        crval_ra=dra,
        crval_dec=ddec,
    )
    return refinement, t, rmse


def iteratively_align_files(inputs, found_stars, smooth_offsets=True):
    data = process_map(iteratively_align_one_file, zip(
                           inputs,
                           (found_stars[input.timestamp]
                            for input in inputs)),
                       total=len(inputs), chunksize=5)

    refinements, t_vals, rmses = zip(*data)
    is_ok = [not r.is_bad for r in refinements]
    refinements = [r for r, ok in zip(refinements, is_ok) if ok]
    t_vals = [t for t, ok in zip(t_vals, is_ok) if ok]
    rmses = [r for r, ok in zip(rmses, is_ok) if ok]
    inputs = [i for i, ok in zip(inputs, is_ok) if ok]

    angle_ts = [r.rotation for r in refinements]
    rpix_ts = np.array([(r.crpix_x, r.crpix_y) for r in refinements])
    rval_ts = np.array([(r.crval_ra, r.crval_dec) for r in refinements])
    
    if smooth_offsets:
        angle_ts = smooth_curve(t_vals, angle_ts,
                sig=smooth_offsets, n_sig=3, outlier_sig=2)
        rpix_ts[:, 0] = smooth_curve(t_vals, rpix_ts[:, 0],
                sig=smooth_offsets, n_sig=3, outlier_sig=2)
        rpix_ts[:, 1] = smooth_curve(t_vals, rpix_ts[:, 1],
                sig=smooth_offsets, n_sig=3, outlier_sig=2)
        rval_ts[:, 0] = smooth_curve(t_vals, rval_ts[:, 0],
                sig=smooth_offsets, n_sig=3, outlier_sig=2)
        rval_ts[:, 1] = smooth_curve(t_vals, rval_ts[:, 1],
                sig=smooth_offsets, n_sig=3, outlier_sig=2)

        for i, r in enumerate(refinements):
            r.rotation = angle_ts[i]
            r.crpix_x, r.crpix_y = rpix_ts[i, :]
            r.crval_ra, r.crval_dec = rval_ts[i, :]
            r.is_bad = False

    outputs = [i.gen_successor() for i in inputs]
    for i, output in enumerate(outputs):
        output.wcs = refinements[i].apply_to(output.wcs)
        output.refinement_applied = refinements[i]
    
    return outputs, angle_ts, rval_ts, rpix_ts, t_vals, rmses


def smooth_curve(x, y, sig=3600*6.5, n_sig=3, outlier_sig=2):
    """
    Applies a Gaussian filter to an unevenly-spaced 1D signal.
    
    The Gaussian is evaluated with resepct to x-axis values, rather than
    array coordinates (or pixel indicies, etc.)
    
    NaN values are ignored in the kernel integration, and NaNs in the input
    will be replaced with a smoothed value.
    
    Parameters
    ----------
    x, y : numpy arrays
        x and y values of the signal
    sig
        Width of the standard deviation of the Gaussian, in the same units as
        ``x``
    n_sig
        The Gaussian kernel is integrated out to this many standard deviations
    outlier_sig
        Before integrating the kernel, the window from -n_sig to +n_sig is
        checked for outliers. Any point deviating from the window's mean by at
        least ``outlier_sig`` times the window standard deviation is ignored.
    """
    output_array = np.zeros_like(y, dtype=float)
    not_nan = np.isfinite(y)
    for i in range(len(y)):
        f = np.abs(x - x[i]) <= n_sig * sig
        f *= not_nan
        xs = x[f]
        ys = y[f]
        window_std = np.std(ys)
        # Skip outlier rejection if there's no variation within the window
        if window_std > 0:
            f = np.abs(ys - np.mean(ys)) <= outlier_sig * window_std
            xs = xs[f]
            ys = ys[f]
        weight = np.exp(-(x[i] - xs)**2 / sig**2)
        output_array[i] = np.sum(weight * ys) / weight.sum()
    return output_array


def iteratively_perturb_projections(inputs, found_stars,
                                    also_shear=False, n_extra_params=0,
                                    do_print=True, weights=None):
    wcses = []
    all_ras = []
    all_decs = []
    all_xs_true = []
    all_ys_true = []
    
    if weights is None:
        weights = np.ones(len(inputs))
    
    if do_print:
        print("Reading files...")
    for input in inputs:
        t = input.timestamp
        series_data = found_stars[t]
        wcs = input.wcs

        # Check up here, especially in case the frame has *zero* identified
        # stars (i.e. a very bad frame)
        if len(series_data) < 50:
            print(f"Only {len(series_data)} stars found in "
                  f"{input.path}---skipping")
            continue

        ras, decs, xs, ys = [], [], [], []
        for star in series_data:
            ras.append(star.ra)
            decs.append(star.dec)
            xs.append(star.x)
            ys.append(star.y)
        ras = np.array(ras)
        decs = np.array(decs)
        xs = np.array(xs)
        ys = np.array(ys)

        xs_comp, ys_comp = np.array(wcs.all_world2pix(ras, decs, 0))
        dx = xs_comp - xs
        dy = ys_comp - ys
        dr = np.sqrt(dx**2 + dy**2)
        outlier_cutoff = dr.mean() + 2 * dr.std()
        inliers = dr < outlier_cutoff

        xs = xs[inliers]
        ys = ys[inliers]
        ras = ras[inliers]
        decs = decs[inliers]

        # Check down here after outlier removal
        if len(series_data) < 50:
            print(f"Only {len(series_data)} stars found in {input.path} "
                   "after outlier removal---skipping")
            continue

        all_ras.append(ras)
        all_decs.append(decs)
        all_xs_true.append(xs)
        all_ys_true.append(ys)

        wcses.append(wcs)
    
    if do_print:
        print("Doing iteration...")

    def f(pv_perts):
        if also_shear:
            shear_x, shear_y = pv_perts[0:2]
            pv_perts = pv_perts[2:]
        else:
            shear_x, shear_y = 0, 0
        refinement = Refinement(pv_factors=pv_perts, shear_x=shear_x, shear_y=shear_y)
        
        err = []
        for w, ras, decs, xs, ys, weight in zip(
                wcses, all_ras, all_decs, all_xs_true, all_ys_true, weights):
            w = refinement.apply_to(w)
            
            xs_comp, ys_comp = np.array(w.all_world2pix(ras, decs, 0))
            ex = xs - xs_comp
            ey = ys - ys_comp
            err.append(np.sqrt(ex**2 + ey**2) * np.sqrt(weight))
        
        return np.concatenate(err)
    
    n_pvs = len(
        [e for e in wcses[0].wcs.get_pv() if e[0] == 2])
    res = scipy.optimize.least_squares(
            f, ([0, 0] if also_shear else []) + ([1] * n_pvs) + ([0] * n_extra_params))
    
    if also_shear:
        shear_x = res.x[0]
        shear_y = res.x[1]
        res.x = res.x[2:]
    else:
        shear_x, shear_y = 0, 0

    refinement = Refinement(pv_factors=res.x, shear_x=shear_x, shear_y=shear_y)

    return refinement


def update_file_with_projection(input_fname, update_dict, out_dir,
        also_shear=False, shear=None):
    with utils.ignore_fits_warnings():
        with fits.open(input_fname) as hdul:
            hdul[0].header.update(update_dict)
            if also_shear:
                for wcs_key in ('A', ' '):
                    w = WCS(hdul[0].header, key=wcs_key)
                    w.wcs.pc = shear @ w.wcs.pc
                    update = w.to_header(key=wcs_key)
                    for k in update:
                        if k.startswith("PC"):
                            hdul[0].header[k] = update[k]
            hdul.writeto(os.path.join(out_dir,
                         os.path.basename(input_fname)),
                         overwrite=True)


def calc_binned_err_components(px_x, px_y, err_x, err_y, ret_coords=False,
                               px_per_bin=16):
    if np.any(px_x > 960) or np.any(px_y > 1024):
        raise ValueError("Unexpected binning for this image")
    
    berr_x, r, c, _ = scipy.stats.binned_statistic_2d(
        px_y, px_x, err_x, 'median', (1024//px_per_bin, 960//px_per_bin),
        expand_binnumbers=True,
        range=((0, 1024), (0, 960)))
    
    berr_y, _, _, _ = scipy.stats.binned_statistic_2d(
        px_y, px_x, err_y, 'median', (1024//px_per_bin, 960//px_per_bin),
        expand_binnumbers=True,
        range=((0, 1024), (0, 960)))
    
    if ret_coords:
        r = (r[1:] + r[:-1]) / 2
        c = (c[1:] + c[:-1]) / 2
        return berr_x, berr_y, c, r
    
    return berr_x, berr_y


def filter_distortion_table(data, blur_sigma=4, med_filter_size=3):
    """
    Returns a filtered copy of a distortion map table.
    
    Any rows/columns at the edges that are all NaNs will be removed and
    replaced with a copy of the closest non-removed edge at the end of
    processing.
    
    Any NaN values that don't form a complete edge row/column will be replaced
    with the median of all surrounding non-NaN pixels.
    
    Then median filtering is performed across the whole map to remove outliers,
    and Gaussian filtering is applied to accept only slowly-varying
    distortions.
    
    Parameters
    ----------
    data
        The distortion map to be filtered
    blur_sigma : float
        The number of pixels constituting one standard deviation of the
        Gaussian kernel. Set to 0 to disable Gaussian blurring.
    med_filter_size : int
        The size of the local neighborhood to consider for median filtering.
        Set to 0 to disable median filtering.
    """
    data = data.copy()
    
    # Trim empty (all-nan) rows and columns
    trimmed = []
    i = 0
    while np.all(np.isnan(data[0])):
        i += 1
        data = data[1:]
    trimmed.append(i)

    i = 0
    while np.all(np.isnan(data[-1])):
        i += 1
        data = data[:-1]
    trimmed.append(i)

    i = 0
    while np.all(np.isnan(data[:, 0])):
        i += 1
        data = data[:, 1:]
    trimmed.append(i)

    i = 0
    while np.all(np.isnan(data[:, -1])):
        i += 1
        data = data[:, :-1]
    trimmed.append(i)
    
    # Replace interior nan values with the median of the surrounding values.
    # We're filling in from neighboring pixels, so if there are any nan pixels
    # fully surrounded by nan pixels, we need to iterate a few times.
    while np.any(np.isnan(data)):
        nans = np.nonzero(np.isnan(data))
        replacements = np.zeros_like(data)
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='All-NaN slice')
            for r, c in zip(*nans):
                r1, r2 = r-1, r+2
                c1, c2 = c-1, c+2
                r1, r2 = max(r1, 0), min(r2, data.shape[0])
                c1, c2 = max(c1, 0), min(c2, data.shape[1])

                replacements[r, c] = np.nanmedian(data[r1:r2, c1:c2])
        data[nans] = replacements[nans]
    
    # Median-filter the whole image
    if med_filter_size:
        data = scipy.ndimage.median_filter(data, size=med_filter_size, mode='reflect')
    
    # Gaussian-blur the whole image
    if blur_sigma > 0:
        data = scipy.ndimage.gaussian_filter(data, sigma=blur_sigma)
    
    # Replicate the edge rows/columns to replace those we trimmed earlier
    data = np.pad(data, [trimmed[0:2], trimmed[2:]], mode='edge')
    
    return data


def add_distortion_table(fname, outname, err_x, err_y, err_px, err_py):
    """
    Adds two distortion maps to a FITS file, for x and y distortion.
    
    Parameters
    ----------
    fname
        The path to the input FITS file, to which distortions should be added
    outname
        The path to which the updated FITS file should be saved. If ``None``,
        the updated ``HDUList`` is returned instead.
    err_x, err_y
        The distortion values, given in the sense of "the coordinate computed
        for a pixel is offset by this much from its true location". The
        negative of these values will be stored as the distortion map, and that
        is the amount by which pixel coordinates will be shifted before being
        converted to world coordinates.
    err_px, err_py
        The x or y coordinate associated with each pixel in the provided
        distortion maps
    """
    dx = DistortionLookupTable(-err_x.astype(np.float32),
                               (1, 1),
                               (err_px[0] + 1, err_py[0] + 1),
                               ((err_px[1] - err_px[0]),
                                   (err_py[1] - err_py[0])))
    dy = DistortionLookupTable(-err_y.astype(np.float32),
                               (1, 1),
                               (err_px[0] + 1, err_py[0] + 1),
                               ((err_px[1] - err_px[0]),
                                   (err_py[1] - err_py[0])))
    with utils.ignore_fits_warnings():
        data, header = fits.getdata(fname, header=True)
        wcs = WCS(header, key='A')
    wcs.cpdis1 = dx
    wcs.cpdis2 = dy
    hdul = wcs.to_fits()
    
    for key in ('extend', 'cpdis1', 'cpdis2',
                'dp1.EXTVER', 'dp1.NAXES', 'dp1.AXIS.1', 'dp1.AXIS.2',
                'dp2.EXTVER', 'dp2.NAXES', 'dp2.AXIS.1', 'dp2.AXIS.2'):
        header[key] = hdul[0].header[key]
    hdul[0].header = header
    hdul[0].data = data
    
    if outname is None:
        return hdul
    with utils.ignore_fits_warnings():
        hdul.writeto(outname, overwrite=True)


def generate_combined_map(search_dir, version_str,
                          subdir='proj_tweaked_images', use_outer=False):
    inner_outer = "_O_" if use_outer else "_I_"
    work_dirs = [
            os.path.join(search_dir, f) for f in sorted(os.listdir(search_dir))
                if version_str in f and inner_outer in f]
    print(("Outer" if use_outer else "Inner") + " FOV")
    for work_dir in work_dirs:
        print(work_dir)
    
    all_errors_x = []
    all_errors_y = []
    all_px_x = []
    all_px_y = []
    
    for work_dir in work_dirs:
        with open(os.path.join(work_dir, 'stars_db_r2.pkl'), 'rb') as f:
            series, sbf = pickle.load(f)
        files = utils.collect_files(os.path.join(work_dir, subdir), separate_detectors=False)
        _, _, px_x, px_y, errors_x, errors_y = series_errors(series, files)
        
        all_errors_x.extend(errors_x)
        all_errors_y.extend(errors_y)
        
        all_px_x.extend(px_x)
        all_px_y.extend(px_x)
    
    errors_x = np.array(errors_x)
    errors_y = np.array(errors_y)
    errors = np.sqrt(errors_x**2 + errors_y**2)
    px_x = np.array(px_x)
    px_y = np.array(px_y)
    
    filter = errors < 2
    err_x, err_y, err_px, err_py = calc_binned_err_components(
            px_x[filter], px_y[filter], errors_x[filter], errors_y[filter],
            ret_coords=True)
    
    display(HTML("<h3>Merged error map</h3>"))
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(err_x, vmin=-0.8, vmax=0.8, cmap='bwr', origin='lower')
    plt.title("X offset table")
    plt.subplot(132)
    plt.imshow(err_y, vmin=-0.8, vmax=0.8, cmap='bwr', origin='lower')
    plt.colorbar(ax=plt.gcf().axes[:2]).set_label("Distortion (px)")
    plt.title("Y offset table")
    plt.subplot(133)
    plt.imshow(np.sqrt(err_x**2 + err_y**2), vmin=0, vmax=1, origin='lower')
    plt.colorbar().set_label("Distortion amplitude (px)")
    plt.title("Error magnitude")
    plt.suptitle("Unsmoothed")
    plt.show()
    
    err_x = filter_distortion_table(err_x)
    err_y = filter_distortion_table(err_y)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(err_x, vmin=-0.8, vmax=0.8, cmap='bwr', origin='lower')
    plt.title("X offset table")
    plt.subplot(132)
    plt.imshow(err_y, vmin=-0.8, vmax=0.8, cmap='bwr', origin='lower')
    plt.colorbar(ax=plt.gcf().axes[:2]).set_label("Distortion (px)")
    plt.title("Y offset table")
    plt.subplot(133)
    plt.imshow(np.sqrt(err_x**2 + err_y**2), vmin=0, vmax=1, origin='lower')
    plt.colorbar().set_label("Distortion amplitude (px)")
    plt.title("Error magnitude")
    plt.suptitle("Smoothed")
    plt.show()

    return err_x, err_y, err_px, err_py, work_dirs


def _write_combined_map(err_x, err_y, ifile, *ofiles, collect_wcs=False,
                        err_px=None, err_py=None):
    with utils.ignore_fits_warnings(), fits.open(ifile) as hdul:
        if len(hdul) > 1:
            hdul[1].data = -err_x.astype(hdul[1].data.dtype)
            hdul[2].data = -err_y.astype(hdul[2].data.dtype)
        else:
            hdul = add_distortion_table(
                ifile, None, err_x, err_y, err_px, err_py)
        for ofile in ofiles:
            hdul.writeto(ofile)
        if collect_wcs:
            return WCS(hdul[0].header, hdul, key='A')


def write_combined_maps(err_x, err_y, work_dirs, *out_dirs,
                        delete_existing=False, collect_wcses=False,
                        err_px=None, err_py=None):
    """
    Write a merged error map to many files
    
    Parameters
    ----------
    err_x, err_y : ``np.ndarray``
        The merged error map, to be written directly into the correct HDUs
    work_dirs : ``list`` of ``str``
        A list of input directories, each containing files that should receive
        the merged maps.
    out_dirs : one or multiple ``list``s of ``str``
        The output directories. Each file can be written to multiple output
        directories. If ``work_dirs`` is length N, and each file is to be
        written into M directories, then M lists should be provided, each
        containing N entries.
    delete_existing : ``bool``
        Whether to delete existing output directories
    """
    ifiles = []
    ofiles = [[] for x in out_dirs]
    for i, work_dir in enumerate(work_dirs):
        files = utils.collect_files(work_dir, separate_detectors=False)
        ifiles.extend(files)
        for j, out_dir in enumerate(out_dirs):
            ofiles[j].extend([
                os.path.join(out_dir[i], os.path.basename(f)) for f in files])

    for out_dir in chain(*out_dirs):
        if os.path.exists(out_dir):
            if delete_existing:
                shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    wcses = {}
    for i in tqdm(range(len(ifiles))):
        ifile = ifiles[i]
        ofile = [of[i] for of in ofiles]
        wcs = _write_combined_map(err_x, err_y, ifile, *ofile,
            collect_wcs=collect_wcses, err_px=err_px, err_py=err_py)
        if collect_wcses:
            wcses[utils.to_timestamp(ifile)] = wcs
    if collect_wcses:
        return wcses


def series_errors(found_stars, inputs, warn_for_missing=True):
    wcses = {i.timestamp: i.wcs for i in inputs}

    errors = []
    errors_x = []
    errors_y = []
    px_x = []
    px_y = []
    missing_keys = []
    for k, stars in found_stars.mapping_by_coord.items():
        ra, dec = k
        
        x_comp = []
        y_comp = []
        xs = []
        ys = []
        for star in stars:
            try:
                wcs = wcses[star.t]
            except KeyError:
                missing_keys.append(star.t)
                continue
            xs.append(star.x)
            ys.append(star.y)
            x, y = wcs.all_world2pix(ra, dec, 0)
            x_comp.append(x)
            y_comp.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        x_comp = np.array(x_comp)
        y_comp = np.array(y_comp)
        
        dx = xs - x_comp
        dy = ys - y_comp
        dr = np.sqrt(dx**2 + dy**2)
        errors.extend(dr)
        errors_x.extend(dx)
        errors_y.extend(dy)
        px_x.extend(xs)
        px_y.extend(ys)
    errors = np.array(errors)
    errors_x = np.array(errors_x)
    errors_y = np.array(errors_y)
    px_x = np.array(px_x)
    px_y = np.array(px_y)

    if warn_for_missing:
        if len(missing_keys):
            print(f"In error calcs, did not find files for times {sorted(set(missing_keys))}")

    return CoordinateErrors(
        rmse=np.sqrt(np.mean(np.square(errors))),
        errors=errors.astype('float32'),
        px_x=px_x.astype('float32'),
        px_y=px_y.astype('float32'),
        errors_x=errors_x.astype('float32'),
        errors_y=errors_y.astype('float32'),
    )


def error_time_series(found_stars, inputs):
    timestamps = []
    rmses = []
    for input in inputs:
        timestamps.append(input.timestamp)
        errors = series_errors(found_stars, [input], warn_for_missing=False)
        rmses.append(errors.rmse)
    return timestamps, rmses


@dataclass
class CoordinateErrors:
    rmse: float
    errors: np.ndarray
    px_x: np.ndarray
    px_y: np.ndarray
    errors_x: np.ndarray
    errors_y: np.ndarray

    def __str__(self):
        return f"CoordinateErrors object with rmse={self.rmse}"

    def __repr__(self):
        return self.__str__()

    def plot_errors(self, vmax=0.8):
        plt.figure(figsize=(13, 9))
        plt.subplot(231)
        stat, _, _, _ = scipy.stats.binned_statistic_2d(
            self.px_y, self.px_x, self.errors, 'median', 100,
            expand_binnumbers=True,
            range=((0, 1024), (0, 960)))
        plt.imshow(stat, origin='lower', extent=(0, 1024, 0, 960), vmin=0, vmax=vmax)
        plt.colorbar().set_label("Star location error (px)")

        plt.subplot(234)
        stat, _, _, _ = scipy.stats.binned_statistic_2d(
            self.px_y, self.px_x, self.errors, 'std', 100,
            expand_binnumbers=True,
            range=((0, 1024), (0, 960)))
        plt.imshow(stat, origin='lower', extent=(0, 1024, 0, 960), vmin=0, vmax=vmax)
        plt.colorbar().set_label("Std. dev. of star location error (px)")

        plt.subplot(232)
        directions = np.arctan2(self.errors_y, self.errors_x)
        stat, _, _, _ = scipy.stats.binned_statistic_2d(
            self.px_y, self.px_x, directions, 'median', 100,
            expand_binnumbers=True,
            range=((0, 1024), (0, 960)))
        plt.imshow(stat, origin='lower', extent=(0, 1024, 0, 960), vmin=-np.pi, vmax=np.pi)
        plt.colorbar().set_label("Star location error direction (radians)")

        plt.subplot(233)
        err_x, r, c, _ = scipy.stats.binned_statistic_2d(
            self.px_y, self.px_x, self.errors_x, 'median', 30,
            expand_binnumbers=True,
            range=((0, 1024), (0, 960)))
        err_y, _, _, _ = scipy.stats.binned_statistic_2d(
            self.px_y, self.px_x, self.errors_y, 'median', 30,
            expand_binnumbers=True,
            range=((0, 1024), (0, 960)))
        XX, YY = np.meshgrid(c[:-1], r[:-1])
        # mags += .4
        plt.quiver(XX, YY, err_x, err_y, headwidth=5)
        plt.gca().set_aspect('equal')

        plt.suptitle(f"RMSE: {self.rmse:.5f}")
