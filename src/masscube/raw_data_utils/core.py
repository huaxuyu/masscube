# Author: Huaxu Yu

# Core data models and processing operations for raw MS data.

# imports

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from dataclasses import dataclass, field
from time import time

from ..params import Params
from ..feature_detection import detect_features, segment_feature
from ..utils_functions import centroid_signals


"""
Classes
------------------------------------------------------------------------------------------------------------------------
"""


@dataclass
class SingleFileMetadata:
    file_name: str = None
    file_path: str = None
    ms_type: str = None
    ion_mode: str = None
    is_centroid: bool = None
    acquisition_time: float = None
    file_format: str = None
    scan_time_unit: str = "minute"
    instrument_name: str = None
    has_ion_mobility: bool = False
    mobility_unit: str = None
    mz_calibration_applied: bool = None
    mobility_calibration_applied: bool = None
    mobility_pressure_compensated: bool = None
    created_at: float = field(default_factory=time)
    masscube_version: str = None


class MSData:
    """
    This class represents a single MS file.
    """

    def __init__(self):
        
        # 1. metadata
        self.metadata = default_single_file_metadata()

        # 2. parameters
        self.params = None              # a Params object that contains the parameters

        # 3. scans
        self.scans = []                 # a list of Scan objects for mass spectra
        self.ms1_idx_arr = []           # scan indexes of MS1 spectra
        self.ms2_idx_arr = []           # scan indexes of MS2 spectra
        self.ms1_time_arr = []          # acquisition time points for MS1 scans
        self.base_peak_arr = []         # base peak signals for MS1 scans

        # 4. features
        self.features = []              # a list of features
        self.feature_mz_arr = None      # m/z of all features
        self.feature_rt_arr = None      # retention time of all features
        
        # 5. processing status
        self.processing_status = default_processing_status()   # a dictionary to record the processing status of the raw data

        # 6. calibration offsets
        self.ms1_mz_calibration_offset = 0.0
        self.ms2_mz_calibration_offset = 0.0


    def update_metadata(self, updates: dict):
        """
        Function to update the metadata of the MSData object.

        Parameters
        ----------
        updates: dict
            A dictionary containing the metadata to be updated. The keys should be the metadata fields, and the values should be the new values for those fields.
        """

        for key, value in updates.items():
            if not hasattr(self.metadata, key):
                continue

            if key == "file_path" and value is not None:
                value = os.fspath(value)
            elif key == "file_name" and value is not None:
                value = _get_pure_file_name(value)

            setattr(self.metadata, key, value)

        # Keep file_name synchronized with file_path unless the caller
        # explicitly supplies a file name in the same update.
        if updates.get("file_path") is not None and "file_name" not in updates:
            self.metadata.file_name = _get_pure_file_name(self.metadata.file_path)


    def read_raw_data(
            self,
            file_path: str,
            params: Params,
    ) -> None:
        """
        Parse a supported raw data source into this object.

        Parameters
        ----------
        file_path: str
            Path to a centroid mzML file or Bruker TDF2 ``.d`` directory.
        params: Params object
            Parameters.
        """

        from .io import read_raw_data_into

        read_raw_data_into(self, file_path, params)


    def extract_scans_mzml(self, scans):
        """
        Function to extract scans and convert them to Scan objects.

        Parameters
        ----------
        scans: pyteomics.mzml.MzML
            An iteratable object that contains all MS1 and MS2 scans.
        """

        from .mzml import extract_scans_mzml

        extract_scans_mzml(self, scans)


    """
    Single file data processing functions
    --------------------------------------------------------------------------------------------
    """

    def detect_features(self):
        """
        Untargeted feature detection. Parameters are specified in self.params (Params object).
        """

        self.features = detect_features(self)


    def segment_features(self, iteration=2):
        """
        Function to segment features by edge detection. Parameters are specified in self.params 
        (Params object).

        Parameters
        ----------
        iteration: int
            Number of iterations to segment features. Increase this number may introduce more false positives.
        """

        for _ in range(iteration):
            self.features = [segment_feature(feature) for feature in self.features]
            self.features = [item for sublist in self.features for item in sublist]


    def finalize_features(self):
        """
        Finalize detected features by calculating summary statistics, 
        removing invalid features, sorting/indexing features, 
        caching m/z and RT arrays, and assigning MS2.
        """      

        for feature in self.features:
            feature.finalize()
            self._set_feature_peak_edges(feature)

        # sort features by m/z
        self.features.sort(key=lambda x: x.mz)

        # index the features
        for idx in range(len(self.features)):
            self.features[idx].id = idx + 1

        # cache m/z and RT arrays for features
        self.feature_mz_arr = np.array([feature.mz for feature in self.features])
        self.feature_rt_arr = np.array([feature.rt for feature in self.features])

        # allocate ms2 to features
        self.allocate_ms2_to_features()

        # find best ms2 for each feature and evaluate its quality
        for feature in self.features:
            if len(feature.ms2_seq) > 0:
                feature.ms2 = find_best_ms2(feature.ms2_seq)
                feature.ms2.precursor_ion_fraction = cal_precursor_ion_fraction(self, feature.ms2, feature.mz)


    def _set_feature_peak_edges(self, feature):
        """
        Set feature baseline edges to the adjacent MS1 scan times when available.
        """

        if len(feature.rt_seq) == 0 or len(feature.scan_idx_seq) == 0 or len(self.ms1_idx_arr) == 0:
            return

        left_edge = feature.rt_seq[0]
        right_edge = feature.rt_seq[-1]

        left_pos = np.searchsorted(self.ms1_idx_arr, feature.scan_idx_seq[0])
        if left_pos < len(self.ms1_idx_arr) and self.ms1_idx_arr[left_pos] == feature.scan_idx_seq[0]:
            if left_pos > 0:
                left_edge = self.scans[self.ms1_idx_arr[left_pos - 1]].time

        right_pos = np.searchsorted(self.ms1_idx_arr, feature.scan_idx_seq[-1])
        if right_pos < len(self.ms1_idx_arr) and self.ms1_idx_arr[right_pos] == feature.scan_idx_seq[-1]:
            if right_pos + 1 < len(self.ms1_idx_arr):
                right_edge = self.scans[self.ms1_idx_arr[right_pos + 1]].time

        feature.peak_edges = (left_edge, right_edge)


    def allocate_ms2_to_features(self, mz_tol: float = 0.1) -> None:
        """
        Assign MS2 scans to features by precursor m/z and retention time.

        The deliberately broad default m/z window tolerates raw files with
        inaccurate precursor-ion m/z metadata. Candidate features within that
        window are disambiguated by peak height.
        
        If multiple features are matched, the MS2 scan is assigned to the
        feature with the highest peak height. Each MS2 scan is assigned to at
        most one feature.

        Parameters
        ----------
        mz_tol : float, default=0.1
            m/z tolerance for matching MS2 precursor m/z to feature m/z.
        """

        if len(self.features) == 0 or len(self.ms2_idx_arr) == 0:
            return

        feature_mz_arr = np.asarray(self.feature_mz_arr, dtype=np.float64)

        sorted_idx = np.argsort(feature_mz_arr)
        sorted_feature_mz_arr = feature_mz_arr[sorted_idx]

        feature_rt_start_arr = np.array(
            [
                feature.peak_edges[0]
                if feature.peak_edges is not None
                else feature.rt_seq[0]
                for feature in self.features
            ],
            dtype=np.float32,
        )
        feature_rt_end_arr = np.array(
            [
                feature.peak_edges[1]
                if feature.peak_edges is not None
                else feature.rt_seq[-1]
                for feature in self.features
            ],
            dtype=np.float32,
        )
        feature_height_arr = np.array([feature.peak_height for feature in self.features], dtype=np.float32)

        for scan_idx in self.ms2_idx_arr:
            ms2 = self.scans[scan_idx]

            left = np.searchsorted(sorted_feature_mz_arr, ms2.precursor_mz - mz_tol, side="left")
            right = np.searchsorted(sorted_feature_mz_arr, ms2.precursor_mz + mz_tol, side="right")
            
            if left == right:
                continue

            candidate_idx = sorted_idx[left:right]
            
            mz_mask = np.abs(feature_mz_arr[candidate_idx] - ms2.precursor_mz) < mz_tol
            rt_mask = (
                (feature_rt_start_arr[candidate_idx] <= ms2.time)
                & (ms2.time <= feature_rt_end_arr[candidate_idx])
            )
            matched_idx = candidate_idx[mz_mask & rt_mask]
            if len(matched_idx) == 0:
                continue

            best_idx = matched_idx[np.argmax(feature_height_arr[matched_idx])]
            ms2.ms2_allocated = True
            self.features[best_idx].ms2_seq.append(ms2)
    

    """
    For data visualization and output
    --------------------------------------------------------------------------------------------
    """

    def plot_bpc(self, time_range=None, label_name=True, output_dir=None):
        """
        Function to plot base peak chromatogram.

        Parameters
        ----------
        time_range: list
            Time range [start, end] to plot the BPC. The unit is minute.
        label_name: bool
            Whether to show the file name on the plot.
        output_dir: str
            Output directory of the plot. If specified, the plot will be saved to the directory.
            If None, the plot will be shown.
        """

        plt.figure(figsize=(10, 3))
        plt.rcParams['font.size'] = 14
        if 'Arial' in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = 'Arial'
        plt.xlabel("Retention Time (min)", fontsize=18)
        plt.ylabel("Intensity", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if time_range is None:
            plt.plot(self.ms1_time_arr, self.base_peak_arr[:, 1], linewidth=1, color="black")
        else:
            v = (self.ms1_time_arr > time_range[0]) & (self.ms1_time_arr < time_range[1])
            plt.plot(self.ms1_time_arr[v], self.base_peak_arr[v, 1], linewidth=1, color="black")

        if label_name:
            plt.text(self.ms1_time_arr[0], np.max(self.base_peak_arr[:,1])*0.9, self.metadata.file_name, fontsize=12, color="gray")

        if output_dir is not None:
            plt.savefig(output_dir, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


    def output_single_file(self, output_path=None, include_peak_shape=None):
        """
        Function to generate a report for features in csv format.

        Parameters
        ----------
        output_path: str
            User defined output path.
        include_peak_shape: bool or None
            Whether to serialize the full RT/intensity trace for every feature.
            The default is ``Params.output_peak_shape`` (False when absent).
            Alignment and gap filling do not require this column's payload;
            peak shapes can be reconstructed from the corresponding mcraw.
        """

        if include_peak_shape is None:
            include_peak_shape = bool(
                getattr(self.params, "output_peak_shape", False)
            )

        result = []

        for f in self.features:
            ms2 = ""
            iso = ""
            peak_shape = ""
            pif = None
            ms2_scan_id = None
            if f.ms2 is not None:
                for s in f.ms2.signals:
                    ms2 += str(np.round(s[0], decimals=4)) + ";" + str(np.round(s[1], decimals=0)) + "|"
                ms2 = ms2[:-1]
                pif = f.ms2.precursor_ion_fraction
                ms2_scan_id = f.ms2.raw_file_id
            if f.isotope_signals is not None:
                for s in f.isotope_signals:
                    iso += str(np.round(s[0], decimals=4)) + ";" + str(np.round(s[1], decimals=0)) + "|"
                iso = iso[:-1]
            if include_peak_shape and f.peak_shape is not None:
                time_range = [f.rt-1, f.rt+1]
                for p in f.peak_shape:
                    if time_range[0] < p[0] < time_range[1]:
                        peak_shape += str(np.round(p[0], decimals=3)) + ";" + str(np.round(p[1], decimals=0)) + "|"

            temp = [f.feature_group_id, f.id, f.mz.__round__(4), f.rt.__round__(3), f.adduct_type, f.is_isotope, 
                    f.is_in_source_fragment, f.scan_idx, f.peak_area, f.peak_height, f.top_average, f.gaussian_similarity.__round__(2), 
                    f.noise_score.__round__(2), f.asymmetry_factor.__round__(2), f.charge_state, iso, f.rt_seq[0].__round__(3),
                    f.rt_seq[-1].__round__(3), f.length, peak_shape, ms2, ms2_scan_id, pif, f.matched_ms2, f.search_mode, f.annotation, f.formula, f.similarity,
                    f.matched_precursor_mz, f.matched_peak_number, f.smiles, f.inchikey]

            result.append(temp)

        # convert result to a pandas dataframe
        columns = [ "group_ID", "feature_ID", "m/z", "RT", "adduct", "is_isotope", "is_in_source_fragment", "scan_idx", "peak_area", "peak_height", "top_average",
                    "Gaussian_similarity", "noise_score", "asymmetry_factor", "charge", "isotopes", "RT_start", "RT_end", "total_scans", "peak_shape",
                    "MS2", "MS2_scan_id", "precursor_ion_fraction", "matched_MS2", "search_mode", "annotation", "formula", "similarity", "matched_mz", "matched_peak_number", "SMILES", "InChIKey"]

        df = pd.DataFrame(result, columns=columns)
        
        # Project paths do not belong to Params. A caller that wants a file
        # supplies its destination explicitly; otherwise return the table.
        if output_path is not None:
            df.to_csv(output_path, index=False, sep="\t")
        return df


    def get_eic_data(self, target_mz, target_rt=None, mz_tol=0.005, rt_tol=0.3, rt_range=None):
        if rt_range is None:
            if target_rt is None:
                rt0, rt1 = 0.0, np.inf
            else:
                rt0, rt1 = target_rt - rt_tol, target_rt + rt_tol
        else:
            rt0, rt1 = rt_range[0], rt_range[1]

        times = self.ms1_time_arr          # (n_ms1,) float32/float64, sorted
        ms1_idx = self.ms1_idx_arr         # (n_ms1,) int32, aligned to times
        scans = self.scans                 # local bind

        # RT window -> contiguous slice via binary search (much faster than mask+where)
        left = np.searchsorted(times, rt0, side="left")
        right = np.searchsorted(times, rt1, side="right")
        if right <= left:
            return _EMPTY_F32, _EMPTY_SIG, _EMPTY_I32

        eic_time_arr = times[left:right]
        eic_scan_idx_arr = ms1_idx[left:right]
        n = eic_scan_idx_arr.size

        # allocate outputs as two 1D arrays (faster than (n,2) then column ops)
        eic_mz = np.full(n, np.nan, dtype=np.float32)
        eic_int = np.zeros(n, dtype=np.float32)

        mz0 = float(target_mz)
        lo = mz0 - mz_tol
        hi = mz0 + mz_tol

        for out_i, scan_i in enumerate(eic_scan_idx_arr):
            sig = scans[int(scan_i)].signals
            if sig is None or sig.shape[0] == 0:
                continue

            mzs = sig[:, 0]  # sorted ascending
            # find m/z window indices in O(log n_peaks)
            l = np.searchsorted(mzs, lo, side="left")
            r = np.searchsorted(mzs, hi, side="right")
            if r <= l:
                continue

            ints = sig[:, 1]
            j = l + int(np.argmax(ints[l:r]))
            eic_mz[out_i] = mzs[j]
            eic_int[out_i] = ints[j]

        eic_signals = np.column_stack((eic_mz, eic_int)).astype(np.float32, copy=False)
        return eic_time_arr, eic_signals, eic_scan_idx_arr
    

    def plot_eics(self, target_mz_arr, target_rt=None, mz_tol=0.005, rt_tol=0.3, rt_range=None,
                  output_file_name=None, show_target_rt=True, ylim: list=None, return_eic_data=False):
        """
        Function to plot multiple EICs in a single plot.

        Parameters
        ----------
        target_mz_arr: list
            A list of target m/z.
        target_rt: float
            Target retention time.
        mz_tol: float
            m/z tolerance.
        rt_tol: float
            Retention time tolerance.
        rt_range: list
            Retention time range [start, end]. The unit is minute.
        output_file_name: str
            Output file name. If not specified, the plot will be shown.
        show_target_rt: bool
            Whether to show the target retention time as a vertical line.
        ylim: list
            [min, max] of the y-axis.
        return_eic_data: bool   
            Whether to return the EIC data as a list of [eic_time_arr, eic_signals, eic_scan_idx].

        Returns
        -------
        eic_data: list
            A list of EIC data: [[eic_time_arr, eic_signals, eic_scan_idx], ...].
        """

        plt.figure(figsize=(10, 3))
        plt.rcParams['font.size'] = 14
        if 'Arial' in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = 'Arial'
        plt.xlabel("Retention Time (min)", fontsize=18)
        plt.ylabel("Intensity", fontsize=18)

        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        if target_rt is not None and show_target_rt:
            plt.axvline(x = target_rt, color = 'b', linestyle = '--', linewidth=1)
        
        if np.ndim(target_mz_arr) == 1:
            eic_data = []
            for target_mz in target_mz_arr:
                # get the eic data
                eic_time_arr, eic_signals, eic_scan_idx_arr = self.get_eic_data(target_mz, target_rt, mz_tol, rt_tol, rt_range)
                plt.plot(eic_time_arr, eic_signals[:, 1], linewidth=1)
                eic_data.append([eic_time_arr, eic_signals, eic_scan_idx_arr])
        elif np.ndim(target_mz_arr) == 0:
            eic_time_arr, eic_signals, eic_scan_idx_arr = self.get_eic_data(target_mz_arr, target_rt, mz_tol, rt_tol, rt_range)
            plt.plot(eic_time_arr, eic_signals[:, 1], linewidth=1, color="black")
            eic_data = [eic_time_arr, eic_signals, eic_scan_idx_arr]

        if output_file_name is not None:
            try:
                plt.savefig(output_file_name, dpi=300, bbox_inches="tight")
                plt.close()
            except:
                print("Invalid output path.")
        else:
            plt.show()

        if return_eic_data:
            return eic_data
        
    
    def find_ms2_by_mzrt(self, mz_target, rt_target, mz_tol=0.01, rt_tol=0.3, return_best=False):
        """
        Function to find MS2 scan by precursor m/z and retention time.

        Parameters
        ----------------------------------------------------------
        mz_target: float
            Precursor m/z.
        rt_target: float
            Retention time.
        mz_tol: float
            m/z tolerance.
        rt_tol: float
            Retention time tolerance.
        return_best: bool
            whether to return the best MS2 scan with the highest total intensity.
        """

        matched_ms2 = []

        for id in self.ms2_idx_arr:
            if abs(self.scans[id].time - rt_target) < rt_tol and abs(self.scans[id].precursor_mz - mz_target) < mz_tol:
                matched_ms2.append(self.scans[id])

        if return_best:
            if len(matched_ms2) > 1:
                total_ints = [np.sum(ms2.signals[:,1]) for ms2 in matched_ms2]
                return matched_ms2[np.argmax(total_ints)]
            elif len(matched_ms2) == 1:
                return matched_ms2[0]
            else:
                return None
        else:
            return matched_ms2
        

    def find_feature_by_mzrt(self, mz_target, rt_target=None, mz_tol=0.01, rt_tol=0.3):
        """
        Function to find feature by precursor m/z and retention time.

        Parameters
        ----------------------------------------------------------
        mz_target: float
            Precursor m/z.
        rt_target: float
            Retention time.
        mz_tol: float
            m/z tolerance.
        rt_tol: float
            Retention time tolerance.
        """

        self.feature_mz_arr = np.array([feature.mz for feature in self.features])
        self.feature_rt_arr = np.array([feature.rt for feature in self.features])

        if rt_target is None:
            tmp = np.abs(self.feature_mz_arr - mz_target) < mz_tol
            found_feature = [self.features[i] for i in np.where(tmp)[0]]
        else:
            tmp1 = np.abs(self.feature_mz_arr - mz_target) < mz_tol
            tmp2 = np.abs(self.feature_rt_arr - rt_target) < rt_tol
            tmp = np.logical_and(tmp1, tmp2)
            found_feature = [self.features[i] for i in np.where(tmp)[0]]
            
        return found_feature
    

    def find_ms1_scan_by_rt(self, rt_target):
        """
        Function to find the nearest n MS1 scan by retention time.

        Parameters
        ----------------------------------------------------------
        rt_target: float
            Retention time.
        """

        idx = np.argmin(np.abs(self.ms1_time_arr - rt_target))

        return self.scans[self.ms1_idx_arr[idx]]
    
    
    def correct_retention_time(self, f):
        """
        Function to correct retention time.

        Parameters
        ----------------------------------------------------------
        f: interp1d object
            A function to correct retention time.
        """

        all_rts = np.array([s.time for s in self.scans])
        all_rts = f(all_rts)
        for i in range(len(self.scans)):
            self.scans[i].time = all_rts[i]

        # EIC extraction uses this compact, sorted MS1 time axis.  Leaving it
        # unchanged would make gap filling search the uncorrected RT window
        # even though individual Scan.time values had been corrected.
        self.ms1_time_arr = np.asarray(
            [self.scans[int(index)].time for index in self.ms1_idx_arr],
            dtype=np.float32,
        )


    def convert_to_mcraw(self, output_path=None, overwrite=False):
        """Write this pre-feature-detection object as a MassCube raw cache."""

        from .mcraw import save_mcraw

        if output_path is None:
            return None
        return save_mcraw(self, output_path, overwrite=overwrite)


    def convert_to_mzpkl(self, output_path=None):
        """
        Legacy writer retained for old callers.

        New MassCube workflows no longer call this method; mcraw is the raw
        cache used by feature grouping and gap filling.

        Parameters
        ----------
        output_path: str
            Output path of the pickle file.
        """

        if output_path is None:
            return None

        from ..mzpkl import convert_MSData_to_mzpkl

        convert_MSData_to_mzpkl(self, output_path)
    
    
    def get_spectral_rate(self):
        """
        Function to calculate the spectral rate of the raw data.

        Returns
        -------
        spectral_rate: float
            Spectral rate of the raw data, in Hz.
        """

        diff = np.diff(self.ms1_time_arr)
        diff = np.mean(diff[diff > 0]) * 60

        return 1 / diff


class Scan:
    """
    This class represents a single scan in MS data.
    """

    def __init__(self, file_name=None, raw_file_id=None, level=None, scan_time=None, signals=None,
                 precursor_mz=None, isolation_window=None, precursor_ion_fraction=None):
        """
        Function to initiate MS1Scan by precursor mz,
        retention time.
        """

        self.file_name = None                   # source file name, if available
        self.raw_file_id = None                 # index in the raw file
        self.level = None                       # 1 for MS1, 2 for MS2
        self.raw_time = None                    # the raw scan time in minutes
        self.time = None                        # the calibrated or corrected scan time
        self.signals = None                     # MS signals for a scan as 2D numpy array in float32, organized as [[m/z, intensity], ...]
        
        # MS/MS only
        self.precursor_mz = None                # precursor m/z
        self.isolation_window = None            # isolation window
        self.precursor_ion_fraction = None      # precursor ion fraction
        self.ms2_allocated = False              # whether the MS2 scan has been allocated to a feature

        # derived information
        self.sum_intensity = None               # sum of all ion intensities in the scan

        # optional ion-mobility/TIMS coordinates, aligned row-for-row to signals
        self.inv_mobility = None                # intensity-weighted inverse reduced mobility (1/K0)
        self.inv_mobility_range = None          # observed [min, max] 1/K0 for each signal
        self.inv_mobility_unit = None
        self.mobility_pressure_compensated = None
        self.bruker_frame_id = None
        self.bruker_precursor_id = None
        self.bruker_parent_frame_id = None
        self.bruker_scan_begin = None
        self.bruker_scan_end = None
        self.tims_pressure = None
        self.collision_energy = None
        self.precursor_charge = None

        if any(value is not None for value in [file_name, raw_file_id, level, scan_time, signals,
                                               precursor_mz, isolation_window,
                                               precursor_ion_fraction]):
            self.add_scan_info(file_name=file_name, raw_file_id=raw_file_id, level=level, scan_time=scan_time,
                               signals=signals, precursor_mz=precursor_mz,
                               isolation_window=isolation_window,
                               precursor_ion_fraction=precursor_ion_fraction)
    

    def add_scan_info(self, 
                      file_name: str = None,
                      raw_file_id: int = None, 
                      level: int = None, 
                      scan_time: float = None, 
                      signals: np.ndarray = None, 
                      precursor_mz: float = None, 
                      isolation_window: float = None, 
                      precursor_ion_fraction: float = None
    ):
        """
        Function to add scan information.

        Parameters
        ----------
        file_name: str
            Source file name.
        raw_file_id: int
            Scan ID in the original, raw file.
        level: int
            Scan level (1 for MS1, 2 for MS2).
        scan_time: float
            Scan time in minutes.
        signals: np.ndarray
            MS signals for a scan as 2D numpy array in float32, organized as [[m/z, intensity], ...].
        precursor_mz: float
            Precursor m/z for MS2 only.
        isolation_window: float
            Isolation window for MS2 only.
        precursor_ion_fraction: float
            Precursor ion fraction for MS2 only.
        """

        if file_name is not None:
            self.file_name = file_name
        if raw_file_id is not None:
            self.raw_file_id = int(raw_file_id)
        if level is not None:
            self.level = int(level)
        if scan_time is not None:
            self.raw_time = float(scan_time)
            self.time = float(scan_time)
        if signals is not None:
            self.signals = signals
        if precursor_mz is not None:
            self.precursor_mz = float(precursor_mz)
        if isolation_window is not None:
            self.isolation_window = isolation_window
        if precursor_ion_fraction is not None:
            self.precursor_ion_fraction = precursor_ion_fraction
    

    def preprocess_signals(self, params):
        """
        Function to preprocess the scan signals by applying m/z and intensity filters.

        Parameters
        ----------
        params: Params object
            A Params object that contains the parameters for signal preprocessing.
        """

        if self.signals is None:
            self.signals = np.empty((0, 2), dtype=np.float32)

        if self.level == 1:
            self.subset_signals_by_mz_intensity(mz_range=[params.mz_lower_limit, params.mz_upper_limit], 
                                                          intensity_range=[params.ms1_abs_int_tol, np.inf])
        elif self.level == 2:
            if len(self.signals) == 0:
                # Detach an empty slice from a possible mcraw memmap backing
                # array before that mapping is released after preprocessing.
                self.signals = np.empty((0, 2), dtype=self.signals.dtype)
                self.sum_intensity = 0.0
                return None
            if params.precursor_mz_offset is None or self.precursor_mz is None:
                upper_mz_limit = params.mz_upper_limit
            else:
                upper_mz_limit = min(
                    params.mz_upper_limit,
                    self.precursor_mz - params.precursor_mz_offset,
                )
            int_lower = max(params.ms2_abs_int_tol, np.max(self.signals[:, 1]) * params.ms2_rel_int_tol)
            self.subset_signals_by_mz_intensity(mz_range=[0, upper_mz_limit], 
                                                intensity_range=[int_lower, np.inf])
        
        if params.centroid_mz_tol is not None:
            self._centroid_aligned_signals(params.centroid_mz_tol)

        self.sum_intensity = float(np.sum(self.signals[:, 1], dtype=np.float64))


    def subset_signals_by_mz_intensity(self, mz_range=[0, np.inf], intensity_range=[0, np.inf]):
        """
        Function to subset the scan signals by m/z and intensity range.

        Parameters
        ----------
        mz_range: list
            m/z range [start, end].
        intensity_range: list
            Intensity range [start, end].

        Returns
        -------
        signals: numpy array
            Subsetted scan signals.
        """

        if self.signals is None:
            return None

        selected = (
            (self.signals[:, 0] > mz_range[0])
            & (self.signals[:, 0] < mz_range[1])
            & (self.signals[:, 1] > intensity_range[0])
            & (self.signals[:, 1] < intensity_range[1])
        )
        self.signals = self.signals[selected]
        if self.inv_mobility is not None:
            if len(self.inv_mobility) != len(selected):
                raise ValueError("inv_mobility must align with scan.signals")
            self.inv_mobility = self.inv_mobility[selected]
        if self.inv_mobility_range is not None:
            if len(self.inv_mobility_range) != len(selected):
                raise ValueError("inv_mobility_range must align with scan.signals")
            self.inv_mobility_range = self.inv_mobility_range[selected]


    def _centroid_aligned_signals(self, mz_tol):
        """Centroid signals while keeping optional mobility arrays aligned."""

        if mz_tol is None or len(self.signals) <= 1:
            return
        if self.inv_mobility is None and self.inv_mobility_range is None:
            self.signals = centroid_signals(self.signals, mz_tol=mz_tol)
            return

        order = np.argsort(self.signals[:, 0], kind="stable")
        signals = self.signals[order]
        mobility = None if self.inv_mobility is None else np.asarray(self.inv_mobility)[order]
        mobility_range = (
            None
            if self.inv_mobility_range is None
            else np.asarray(self.inv_mobility_range)[order]
        )
        starts = np.r_[0, np.flatnonzero(np.diff(signals[:, 0]) >= mz_tol) + 1]
        intensity = np.add.reduceat(signals[:, 1], starts)
        mz = np.add.reduceat(signals[:, 0] * signals[:, 1], starts) / intensity
        self.signals = np.column_stack((mz, intensity)).astype(np.float32, copy=False)
        if mobility is not None:
            self.inv_mobility = (
                np.add.reduceat(mobility * signals[:, 1], starts) / intensity
            ).astype(mobility.dtype, copy=False)
        if mobility_range is not None:
            self.inv_mobility_range = np.column_stack(
                (
                    np.minimum.reduceat(mobility_range[:, 0], starts),
                    np.maximum.reduceat(mobility_range[:, 1], starts),
                )
            ).astype(mobility_range.dtype, copy=False)


    def plot_scan(self, mz_range=None, max_int=None, return_data=False):
        """
        Function to plot a scan.
        
        Parameters
        ----------
        mz_range: list
            m/z range [start, end].
        max_int: float
            Maximum intensity to plot.
        return_data: bool
            Whether to return the scan signals with restricted m/z range.

        Returns
        -------
        signals: numpy array
            Restricted scan signals.
        """

        signals = self.signals

        if len(signals) == 0:
            mz_range = [0, 1000]
            signals = np.array([[0, 0], [1000, 0]], dtype=np.float32)
            max_int = 1000
        
        else:
            if mz_range is None:
                mz_range = [np.min(signals[:, 0])-10, np.max(signals[:, 0])+10]
            else:
                signals = signals[(signals[:, 0] > mz_range[0]) & (signals[:, 0] < mz_range[1])]

            if max_int is None:
                max_int = np.max(signals[:, 1])
        
        # plot the scan
        plt.figure(figsize=(10, 3))
        plt.rcParams['font.size'] = 14
        if 'Arial' in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = 'Arial'
        plt.ylim(0, max_int*1.2)
        plt.xlim(mz_range[0], mz_range[1])
        plt.vlines(x = signals[:,0], ymin = 0, ymax = signals[:,1], color="black", linewidth=1.5)
        plt.hlines(y = 0, xmin = mz_range[0], xmax = mz_range[1], color="black", linewidth=1.5)
        plt.xlabel("m/z, Dalton", fontsize=18)
        plt.ylabel("Intensity", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.text(mz_range[0]+(mz_range[1]-mz_range[0])*0.35, max_int*1.1, 
                 "Time = {:.3f} min".format(self.time), fontsize=11)
        if self.level == 2:
            plt.text(mz_range[0]+(mz_range[1]-mz_range[0])*0.05, max_int*1.1, 
                     "Precursor m/z = {:.4f}".format(self.precursor_mz), fontsize=11)
            if self.precursor_ion_fraction is not None:
                plt.text(mz_range[0]+(mz_range[1]-mz_range[0])*0.55, max_int*1.1, 
                         "PIF = {:.2f}".format(self.precursor_ion_fraction), fontsize=11)
        if len(self.signals) == 0:
            plt.text(mz_range[0]+(mz_range[1]-mz_range[0])*0.4, max_int*0.5, "No signals", fontsize=14, color="red")
        plt.show()

        if return_data:
            return signals


"""
Helper functions
------------------------------------------------------------------------------------------------------------------------
"""

def finalize_scan_indexes(data: MSData) -> None:
    """Rebuild scan indexes, retention times, and the MS1 base-peak cache."""

    data.ms1_idx_arr = np.asarray(
        [index for index, scan in enumerate(data.scans) if scan.level == 1],
        dtype=np.int32,
    )
    data.ms2_idx_arr = np.asarray(
        [index for index, scan in enumerate(data.scans) if scan.level == 2],
        dtype=np.int32,
    )
    data.ms1_time_arr = np.asarray(
        [data.scans[int(index)].time for index in data.ms1_idx_arr],
        dtype=np.float32,
    )

    base_peaks = []
    for index in data.ms1_idx_arr:
        signals = data.scans[int(index)].signals
        if signals is None or len(signals) == 0:
            base_peaks.append([0.0, 0.0])
        else:
            base_peaks.append(signals[int(np.argmax(signals[:, 1]))])
    data.base_peak_arr = np.asarray(base_peaks, dtype=np.float32).reshape(-1, 2)


def preprocess_msdata(data: MSData, params: Params) -> MSData:
    """Create MassCube's parameter-specific analysis view in-place.

    mcraw contains all decoded MS1/MS2 scans before intensity filtering and
    centroid repair.  This function applies the same scan selection and signal
    preprocessing used by direct raw reading, while keeping TIMS mobility
    arrays aligned with their signals.
    """

    selected_scans = []
    scan_levels = set(int(level) for level in params.scan_levels)
    for scan in data.scans:
        scan_time = scan.raw_time if scan.raw_time is not None else scan.time
        if scan.level not in scan_levels:
            continue
        if not params.rt_lower_limit <= scan_time <= params.rt_upper_limit:
            continue
        scan.preprocess_signals(params)
        selected_scans.append(scan)

    data.scans = selected_scans
    data.params = params
    finalize_scan_indexes(data)
    return data


def find_best_ms2(ms2_list):
    """
    Function to find the best MS2 spectrum for a list of MS2 spectra.
    """

    if len(ms2_list) > 0:
        total_ints = [np.sum(ms2.signals[:,1]) for ms2 in ms2_list]
        if np.max(total_ints) == 0:
            return ms2_list[0]
        else:
            return ms2_list[max(range(len(total_ints)), key=total_ints.__getitem__)]
    else:
        return None


def cal_precursor_ion_fraction(d: MSData, ms2: Scan, mz: float=None) -> float:
    """
    Calculate the precursor ion fraction for an MS2 spectrum.

    Parameters
    ----------
    d : MSData
        The MSData object containing the raw data.
    ms2 : Scan
        The MS2 scan object.
    mz : float
        The m/z value of the feature that the MS2 was assigned to.

    Returns
    -------
    pif : float or None
        The precursor ion fraction. None is returned when required MS2 isolation-window
        metadata are missing.
    """

    if mz is None:
        mz = ms2.precursor_mz
    ms2_rt = ms2.time

    time_arr = d.ms1_time_arr
    if len(time_arr) == 0:
        return None
    
    # find the ms1 scan cloest to the ms2 scan
    idx = np.argmin(np.abs(time_arr - ms2_rt))
    ms1_scan = d.scans[d.ms1_idx_arr[idx]]

    s = ms1_scan.signals
    iso_window = ms2.isolation_window

    if iso_window is None or len(iso_window) != 2:
        return None
    s = s[(s[:,0] > mz - iso_window[0]) & (s[:,0] < mz + iso_window[1])]
    
    if len(s) == 0:
        return 0.0
    
    total_int = np.sum(s[:, 1])
    ion_int = s[np.argmin(np.abs(s[:,0] - mz)), 1]
    
    if total_int > 0:
        pif = ion_int / total_int
    else:
        pif = 0.0

    return pif


"""
Constants
----------------------------------------------------------------------------------------------------------------
"""

_EMPTY_F32 = np.empty(0, dtype=np.float32)
_EMPTY_I32 = np.empty(0, dtype=np.int32)
_EMPTY_SIG = np.empty((0, 2), dtype=np.float32)


def _get_pure_file_name(file_path):
    """Return the final path component without its last file extension."""

    return os.path.splitext(os.path.basename(os.fspath(file_path)))[0]


def default_single_file_metadata():
    """
    Create default metadata for one MSData object.
    """

    from .. import __version__

    return SingleFileMetadata(masscube_version=__version__)

def default_processing_status():
    """
    Create default processing status for one MSData object.
    """

    return {
        "ms1_calibrated": False,
        "ms2_calibrated": False,
        "rt_calibrated": False,
        "signal_normalized": False,
        "sample_normalized": False,
        "quality_flags": [],
    }


UNSUPPORTED_RAW_FORMAT_MESSAGE = (
    "Unsupported raw data format. MassCube supports centroid mzML files, "
    "Bruker TDF2 .d directories containing analysis.tdf and analysis.tdf_bin, "
    "and MassCube .mcraw cache directories. "
    "mzXML is not supported because required MS metadata are often incomplete. "
    "Please convert other raw formats to centroid mzML. "
    "If you must process mzXML data, please use MassCube version 1."
)
