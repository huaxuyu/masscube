# Author: Huaxu Yu

# A module to read and process the raw MS data

# imports
from pyteomics import mzml, mzxml
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from .params import Params, find_ms_info
from .feature_detection import detect_features, segment_feature
from .mzpkl import convert_MSData_to_mzpkl, read_mzpkl_to_MSData


"""
Classes
------------------------------------------------------------------------------------------------------------------------
"""

class MSData:
    """
    A class that models a single file (mzML, mzXML, mzjson
    or compressed mzjson file) and processes the raw data.
    """

    def __init__(self):

        self.scans = []                 # A list of Scan objects for mass spectra
        self.ms1_idx = []               # Scan indexes of MS1 spectra
        self.ms1_time_arr = []          # Time of MS1 scans
        self.ms2_idx = []               # Scan indexes of MS2 spectra
        self.params = None              # A Params object that contains all parameters
        self.base_peak_arr = []         # Base peak chromatogram, [[m/z, intensity], ...]
        self.features = []              # A list of features
        self.feature_mz_arr = None      # m/z of all ROIs
        self.feature_rt_arr = None      # Retention time of all ROIs


    def read_raw_data(self, file_name, params=None, scan_levels=[1,2], centroid_mz_tol=0.005, 
                      ms1_abs_int_tol=None, ms2_abs_int_tol=None, ms2_rel_int_tol=0.01, precursor_mz_offset=2):
        """
        Read raw data (mzML, mzXML, mzjson or compressed mzjson). Parsing of the mzML and mzXML files
        is performed using pyteomics.

        Parameters
        ----------
        file_name: str
            Name of the raw data file. Valid extensions are mzML, mzXML, mzjson and gz.
        params: Params object
            A Params object that contains the parameters.
        scan_levels: list
            MS levels to read, default is [1, 2] for MS1 and MS2 respectively.
        centroid_mz_tol: float
            m/z tolerance for centroiding. Set to None to disable centroiding.
        ms1_abs_int_tol: int
            Abolute intensity tolerance for MS1 scans.
        ms2_abs_int_tol: int
            Abolute intensity tolerance for MS2 scans. The final tolerance is the maximum of
            ms2_abs_int_tol and base signal intensity * ms2_rel_int_tol.
        ms2_rel_int_tol: float
            Relative intensity cutoff to the base signal for MS2 scans. Default is 0.01.
            Set to zero to disable this filter.
        precursor_mz_offset: float
            To remove the precursor ion from MS2 scan. The m/z upper limit of signals 
            in MS2 scans is calculated as precursor_mz - precursor_mz_offset.
        """

        if file_name.lower().endswith(".mzpkl"):
            self.params = Params()
            read_mzpkl_to_MSData(self, file_name)
            return None

        # priority for parameter setting:
        # 1. a Params object
        # 2. parameters provided through the function
        # 3. default parameters

        if params is None:
            params = Params()
            ms_type, ion_mode, centroid = find_ms_info(file_name)
            params.scan_levels = scan_levels
            params.centroid_mz_tol = centroid_mz_tol
            params.ms1_abs_int_tol = ms1_abs_int_tol
            params.ms2_abs_int_tol = ms2_abs_int_tol
            params.ms2_rel_int_tol = ms2_rel_int_tol
            params.precursor_mz_offset = precursor_mz_offset
            params.ms_type = ms_type
            params.ion_mode = ion_mode
            params.is_centroid = centroid
        
        # set intensity tolerance for MS1 scans if not provided
        if params.ms1_abs_int_tol is None:
            # 30000 for orbitrap, 1000 for other types
            if ms_type == "orbitrap":
                params.ms1_abs_int_tol = 30000
            else:
                params.ms1_abs_int_tol = 1000
        if params.ms2_abs_int_tol is None:
            if ms_type == "orbitrap":
                params.ms2_abs_int_tol = 10000
            else:
                params.ms2_abs_int_tol = 500

        self.params = params

        # for file name
        self.params.file_path = file_name
        base_name = os.path.basename(file_name)
        self.params.file_name = base_name.split(".")[0]
        
        if os.path.isfile(file_name):
            if base_name.lower().endswith(".mzml"):
                with mzml.MzML(file_name) as reader:
                    self.extract_scan_mzml(scans=reader)
                    self.params.file_format = "mzml"
            elif base_name.lower().endswith(".mzxml"):
                with mzxml.MzXML(file_name) as reader:
                    self.extract_scan_mzxml(scans=reader)
                    self.params.file_format = "mzxml"
            else:
                raise ValueError("Unsupported raw data format. " +
                                 "Raw data must be mzML, mzXML, mzjson or mzjson.gz.")
        else:
            print("File {} does not exist.".format(file_name))


    def extract_scan_mzml(self, scans):
        """
        Function to extract all scans and convert them to Scan objects.

        Parameters
        ----------
        scans: iteratable object from pyteomics mzml.MzML
            An iteratable object that contains all MS1 and MS2 scans.
        """

        if self.params is None:
            raise ValueError("Please set the parameters before extracting scans.")

        time_unit = scans[0]['scanList']['scan'][0]['scan start time'].unit_info

        for idx, spec in enumerate(scans):
            
            # get time
            if "scan start time" in spec['scanList']['scan'][0]:
                scan_time = spec['scanList']['scan'][0]['scan start time']
            elif "scan time" in spec['scanList']['scan'][0]:
                scan_time = spec['scanList']['scan'][0]['scan time']   # not a standard format

            if time_unit == 'second':
                scan_time /= 60     # convert to minute

            # get level of mass spectrum
            level = spec['ms level']

            # skip scans not in the defined scan levels or outside the defined retention time range
            if (level not in self.params.scan_levels) or (scan_time < self.params.rt_lower_limit) or (scan_time > self.params.rt_upper_limit):
                self.scans.append(Scan(level=level, id=idx, scan_time=scan_time, signals=None, precursor_mz=None))
                continue

            signals = np.array([spec['m/z array'], spec['intensity array']], dtype=np.float32).T
            
            if level == 2:
                precursor_mz = spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z']
            else:
                precursor_mz = None
            
            self.scans.append(_preprocess_signals_to_scan(level=level, id=idx, scan_time=scan_time, signals=signals, 
                                                          params=self.params, precursor_mz=precursor_mz))
        
        self.ms1_idx = np.array([s.id for s in self.scans if s.level == 1 and s.signals is not None and len(s.signals) > 0])
        self.ms2_idx = np.array([s.id for s in self.scans if s.level == 2 and s.signals is not None and len(s.signals) > 0])
        self.ms1_time_arr = np.array([self.scans[i].time for i in self.ms1_idx])
        self.base_peak_arr = np.array([self.scans[i].signals[np.argmax(self.scans[i].signals[:, 1])] for i in self.ms1_idx])


    def extract_scan_mzxml(self, scans):
        """
        Function to extract all scans and convert them to Scan objects.

        Parameters
        ----------
        scans: iteratable object from pyteomics mzxml.MzXML
            An iteratable object that contains all MS1 and MS2 scans.
        """

        if self.params is None:
            raise ValueError("Please set the parameters before extracting scans.")
        
        time_unit = scans[0]["retentionTime"].unit_info

        for idx, spec in enumerate(scans):

            # get time
            scan_time = spec["retentionTime"]

            if time_unit == 'second':
                scan_time = scan_time / 60  # convert to minute

            # get level of mass spectrum
            level = spec['msLevel']

            # skip scans not in the defined scan levels or outside the defined retention time range
            if (level not in self.params.scan_levels) or (scan_time < self.params.rt_lower_limit) or (scan_time > self.params.rt_upper_limit):
                self.scans.append(Scan(level=level, id=idx, scan_time=scan_time, signals=None, precursor_mz=None))
                continue

            signals = np.array([spec['m/z array'], spec['intensity array']], dtype=np.float32).T
            
            if level == 2:
                precursor_mz = spec['precursorMz'][0]['precursorMz']
            else:
                precursor_mz = None
            
            self.scans.append(_preprocess_signals_to_scan(level=level, id=idx, scan_time=scan_time, signals=signals,
                                                          params=self.params, precursor_mz=precursor_mz))
        
        self.ms1_idx = np.array([s.id for s in self.scans if s.level == 1 and s.signals is not None and len(s.signals) > 0])
        self.ms2_idx = np.array([s.id for s in self.scans if s.level == 2 and s.signals is not None and len(s.signals) > 0])
        self.ms1_time_arr = np.array([self.scans[i].time for i in self.ms1_idx])
        self.base_peak_arr = np.array([self.scans[i].signals[np.argmax(self.scans[i].signals[:, 1])] for i in self.ms1_idx])


    def drop_ms1_ions_by_intensity(self, int_tol):
        """
        Function to drop ions in all MS1 scans by intensity threshold.

        Parameters
        ----------
        int_tol: int
            Abolute intensity tolerance.
        """

        for idx in self.ms1_idx:
            self.scans[idx].signals = self.scans[idx].signals[self.scans[idx].signals[:, 1] > int_tol]

    """
    For data processing including feature detection, feature segmentation, feature summarization
    --------------------------------------------------------------------------------------------
    """

    def detect_features(self):
        """
        Run feature detection. Parameters are specified in self.params (Params object).
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

        distance = np.clip(0.05 / np.mean(np.diff(self.ms1_time_arr)), 1, 5)

        for _ in range(iteration):
            self.features = [segment_feature(feature, peak_height_tol=self.params.ms1_abs_int_tol*3, distance=distance) for feature in self.features]
            # flatten the list
            self.features = [item for sublist in self.features for item in sublist]


    def summarize_features(self, cal_g_score=True, cal_a_score=True):
        """
        Function to process features to calculate the summary statistics.

        Parameters
        ----------
        cal_g_score: bool
            Whether to calculate Gaussian similarity.
        cal_a_score: bool
            Whether to calculate asymmetry factor.
        """      

        for feature in self.features:
            feature.summarize(g_score=cal_g_score, a_score=cal_a_score)

        # sort features by m/z
        self.features.sort(key=lambda x: x.mz)

        # index the features
        for idx in range(len(self.features)):
            self.features[idx].id = idx

        # extract mz and rt of all features for further use (feature grouping)
        self.feature_mz_arr = np.array([feature.mz for feature in self.features])
        self.feature_rt_arr = np.array([feature.rt for feature in self.features])

        # allocate ms2 to features
        self.allocate_ms2_to_features()

        # find best ms2 for each feature
        for feature in self.features:
            if len(feature.ms2_seq) > 0:
                feature.ms2 = find_best_ms2(feature.ms2_seq)


    def allocate_ms2_to_features(self, mz_tol=0.015):
        """
        Function to allocate MS2 scans to ROIs.

        Parameters
        ----------
        mz_tol: float
            m/z tolerance to match the precursor m/z of MS2 scans to features.
        """

        for i in self.ms2_idx:
            if len(self.scans[i].signals) == 0:
                continue
            idx = np.where(np.abs(self.feature_mz_arr - self.scans[i].precursor_mz) < mz_tol)[0]
            matched_features = []
            for j in idx:
                if self.features[j].rt_seq[0] < self.scans[i].time < self.features[j].rt_seq[-1]:
                    matched_features.append(self.features[j])
            if len(matched_features) == 1:
                matched_features[0].ms2_seq.append(self.scans[i])
            elif len(matched_features) > 1:
                # assign ms2 to the feature with the highest peak height
                matched_features[np.argmax([feature.peak_height for feature in matched_features])].ms2_seq.append(self.scans[i])


    def drop_features_without_ms2(self):
        """
        Function to drop features without MS2 scans.
        """

        self.features = [feature for feature in self.features if len(feature.ms2_seq) > 0]


    def drop_features_by_length(self, length=5):
        """
        Function to drop features by length (number of non-zero scans).
        """

        self.features = [feature for feature in self.features if feature.length >= length]


    def drop_isotope_features(self):
        """
        Function to drop features annotated as isotopes.
        """

        self.features = [feature for feature in self.features if not feature.is_isotope]
    

    def drop_in_source_fragment_features(self):
        """
        Function to discard in-source fragments.
        """

        self.features = [feature for feature in self.features if not feature.is_in_source_fragment]
    

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
            plt.text(self.ms1_time_arr[0], np.max(self.base_peak_arr[:,1])*0.9, self.params.file_name, fontsize=12, color="gray")

        if output_dir is not None:
            plt.savefig(output_dir, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


    def output_single_file(self, output_path=None):
        """
        Function to generate a report for features in csv format.

        Parameters
        ----------
        output_path: str
            User defined output path.
        """

        result = []

        for f in self.features:
            iso = ""
            ms2 = ""
            if f.ms2 is not None:
                for s in f.ms2.signals:
                    ms2 += str(np.round(s[0], decimals=4)) + ";" + str(np.round(s[1], decimals=0)) + "|"
                ms2 = ms2[:-1]

            temp = [f.feature_group_id, f.id, f.mz.__round__(4), f.rt.__round__(3), f.adduct_type, f.is_isotope, 
                    f.is_in_source_fragment, f.scan_idx, f.peak_area, f.peak_height, f.top_average, f.gaussian_similarity.__round__(2), 
                    f.noise_score.__round__(2), f.asymmetry_factor.__round__(2), f.charge_state, iso, f.rt_seq[0].__round__(3),
                    f.rt_seq[-1].__round__(3), f.length, ms2, f.matched_ms2, f.search_mode, f.annotation, f.formula, f.similarity,
                    f.matched_precursor_mz, f.matched_peak_number, f.smiles, f.inchikey]

            result.append(temp)

        # convert result to a pandas dataframe
        columns = [ "group_ID", "feature_ID", "m/z", "RT", "adduct", "is_isotope", "is_in_source_fragment", "scan_idx", "peak_area", "peak_height", "top_average",
                    "Gaussian_similarity", "noise_score", "asymmetry_factor", "charge", "isotopes", "RT_start", "RT_end", "total_scans",
                    "MS2", "matched_MS2", "search_mode", "annotation", "formula", "similarity", "matched_mz", "matched_peak_number", "SMILES", "InChIKey"]

        df = pd.DataFrame(result, columns=columns)
        
        # save the dataframe to csv file
        if output_path is None:
            df.to_csv(os.path.join(self.params.single_file_dir, self.params.file_name + ".txt"), index=False, sep="\t")
        if output_path is not None:
            df.to_csv(output_path, index=False, sep="\t")


    def get_eic_data(self, target_mz, target_rt=None, mz_tol=0.005, rt_tol=0.3, rt_range=None):
        """
        To get the EIC data of a target m/z.

        Parameters
        ----------
        target_mz: float
            Target m/z.
        target_rt: float
            Target retention time.
        mz_tol: float
            m/z tolerance.
        rt_tol: float
            Retention time tolerance.
        rt_range: list
            Retention time range [start, end]. The unit is minute.

        Returns
        -------
        eic_time_arr: numpy array
            Retention time of the EIC.
        eic_signals: numpy array
            m/z and intensity of the EIC organized as [[m/z, intensity], ...].
        eic_scan_idx_arr: numpy array
            Scan index of the EIC.
        """

        eic_time_arr = []
        eic_signals = []
        eic_scan_idx_arr = []

        if rt_range is None:
            if target_rt is None:
                rt_range = [0, np.inf]
            else:
                rt_range = [target_rt - rt_tol, target_rt + rt_tol]

        for i in self.ms1_idx:
            if rt_range[0] < self.scans[i].time < rt_range[1]:
                mz_diff = np.abs(self.scans[i].signals[:, 0] - target_mz)
                # if scan is not empty and at least one ion is matched
                if len(mz_diff) > 0 and np.min(mz_diff) < mz_tol:
                    eic_time_arr.append(self.scans[i].time)
                    eic_signals.append(self.scans[i].signals[np.argmin(mz_diff)])
                    eic_scan_idx_arr.append(i)
                else:
                    eic_time_arr.append(self.scans[i].time)
                    eic_signals.append([target_mz, 0])
                    eic_scan_idx_arr.append(i)

            if self.scans[i].time > rt_range[1]:
                break
        
        eic_time_arr = np.array(eic_time_arr, dtype=np.float32)
        eic_signals = np.array(eic_signals, dtype=np.float32)
        eic_scan_idx_arr = np.array(eic_scan_idx_arr, dtype=np.int32)

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

        for id in self.ms2_idx:
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

        return self.scans[self.ms1_idx[idx]]
    
    
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


    def plot_feature(self, feature_idx, mz_tol=0.005, rt_range=[0, np.inf], rt_window=None, output=False):
        """
        Function to plot EIC of a ROI.

        Parameters
        ----------
        feature_idx: int
            Index of the ROI.
        mz_tol: float
            m/z tolerance.
        rt_range: list
            Retention time range [start, end]. The unit is minute.
        rt_window: float
            Retention time window.
        output: str
            Output file name. If not specified, the plot will be shown.
        """

        if rt_window is not None:
            rt_range = [self.features[feature_idx].rt - rt_window, self.features[feature_idx].rt + rt_window]

        # get the eic data
        eic_rt, eic_int, _, eic_scan_idx = self.get_eic_data(self.features[feature_idx].mz, mz_tol=mz_tol, rt_range=rt_range)
        idx_start = np.where(eic_scan_idx == self.features[feature_idx].scan_idx_seq[0])[0][0]
        idx_end = np.where(eic_scan_idx == self.features[feature_idx].scan_idx_seq[-1])[0][0] + 1

        plt.figure(figsize=(7, 3))
        plt.rcParams['font.size'] = 14
        if 'Arial' in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = 'Arial'
        plt.plot(eic_rt, eic_int, linewidth=1, color="black")
        plt.fill_between(eic_rt[idx_start:idx_end], eic_int[idx_start:idx_end], color="black", alpha=0.4)
        plt.xlabel("Retention Time (min)", fontsize=18)
        plt.ylabel("Intensity", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if output:
            plt.savefig(output, dpi=300, bbox_inches="tight")
            plt.close()
            return None
        else:
            plt.show()
            return eic_rt[np.argmax(eic_int)], np.max(eic_int), eic_scan_idx[np.argmax(eic_int)]


    def convert_to_mzpkl(self):
        """
        Function to output all MS1 scans as an intermediate mzjson file for faster data loading, 
        if the file needs to be reloaded multiple times.

        Parameters
        ----------
        output_path: str
            Output path of the pickle file.
        """

        if self.params.tmp_file_dir is None:
            return None
        
        output_path = os.path.join(self.params.tmp_file_dir, self.params.file_name + ".mzpkl")
        convert_MSData_to_mzpkl(self, output_path)
    

    def read_mzpkl(self, data):
        """
        Function to read pickle file.

        Parameters
        ----------
        data: dict
            Dictionary from a pickle file.
        """
        
        self.ms1_idx = [i for i in range(len(data['time']))]
        self.ms1_time_arr = data['time']
        self.scans = [Scan(level=1, id=i, scan_time=self.ms1_time_arr[i], 
                           signals=data['signals'][i]) for i in range(len(data['time']))]

class Scan:
    """
    A class that represents a MS scan.
    """

    def __init__(self, level=None, id=None, scan_time=None, signals=None, precursor_mz=None):
        """
        Function to initiate MS1Scan by precursor mz,
        retention time.

        Parameters
        ----------
        level: int
            Level of MS scan.
        id: int
            Scan number ordered by time.
        rt: float
            Retention time.
        signals: numpy array
            MS signals for a scan as 2D numpy array in float32, organized as [[m/z, intensity], ...].
        precursor_mz: float
            Precursor m/z for MS2 scan only.
        """

        self.level = level                  # level of mass spectrum
        self.id = id                        # scan number ordered by time
        self.time = scan_time               # scan time in minute
        self.signals = signals              # MS signals for a scan as 2D numpy array in float32, organized as [[m/z, intensity], ...]
        self.precursor_mz = precursor_mz    # for MS2 only


    def add_signals(self, signals, precursor_mz=None):
        """
        Function to add peaks and precursor m/z (if applicable) to a scan.

        Parameters
        ----------
        signals: numpy array
            Peaks data, float32.
        precursor_mz: float
            Precursor m/z.
        """

        self.signals = signals
        if precursor_mz is not None:
            self.precursor_mz = precursor_mz


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
        plt.text(mz_range[0]+(mz_range[1]-mz_range[0])*0.4, max_int*1.1, 
                 "Time = {:.3f} min".format(self.time), fontsize=11)
        if self.level == 2:
            plt.text(mz_range[0]+(mz_range[1]-mz_range[0])*0.05, max_int*1.1, 
                     "Precursor m/z = {:.4f}".format(self.precursor_mz), fontsize=11)
        plt.show()

        if return_data:
            return signals


"""
Helper functions
------------------------------------------------------------------------------------------------------------------------
"""

def clean_signals(signals, mz_range=[0,np.inf], intensity_range=[0,np.inf]):
    """
    A function to clean signals in a mass spectrum by m/z and intensity range.

    Parameters
    ----------
    signals: numpy array
        MS signals for a scan as 2D numpy array in float32, organized as [[m/z, intensity], ...].
    mz_range: list
        m/z range [start, end].
    intensity_range: list
        Intensity range [start, end].

    Returns
    -------
    signals: numpy array
        Cleaned signals.
    """
    
    return signals[(signals[:, 0] > mz_range[0]) & (signals[:, 0] < mz_range[1]) & 
                   (signals[:, 1] > intensity_range[0]) & (signals[:, 1] < intensity_range[1])]


def centroid_signals(signals, mz_tol=0.005):
    """
    Function to centroid signals in a mass spectrum.

    Parameters
    ----------
    signals: numpy array
        MS signals for a scan as 2D numpy array in float32, organized as [[m/z, intensity], ...].
    mz_tol: float
        m/z tolerance for centroiding. Default is 0.005 Da.

    Returns
    -------
    signals: numpy array
        Centroided signals.
    """

    if mz_tol is None or mz_tol < 1e-6:
        return signals

    v = np.diff(signals[:, 0]) < mz_tol

    if np.sum(v) == 0:
        return signals
    
    # merge signals with m/z difference less than mz_tol
    idx_f = 0
    idx_e = 0
    new_signals = []
    for i in range(len(v)):
        if v[i]:
            idx_e = i + 1
            continue
        else:
            if idx_f == idx_e:
                new_signals.append(signals[idx_f])
            else:
                # weighted average of m/z and intensity
                new_signals.append([np.average(signals[idx_f:idx_e+1, 0], weights=signals[idx_f:idx_e+1, 1]), 
                                    np.sum(signals[idx_f:idx_e+1, 1])])
            idx_f = i + 1
            idx_e = i + 1
    
    if idx_f == idx_e:
        new_signals.append(signals[idx_f])
    else:
        new_signals.append([np.average(signals[idx_f:idx_e+1, 0], weights=signals[idx_f:idx_e+1, 1]),
                            np.sum(signals[idx_f:idx_e+1, 1])])

    return np.array(new_signals, dtype=np.float32)


def read_raw_file_to_obj(file_name, params=None, scan_levels=[1,2], centroid_mz_tol=0.005, 
                         ms1_abs_int_tol=None, ms2_abs_int_tol=None, ms2_rel_int_tol=0.01, 
                         precursor_mz_offset=2):
    """
    Read a raw file to a MSData object. It's a useful function for data visualization or 
    simple data analysis. See the MSData class for detailed parameter settings.

    Parameters
    ----------
    file_name: str
        Name of the raw data file. Valid extensions are mzML, mzXML, mzjson and gz.
    params: Params object
        A Params object that contains the parameters.
    scan_levels: list
        MS levels to read, default is [1, 2] for MS1 and MS2 respectively.
    centroid_mz_tol: float
        m/z tolerance for centroiding. Set to None to disable centroiding.
    ms1_abs_int_tol: int
        Abolute intensity tolerance for MS1 scans.
    ms2_abs_int_tol: int
        Abolute intensity tolerance for MS2 scans. The final tolerance is the maximum of
        ms2_abs_int_tol and base signal intensity * ms2_rel_int_tol.
    ms2_rel_int_tol: float
        Relative intensity cutoff to the base signal for MS2 scans. Default is 0.01.
        Set to zero to disable this filter.
    precursor_mz_offset: float
        To remove the precursor ion from MS2 scan. The m/z upper limit of signals 
        in MS2 scans is calculated as precursor_mz - precursor_mz_offset.

    Returns
    -------
    d : MSData object
        A MSData object.
    """

    # create a MSData object
    d = MSData()
    d.read_raw_data(file_name, params=params, scan_levels=scan_levels, centroid_mz_tol=centroid_mz_tol,
                    ms1_abs_int_tol=ms1_abs_int_tol, ms2_abs_int_tol=ms2_abs_int_tol, 
                    ms2_rel_int_tol=ms2_rel_int_tol, precursor_mz_offset=precursor_mz_offset)
    return d


def find_best_ms2(ms2_list):
    """
    Function to find the best MS2 spectrum for a list of MS2 spectra.
    """

    if len(ms2_list) > 0:
        total_ints = [np.sum(ms2.signals[:,1]) for ms2 in ms2_list]
        if np.max(total_ints) == 0:
            return None
        else:
            return ms2_list[max(range(len(total_ints)), key=total_ints.__getitem__)]
    else:
        return None


def _preprocess_signals_to_scan(level, id, scan_time, signals, params, precursor_mz=None):
    """
    Function to generate a Scan object from signals.
    """

    if len(signals) == 0:
        return Scan(level=level, id=id, scan_time=scan_time, signals=signals, precursor_mz=precursor_mz)

    if level == 1:
        signals = clean_signals(signals, intensity_range=[params.ms1_abs_int_tol, np.inf])

    elif level == 2:
        int_lower = max(params.ms2_abs_int_tol, np.max(signals[:, 1]) * params.ms2_rel_int_tol)
        signals = clean_signals(signals, mz_range=[0, precursor_mz - params.precursor_mz_offset],
                                intensity_range=[int_lower, np.inf])
    
    if params.centroid_mz_tol is not None:
        signals = centroid_signals(signals, mz_tol=params.centroid_mz_tol)
    
    return Scan(level=level, id=id, scan_time=scan_time, signals=signals, precursor_mz=precursor_mz)