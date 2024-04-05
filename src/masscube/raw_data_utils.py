# Author: Hauxu Yu

# A module to read and process the raw MS data
# Classes are defined in order to handle the data

# Import modules
from pyteomics import mzml, mzxml
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from .params import Params
from .peak_detect import find_rois, cut_roi


class MSData:
    """
    A class that models a single file (mzML or mzXML) and
    processes the raw data.

    Attributes
    ----------------------------------------------------------
    scans: list, a list of Scan objects
    ms1_rt_seq: numpy array, retention times of all MS1 scans
    bpc_int: numpy array, intensity of the BPC
    rois: list, a list of ROI objects
    rois_mz_seq: numpy array, m/z of all ROIs
    params: Params object, a Params object that contains the parameters
    """


    def __init__(self):
        """
        Function to initiate MSData.
        ----------------------------------------------------------
        """

        self.scans = []             # A list of MS scans
        self.ms1_rt_seq = []        # Retention times of all MS1 scans
        self.bpc_int = []           # Intensity of the BPC
        self.rois = []              # A list of ROIs
        self.params = None          # A Params object
        self.file_name = None       # File name of the raw data without extension
        self.roi_mz_seq = None      # m/z of all ROIs
        self.roi_rt_seq = None      # Retention time of all ROIs
        self.start_time = None      # Start acquisition time of the raw data


    def read_raw_data(self, file_name, params, centroid=True):
        """
        Function to read raw data to MS1 and MS2 (if available)
        (supported by pyteomics package).

        Parameters
        ----------------------------------------------------------
        file_name: str
            File name of raw MS data (mzML or mzXML).
        params: Params object
            A Params object that contains the parameters.
        """

        self.params = params

        if os.path.isfile(file_name):
            # get extension from file name
            ext = os.path.splitext(file_name)[1]

            if ext.lower() != ".mzml" and ext.lower() != ".mzxml":
                raise ValueError("Unsupported raw data format. Raw data must be in mzML or mzXML.")

            self.file_name = os.path.splitext(os.path.basename(file_name))[0]

            self.start_time = get_start_time(file_name)

            if ext.lower() == ".mzml":
                with mzml.MzML(file_name) as reader:
                    self.extract_scan_mzml(reader, centroid)
            elif ext.lower() == ".mzxml":
                with mzxml.MzXML(file_name) as reader:
                    self.extract_scan_mzxml(reader, centroid)
        else:
            print("File does not exist.")


    def extract_scan_mzml(self, spectra, centroid=True):
        """
        Function to extract all scans and convert them to Scan objects.

        Parameters
        ----------------------------------------------------------
        spectra: pyteomics object
            An iteratable object that contains all MS1 and MS2 scans.
        """

        idx = 0     # Scan number
        self.ms1_idx = []   # MS1 scan index
        self.ms2_idx = []   # MS2 scan index

        rt_unit = spectra[0]['scanList']['scan'][0]['scan start time'].unit_info

        # Iterate over all scans
        for spec in spectra:
            # Get the retention time and convert to minute
            try:
                rt = spec['scanList']['scan'][0]['scan start time']
            except:
                rt = spec['scanList']['scan'][0]['scan time']
            
            if rt_unit == 'second':
                rt = rt / 60

            # Check if the retention time is within the range
            if self.params.rt_range[0] < rt < self.params.rt_range[1]:
                if spec['ms level'] == 1:
                    temp_scan = Scan(level=1, scan=idx, rt=rt)
                    mz_array = spec['m/z array']
                    int_array = spec['intensity array']
                    if centroid:
                        mz_array, int_array = _centroid(mz_array, int_array)

                    temp_scan.add_info_by_level(mz_seq=mz_array, int_seq=int_array)
                    self.ms1_idx.append(idx)

                    # update base peak chromatogram
                    self.bpc_int.append(np.max(spec['intensity array']))
                    self.ms1_rt_seq.append(rt)

                elif spec['ms level'] == 2:
                    temp_scan = Scan(level=2, scan=idx, rt=rt)
                    precursor_mz = spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z']
                    peaks = np.array([spec['m/z array'], spec['intensity array']], dtype=np.float64).T
                    temp_scan.add_info_by_level(precursor_mz=precursor_mz, peaks=peaks)
                    _clean_ms2(temp_scan)
                    self.ms2_idx.append(idx)
                
                self.scans.append(temp_scan)
                idx += 1


    def extract_scan_mzxml(self, spectra, centroid=True):
        """
        Function to extract all scans and convert them to Scan objects.

        Parameters
        ----------------------------------------------------------
        spectra: pyteomics object
            An iteratable object that contains all MS1 and MS2 scans.
        """

        idx = 0     # Scan number
        self.ms1_idx = []   # MS1 scan index
        self.ms2_idx = []   # MS2 scan index

        rt_unit = spectra[0]['scanList']['scan'][0]['scan start time'].unit_info

        # Iterate over all scans
        for spec in spectra:
            # Get the retention time and convert to minute
            rt = spec["retentionTime"]    # retention time of mzXML is in minute

            if rt_unit == 'second':
                rt = rt / 60

            # Check if the retention time is within the range
            if self.params.rt_range[0] < rt < self.params.rt_range[1]:
                if spec['msLevel'] == 1:
                    temp_scan = Scan(level=1, scan=idx, rt=rt)
                    mz_array = spec['m/z array']
                    int_array = spec['intensity array']
                    if centroid:
                        mz_array, int_array = _centroid(mz_array, int_array)

                    temp_scan.add_info_by_level(mz_seq=mz_array, int_seq=int_array)
                    self.ms1_idx.append(idx)

                    # update base peak chromatogram
                    self.bpc_int.append(np.max(spec['intensity array']))
                    self.ms1_rt_seq.append(rt)

                elif spec['msLevel'] == 2:
                    temp_scan = Scan(level=2, scan=idx, rt=rt)
                    precursor_mz = spec['precursorMz'][0]['precursorMz']
                    peaks = np.array([spec['m/z array'], spec['intensity array']], dtype=np.float64).T
                    temp_scan.add_info_by_level(precursor_mz=precursor_mz, peaks=peaks)
                    _clean_ms2(temp_scan)
                    self.ms2_idx.append(idx)
                
                self.scans.append(temp_scan)
                idx += 1

    
    def drop_ion_by_int(self):
        """
        Function to drop ions by intensity.

        Parameters
        ----------------------------------------------------------
        tol: int
            Intensity tolerance.
        """

        for idx in self.ms1_idx:
            self.scans[idx].mz_seq = self.scans[idx].mz_seq[self.scans[idx].int_seq > self.params.int_tol]
            self.scans[idx].int_seq = self.scans[idx].int_seq[self.scans[idx].int_seq > self.params.int_tol]


    def find_rois(self):
        """
        Function to find ROI in MS1 scans.

        Parameters
        ----------------------------------------------------------
        params: Params object
            A Params object that contains the parameters.
        """

        self.rois = find_rois(self)
    

    def cut_rois(self):
        """
        Function to cut ROI into smaller pieces.
        """

        self.rois = [cut_roi(r, int_tol=self.params.int_tol) for r in self.rois]
        tmp = []
        for roi in self.rois:
            tmp.extend(roi)

        self.rois = tmp


    def summarize_roi(self, cal_gss=True):
        """
        Function to process ROIs.

        Parameters
        ----------------------------------------------------------
        params: Params object
            A Params object that contains the parameters.
        """      

        for roi in self.rois:
            roi.sum_roi(cal_gss=cal_gss)

        # sort rois by m/z
        self.rois.sort(key=lambda x: x.mz)

        # index the rois
        for idx in range(len(self.rois)):
            self.rois[idx].id = idx

        # extract mz and rt of all rois for further use (feature grouping)
        self.roi_mz_seq = np.array([roi.mz for roi in self.rois])
        self.roi_rt_seq = np.array([roi.rt for roi in self.rois])

        # allocate ms2 to rois
        for i in self.ms2_idx:
            idx = np.where(np.abs(self.roi_mz_seq - self.scans[i].precursor_mz) < self.params.mz_tol_ms2)[0]
            for j in idx:
                if self.rois[j].rt_seq[0] < self.scans[i].rt < self.rois[j].rt_seq[-1]:
                    self.rois[j].ms2_seq.append(self.scans[i])
                    break

        # find best ms2 for each roi
        for roi in self.rois:
            if len(roi.ms2_seq) > 0:
                roi.best_ms2 = find_best_ms2(roi.ms2_seq)
        
    
    def drop_rois_without_ms2(self):
        """
        Function to drop ROIs without MS2.
        """

        self.rois = [roi for roi in self.rois if len(roi.ms2_seq) > 0]
    

    def drop_rois_by_length(self, length=5):
        """
        Function to drop ROIs by length.
        """

        self.rois = [roi for roi in self.rois if roi.length >= length]
    

    def _discard_isotopes(self):
        """
        Function to discard isotopes.
        """

        self.rois = [roi for roi in self.rois if not roi.is_isotope]

        for idx in range(len(self.rois)):
            self.rois[idx].id = idx
    

    def plot_bpc(self, label_name=False, output=False):
        """
        Function to plot base peak chromatogram.

        Parameters
        ----------------------------------------------------------
        output: str
            Output file name. If not specified, the plot will be shown.
        """

        plt.figure(figsize=(10, 3))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Arial'
        plt.plot(self.ms1_rt_seq, self.bpc_int, linewidth=1, color="black")
        plt.xlabel("Retention Time (min)", fontsize=18, fontname='Arial')
        plt.ylabel("Intensity", fontsize=18, fontname='Arial')
        plt.xticks(fontsize=14, fontname='Arial')
        plt.yticks(fontsize=14, fontname='Arial')
        if label_name:
            plt.text(self.ms1_rt_seq[0], np.max(self.bpc_int)*0.9, self.file_name, fontsize=12, fontname='Arial', color="gray")

        if output:
            plt.savefig(output, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
        
    
    def output_single_file(self, user_defined_output_path=None):
        """
        Function to generate a report for rois in csv format.

        Parameters
        ----------------------------------------------------------
        user_defined_output_path: str
            User defined output path.
        """

        result = []

        for roi in self.rois:
            iso_dist = ""
            for i in range(len(roi.isotope_mz_seq)):
                iso_dist += str(np.round(roi.isotope_mz_seq[i], decimals=4)) + ";" + str(np.round(roi.isotope_int_seq[i], decimals=0)) + "|"
            iso_dist = iso_dist[:-1]

            ms2 = ""
            if roi.best_ms2 is not None:
                for i in range(len(roi.best_ms2.peaks)):
                    ms2 += str(np.round(roi.best_ms2.peaks[i, 0], decimals=4)) + ";" + str(np.round(roi.best_ms2.peaks[i, 1], decimals=0)) + "|"
                ms2 = ms2[:-1]

            temp = [roi.id, roi.mz.__round__(4), roi.rt.__round__(3), roi.length, roi.rt_seq[0],
                    roi.rt_seq[-1], roi.peak_area, roi.peak_height, roi.gaussian_similarity.__round__(2), 
                    roi.noise_level.__round__(2), roi.charge_state, roi.is_isotope, str(roi.isotope_id_seq)[1:-1], iso_dist,
                    roi.is_in_source_fragment, roi.isf_parent_roi_id, str(roi.isf_child_roi_id)[1:-1],
                    roi.adduct_type, roi.adduct_parent_roi_id, str(roi.adduct_child_roi_id)[1:-1],
                    ]
            
            temp.extend([ms2, roi.annotation, roi.formula, roi.similarity, roi.matched_peak_number, roi.smiles, roi.inchikey])

            result.append(temp)

        # convert result to a pandas dataframe
        columns = [ "ID", "m/z", "RT", "length", "RT_start", "RT_end", "peak_area", "peak_height",
                    "Gaussian_similarity", "noise_level", "charge", "is_isotope", "isotope_IDs", "isotopes", "is_in_source_fragment",
                    "ISF_parent_ID", "ISF_child_ID", "adduct", "adduct_base_ID", "adduct_other_ID"]
                    
        
        columns.extend(["MS2", "annotation", "formula", "similarity", "matched_peak_number", "SMILES", "InChIKey"])

        df = pd.DataFrame(result, columns=columns)
        
        # save the dataframe to csv file
        if user_defined_output_path:
            path = user_defined_output_path
        else:
            path = os.path.join(self.params.single_file_dir)
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(self.params.single_file_dir, self.file_name + ".csv")
        df.to_csv(path, index=False)
    

    def get_eic_data(self, target_mz, target_rt=None, mz_tol=0.005, rt_tol=0.3, rt_range=None):
        """
        To get the EIC data of a target m/z.

        Parameters
        ----------
        target_mz: float
            Target m/z.
        mz_tol: float
            m/z tolerance.
        target_rt: float
            Target retention time.
        rt_tol: float
            Retention time tolerance.

        Returns
        -------
        eic_rt: numpy array
            Retention time of the EIC.
        eic_int: numpy array
            Intensity of the EIC.
        eic_mz: numpy array
            m/z of the EIC.
        eic_scan_idx: numpy array
            Scan index of the EIC.
        """

        eic_rt = []
        eic_int = []
        eic_mz = []
        eic_scan_idx = []

        if target_rt is not None:
            rt_range = [target_rt - rt_tol, target_rt + rt_tol]
        elif rt_range is None:
            rt_range = [0, np.inf]

        for i in self.ms1_idx:
            if self.scans[i].rt > rt_range[0] and self.scans[i].rt < rt_range[1]:
                mz_diff = np.abs(self.scans[i].mz_seq - target_mz)
                if len(mz_diff)>0 and np.min(mz_diff) < mz_tol:
                    eic_rt.append(self.scans[i].rt)
                    eic_int.append(self.scans[i].int_seq[np.argmin(mz_diff)])
                    eic_mz.append(self.scans[i].mz_seq[np.argmin(mz_diff)])
                    eic_scan_idx.append(i)
                else:
                    eic_rt.append(self.scans[i].rt)
                    eic_int.append(0)
                    eic_mz.append(0)
                    eic_scan_idx.append(i)

            if self.scans[i].rt > rt_range[1]:
                break
        
        eic_rt = np.array(eic_rt)
        eic_int = np.array(eic_int)
        eic_mz = np.array(eic_mz)
        eic_scan_idx = np.array(eic_scan_idx)

        return eic_rt, eic_int, eic_mz, eic_scan_idx    
    

    def plot_eic(self, target_mz, target_rt=None, mz_tol=0.005, rt_tol=0.3, output=False, return_eic_data=False):
        """
        Function to plot EIC of a target m/z.

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
        output: str
            Output file name. If not specified, the plot will be shown.
        return_eic_data: bool   
            True: return the EIC data.
            False: do not return the EIC data.
        """

        # get the eic data
        eic_rt, eic_int, _, _ = self.get_eic_data(target_mz, target_rt, mz_tol, rt_tol)

        plt.figure(figsize=(10, 3))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Arial'
        plt.plot(eic_rt, eic_int, linewidth=1, color="black")
        plt.xlabel("Retention Time (min)", fontsize=18, fontname='Arial')
        plt.ylabel("Intensity", fontsize=18, fontname='Arial')
        plt.xticks(fontsize=14, fontname='Arial')
        plt.yticks(fontsize=14, fontname='Arial')
        if target_rt is not None:
            plt.axvline(x = target_rt, color = 'b', linestyle = '--', linewidth=1)

        if output:
            plt.savefig(output, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        if return_eic_data:
            return eic_rt, eic_int
        
    
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
            True: only return the best MS2 scan with the highest intensity.
            False: return all MS2 scans as a list.
        """

        matched_ms2 = []

        for id in self.ms2_idx:
            rt = self.scans[id].rt

            if rt < rt_target - rt_tol:
                continue
            
            mz = self.scans[id].precursor_mz
            
            if abs(mz - mz_target) < mz_tol and abs(rt - rt_target) < rt_tol:
                matched_ms2.append(self.scans[id])
        
            if rt > rt_target + rt_tol:
                break

        if return_best:
            if len(matched_ms2) > 1:
                total_ints = [np.sum(ms2.peaks[:,1]) for ms2 in matched_ms2]
                return matched_ms2[np.argmax(total_ints)]
            elif len(matched_ms2) == 1:
                return matched_ms2[0]
            else:
                return None
        else:
            return matched_ms2
        
    def find_roi_by_mzrt(self, mz_target, rt_target=None, mz_tol=0.01, rt_tol=0.3):
        """
        Function to find roi by precursor m/z and retention time.

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

        if rt_target is None:
            found_roi = [r for r in self.rois if abs(r.mz - mz_target) < mz_tol]
        else:
            found_roi = [r for r in self.rois if abs(r.mz - mz_target) < mz_tol and abs(r.rt - rt_target) < rt_tol]
            
        return found_roi   


    def plot_roi(self, roi_idx, mz_tol=0.005, rt_range=[0, np.inf], rt_window=None, output=False):
        """
        Function to plot EIC of a target m/z.
        """

        if rt_window is not None:
            rt_range = [self.rois[roi_idx].rt - rt_window, self.rois[roi_idx].rt + rt_window]

        # get the eic data
        eic_rt, eic_int, _, eic_scan_idx = self.get_eic_data(self.rois[roi_idx].mz, mz_tol=mz_tol, rt_range=rt_range)
        idx_start = np.where(eic_scan_idx == self.rois[roi_idx].scan_idx_seq[0])[0][0]
        idx_end = np.where(eic_scan_idx == self.rois[roi_idx].scan_idx_seq[-1])[0][0] + 1

        plt.figure(figsize=(7, 3))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Arial'
        plt.plot(eic_rt, eic_int, linewidth=1, color="black")
        plt.fill_between(eic_rt[idx_start:idx_end], eic_int[idx_start:idx_end], color="black", alpha=0.4)
        plt.xlabel("Retention Time (min)", fontsize=18, fontname='Arial')
        plt.ylabel("Intensity", fontsize=18, fontname='Arial')
        plt.xticks(fontsize=14, fontname='Arial')
        plt.yticks(fontsize=14, fontname='Arial')

        if output:
            plt.savefig(output, dpi=300, bbox_inches="tight")
            plt.close()
            return None
        else:
            plt.show()
            return eic_rt[np.argmax(eic_int)], np.max(eic_int), eic_scan_idx[np.argmax(eic_int)]


    def plot_all_rois(self, output_path, mz_tol=0.01, rt_range=[0, np.inf], rt_window=None):
        """
        Function to plot EIC of all ROIs.
        """

        if output_path[-1] != "/":
            output_path += "/"

        for idx, roi in enumerate(self.rois):

            if rt_window is not None:
                rt_range = [roi.rt_seq[0] - rt_window, roi.rt_seq[-1] + rt_window]

            # get the eic data
            eic_rt, eic_int, _, eic_scan_idx = self.get_eic_data(roi.mz, mz_tol=mz_tol, rt_range=rt_range)
            idx_start = np.where(eic_scan_idx == roi.scan_idx_seq[0])[0][0]
            idx_end = np.where(eic_scan_idx == roi.scan_idx_seq[-1])[0][0] + 1

            plt.figure(figsize=(9, 3))
            plt.rcParams['font.size'] = 14
            plt.rcParams['font.family'] = 'Arial'
            plt.plot(eic_rt, eic_int, linewidth=0.5, color="black")
            plt.fill_between(eic_rt[idx_start:idx_end], eic_int[idx_start:idx_end], color="black", alpha=0.2)
            plt.axvline(x = roi.rt, color = 'b', linestyle = '--', linewidth=1)
            plt.xlabel("Retention Time (min)", fontsize=18, fontname='Arial')
            plt.ylabel("Intensity", fontsize=18, fontname='Arial')
            plt.xticks(fontsize=14, fontname='Arial')
            plt.yticks(fontsize=14, fontname='Arial')
            plt.text(eic_rt[0], np.max(eic_int)*0.95, "m/z = {:.4f}".format(roi.mz), fontsize=12, fontname='Arial')
            plt.text(eic_rt[0] + (eic_rt[-1]-eic_rt[0])*0.6, np.max(eic_int)*0.95, self.file_name, fontsize=10, fontname='Arial', color="gray")

            file_name = output_path + "roi{}_".format(idx) + str(roi.mz.__round__(4)) + ".png"

            plt.savefig(file_name, dpi=300, bbox_inches="tight")
            plt.close()


class Scan:
    """
    A class that represents a MS scan.
    A MS1 spectrum has properties including:
        scan number, retention time, 
        m/z and intensities.
    A MS2 spectrum has properties including:
        scan number, retention time,
        precursor m/z, product m/z and intensities.
    """

    def __init__(self, level=None, scan=None, rt=None):
        """
        Function to initiate MS1Scan by precursor mz,
        retention time.

        Parameters
        ----------------------------------------------------------
        level: int
            Level of MS scan.
        scan: int
            Scan number.
        rt: float
            Retention time.
        """

        self.level = level
        self.scan = scan
        self.rt = rt

        # for MS1 scans:
        self.mz_seq = None
        self.int_seq = None

        # for MS2 scans:
        self.precursor_mz = None
        self.peaks = None
    

    def add_info_by_level(self, **kwargs):
        """
        Function to add scan information by level.
        """

        if self.level == 1:
            self.mz_seq = kwargs['mz_seq']
            self.int_seq = np.int64(kwargs['int_seq'])

        elif self.level == 2:
            self.precursor_mz = kwargs['precursor_mz']
            self.peaks = kwargs['peaks']


    def show_scan_info(self):
        """
        Function to print a scan's information.

        Parameters
        ----------------------------------------------------------
        scan: MS1Scan or MS2Scan object
            A MS1Scan or MS2Scan object.
        """

        print("Scan number: " + str(self.scan))
        print("Retention time: " + str(self.rt))

        if self.level == 1:
            print("m/z: " + str(np.around(self.mz_seq, decimals=4)))
            print("Intensity: " + str(np.around(self.int_seq, decimals=0)))

        elif self.level == 2:
            # keep 4 decimal places for m/z and 0 decimal place for intensity
            print("Precursor m/z: " + str(np.round(self.precursor_mz, decimals=4)))
            print(self.peaks)
    

    def plot_scan(self, mz_range=None, return_data=False):
        """
        Function to plot a scan.
        
        Parameters
        ----------------------------------------------------------
        """

        if self.level == 1:
            x = self.mz_seq
            y = self.int_seq
        elif self.level == 2:
            x = self.peaks[:, 0]
            y = self.peaks[:, 1]
        
        if mz_range is None:
            mz_range = [np.min(x)-10, np.max(x)+10]
        else:
            y = y[np.logical_and(x > mz_range[0], x < mz_range[1])]
            x = x[np.logical_and(x > mz_range[0], x < mz_range[1])]

        plt.figure(figsize=(10, 3))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Arial'
        # plt.scatter(eic_rt, eic_int, color="black")
        plt.vlines(x = x, ymin = 0, ymax = y, color="black", linewidth=1.5)
        plt.hlines(y = 0, xmin = mz_range[0], xmax = mz_range[1], color="black", linewidth=1.5)
        plt.xlabel("m/z, Dalton", fontsize=18, fontname='Arial')
        plt.ylabel("Intensity", fontsize=18, fontname='Arial')
        plt.xticks(fontsize=14, fontname='Arial')
        plt.yticks(fontsize=14, fontname='Arial')
        plt.show()

        if return_data:
            return x, y


def _clean_ms2(ms2, offset=2):
    """
    A function to clean MS/MS by
    1. Drop ions with m/z > (precursor_mz - offset)   
    2. Drop ions with intensity < 1% of the base peak intensity
    """
    
    if ms2.peaks.shape[0] > 0:
        ms2.peaks = ms2.peaks[ms2.peaks[:, 0] < ms2.precursor_mz - offset]
    if ms2.peaks.shape[0] > 0:
        ms2.peaks = ms2.peaks[ms2.peaks[:, 1] > 0.01 * np.max(ms2.peaks[:, 1])]


def _centroid(mz_seq, int_seq, centroiding_mz_tol=0.005):
    """
    Function to centroid the m/z and intensity sequences.

    Parameters
    ----------
    mz_seq: numpy array or list
        m/z sequence.
    int_seq: numpy array or list
        Intensity sequence.
    centroiding_mz_tol: float
        m/z tolerance for centroiding. Default is 0.005.
    """

    diff = np.diff(mz_seq)
    tmp = np.where(diff < centroiding_mz_tol)[0]

    if len(tmp) == 0:
        return mz_seq, int_seq
    
    mz_seq = list(mz_seq)
    int_seq = list(int_seq)
    for i in tmp[::-1]:
        mz_seq[i] = (mz_seq[i]*int_seq[i] + mz_seq[i+1]*int_seq[i+1]) / (int_seq[i] + int_seq[i+1])
        int_seq[i] += int_seq[i+1]
        mz_seq.pop(i+1)
        int_seq.pop(i+1)

    return np.array(mz_seq), int_seq


def read_raw_file_to_obj(file_name, params=None, int_tol=0.0, centroid=True, print_summary=False):
    """
    Read a raw file to a MSData object.
    It's a useful function for data visualization or brief data analysis.

    Parameters
    ----------
    file_name : str
        The file name of the raw file.
    params : Params object
        A Params object that contains the parameters.
    int_tol : float
        Intensity tolerance for dropping ions.
    centroid : bool
        True: centroid the raw data.
        False: do not centroid the raw data.
    print_summary : bool
        True: print the summary of the raw data.
        False: do not print the summary of the raw data.

    Returns
    -------
    d : MSData object
        A MSData object.
    """

    # create a MSData object
    d = MSData()

    # read raw data
    if params is None:
        params = Params()
    d.read_raw_data(file_name, params, centroid)

    if int_tol > 0:
        d.params.int_tol = int_tol
        d.drop_ion_by_int()
    
    if print_summary:
        print("Number of MS1 scans: " + str(len(d.ms1_idx)), "Number of MS2 scans: " + str(len(d.ms2_idx)))
    
    return d


def find_best_ms2(ms2_seq):
    """
    Function to find the best MS2 spectrum for a list of MS2 spectra.
    """

    if len(ms2_seq) > 0:
        total_ints = [np.sum(ms2.peaks[:,1]) for ms2 in ms2_seq]
        if np.max(total_ints) == 0:
            return None
        else:
            return ms2_seq[max(range(len(total_ints)), key=total_ints.__getitem__)]
    else:
        return None


def get_start_time(file_name):
    """
    Function to get the start time of the raw data.

    Parameters
    ----------
    file_name : str
        Absolute path of the raw data.
    """

    with open(file_name, "rb") as f:
        for l in f:
            l = str(l)
            if "startTimeStamp" in str(l):
                t = l.split("startTimeStamp")[1].split('"')[1]
                return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")