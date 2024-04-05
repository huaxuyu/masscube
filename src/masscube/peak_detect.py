# Author: Hauxu Yu

# A module for feature/peak detection

# Import modules
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
from .feature_evaluation import calculate_noise_level

from .feature_evaluation import calculate_gaussian_similarity

def find_rois(d):
    """
    A function to find the region of interest (ROI) in the MS data.

    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object that contains the MS data.
    """

    # A list to store the rois in progress
    rois = []
    # A list for the finally detected ROIs
    final_rois = []

    # Initiate a set of rois using the first MS1 scan
    fs = d.scans[d.ms1_idx[0]]    # The first scan

    for i in range(len(fs.int_seq)):
        roi = Roi(scan_idx=d.ms1_idx[0], rt=fs.rt, mz=fs.mz_seq[i], intensity=fs.int_seq[i])
        rois.append(roi)
    
    last_ms1_idx = d.ms1_idx[0]
    last_rt = fs.rt

    # Loop over all MS1 scans
    for ms1_idx in d.ms1_idx[1:]:

        s = d.scans[ms1_idx]    # The current MS1 scan

        visited_idx = []        # A list to store the visited indices of ions in the current MS1 scan
        visited_rois_idx = []   # A list to store the visited indices of rois

        # Loop over all current rois
        for i, roi in enumerate(rois):
            
            mz_diff = np.abs(roi.mz_seq[-1] - s.mz_seq)
            min_idx = np.argmin(mz_diff)

            if mz_diff[min_idx] < d.params.mz_tol_ms1:
                if min_idx not in visited_idx:
                    roi.extend_roi(scan_idx=ms1_idx, rt=s.rt, mz=s.mz_seq[min_idx], intensity=s.int_seq[min_idx])
                    roi.gap_counter = 0
                    visited_idx.append(min_idx)
                    visited_rois_idx.append(i)
            
        to_be_moved = []

        # Plus one to the gap counter of the rois that are not visited
        for i in range(len(rois)):
            if i not in visited_rois_idx:
                rois[i].extend_roi(scan_idx=ms1_idx, rt=s.rt, mz=rois[i].mz_seq[-1], intensity=0)
                rois[i].gap_counter = rois[i].gap_counter + 1
                if rois[i].gap_counter > d.params.roi_gap:
                    to_be_moved.append(i)
        
        # Move the rois that have not been visited for a long time to final_rois
        for i in to_be_moved[::-1]:
            final_rois.append(rois.pop(i))
        
        # Create new rois for the rest
        for i in range(len(s.int_seq)):
            if i not in visited_idx:
                # Add a zero before the new roi
                roi = Roi(scan_idx=last_ms1_idx, rt=last_rt, mz=s.mz_seq[i], intensity=0)
                roi.extend_roi(scan_idx=ms1_idx, rt=s.rt, mz=s.mz_seq[i], intensity=s.int_seq[i])
                rois.append(roi)
        last_ms1_idx = ms1_idx
        last_rt = s.rt
           
    # Move all rois to final_rois
    for roi in rois:
        final_rois.append(roi)

    return final_rois


def cut_roi(r, int_tol=1000):
    """
    Function to cut an ROI by providing the start and end positions.
    """

    r.int_seq = np.array(r.int_seq)
    r.noise_level = calculate_noise_level(r.int_seq)

    if r.noise_level > 0.4 or len(r.int_seq) < 10 or r.peak_height < 3*int_tol:
        return [r]

    ss = gaussian_filter1d(r.int_seq, sigma=1)
    peaks, _ = find_peaks(ss, prominence=np.max(ss)*0.01, distance=5)

    peaks = peaks[r.int_seq[peaks] > 2*int_tol]

    if len(peaks) < 2:
        return [r]

    if len(peaks) > 4:
        r.int_seq = np.array(r.int_seq)
        peaks = np.sort(peaks[np.argsort(r.int_seq[peaks])[-4:]])
    
    positions = [0]
    for i in range(len(peaks)-1):
        lowest_int = 1e10
        for j in range(peaks[i], peaks[i+1]):
            if r.int_seq[j] < lowest_int:
                lowest_int = r.int_seq[j]
                lowest_int_idx = j
        positions.append(lowest_int_idx)
    positions.append(len(r.int_seq)-1)
    
    cut_rois = []
    for i in range(len(positions)-1):
        tmp = deepcopy(r)
        tmp.subset_roi(start=positions[i], end=positions[i+1]+1)
        tmp.cut = True
        cut_rois.append(tmp)

    return cut_rois


class Roi:
    """
    A class to store a region of interest (ROI).
    """

    def __init__(self, scan_idx, rt, mz, intensity):
        """
        Function to initiate an ROI by providing the scan indices, 
        retention time, m/z and intensity.

        Parameters
        ----------------------------------------------------------
        scan_idx: int
            Scan index of the first ion in the ROI
        rt: float
            Retention time of the first ion in the ROI
        mz: float
            m/z value of the first ion in the ROI
        intensity: int
            Intensity of the first ion in the ROI
        """

        self.id = None
        self.scan_idx_seq = [scan_idx]

        self.rt_seq = [rt]
        self.mz_seq = [mz]
        self.int_seq = [intensity]
        self.ms2_seq = []

        # Count the gaps in the ROI's tail
        self.gap_counter = 0

        # Create attributes for the summarized values of the ROI
        self.mz = mz
        self.rt = np.nan
        self.scan_number = 0
        self.peak_area = np.nan
        self.peak_height = np.nan
        self.best_ms2 = None
        self.length = 0
        self.merged = False
        self.gaussian_similarity = 0.0
        self.noise_level = 0.0
        self.cut = False
        
        # Isotopes
        self.charge_state = 1
        self.is_isotope = False
        self.isotope_mz_seq = []
        self.isotope_int_seq = []
        self.isotope_id_seq = []

        # In-source fragments
        self.is_in_source_fragment = False
        self.isf_child_roi_id = []
        self.isf_parent_roi_id = None

        # Adducts
        self.adduct_type = None
        self.adduct_parent_roi_id = None
        self.adduct_child_roi_id = []

        # Annotation
        self.annotation = None
        self.formula = None
        self.similarity = None
        self.matched_peak_number = None
        self.smiles = None
        self.inchikey = None


    def extend_roi(self, scan_idx, rt, mz, intensity):
        """
        Function to extend an ROI by providing the scan indices, 
        retention time, m/z and intensity.

        Parameters
        ----------------------------------------------------------
        scan_idx: int
            Scan index of the ion to be added to the ROI
        rt: float
            Retention time of the ion to be added to the ROI
        mz: float
            m/z value of the ion to be added to the ROI
        intensity: int
            Intensity of the ion to be added to the ROI
        """

        # Extend the ROI
        self.scan_idx_seq.append(scan_idx)
        self.rt_seq.append(rt)
        self.mz_seq.append(mz)
        self.int_seq.append(intensity)


    def show_roi_info(self, show_annotation=False):
        """
        Function to print the information of the ROI.
        """

        print(f"ROI: {self.mz:.4f} m/z, {self.rt:.2f} min, {self.peak_area:.2f} area, {self.peak_height:.2f} height")
        print(f"ROI start time: {self.rt_seq[0]:.2f} min, ROI end time: {self.rt_seq[-1]:.2f} min")

        if show_annotation:
            # show isotopes, in-source fragments and adducts
            print("Isotope information:")
            print(f"Isotope charge state: {self.charge_state}")
            print(f"Isotope state: {self.isotope_state}")
            print(f"Isotope m/z: {self.isotope_mz_seq}")
            print(f"Isotope intensity: {self.isotope_int_seq}")

            print("In-source fragment information:")
            print(f"In-source fragment: {self.is_in_source_fragment}")
            print(f"Isf child roi id: {self.isf_child_roi_id}")
            print(f"Isf parent roi id: {self.isf_parent_roi_id}")

            print("Adduct information:")
            print(f"Adduct type: {self.adduct_type}")
            print(f"Adduct parent roi id: {self.adduct_parent_roi_id}")
            print(f"Adduct child roi id: {self.adduct_child_roi_id}")
    

    def get_mz_error(self):
        """
        Function to calculate the m/z error (maximum - minimum) of the ROI.
        """

        return np.max(self.mz_seq) - np.min(self.mz_seq)


    def find_rt_ph_pa(self):
        """
        Function to find the peak area of the ROI using trapzoidal rule.

        """
        
        idx = np.argmax(self.int_seq)
        self.mz = self.mz_seq[idx]
        self.rt = self.rt_seq[idx]
        self.peak_height = self.int_seq[idx]
        self.peak_area = int(np.trapz(y=self.int_seq, x=self.rt_seq) * 60)
    

    def find_top_average(self, num=3):
        """
        Function to find the peak height of the ROI by averaging
        the heighest three intensities.
        """

        d = np.sort(self.int_seq)[-num:]
        # calculate mean of non-zero values
        d = d[d != 0]
        self.top_average = np.mean(d, dtype=np.int64)
    

    def sum_roi(self, cal_gss=True):
        """
        Function to summarize the ROI to generate attributes.
        """
        self.int_seq = np.array(self.int_seq)
        end_idx = len(self.int_seq)-1

        while self.int_seq[end_idx] == 0 and self.int_seq[end_idx-1] == 0:
            end_idx -= 1

        end_idx += 1

        # keep one zero in the end of ROI
        self.mz_seq = self.mz_seq[:end_idx]
        self.int_seq = self.int_seq[:end_idx]
        self.rt_seq = self.rt_seq[:end_idx]
        self.scan_idx_seq = self.scan_idx_seq[:end_idx]
        
        self.find_rt_ph_pa()

        self.noise_level = calculate_noise_level(self.int_seq)

        if cal_gss:
            self.gaussian_similarity = calculate_gaussian_similarity(self.rt_seq, self.int_seq)
        self.length = np.sum(self.int_seq > 0)
    

    def subset_roi(self, start, end):
        """
        Function to subset the ROI by providing the positions.

        Parameters
        ----------------------------------------------------------
        start: int
            The start position of the ROI to be subsetted.
        end: int
            The end position of the ROI to be subsetted.
        """

        self.scan_idx_seq = self.scan_idx_seq[start:end]
        self.rt_seq = self.rt_seq[start:end]
        self.mz_seq = self.mz_seq[start:end]
        self.int_seq = self.int_seq[start:end]

        ms2_seq = []

        for ms2 in self.ms2_seq:
            if ms2.scan > self.scan_idx_seq[0] and ms2.scan < self.scan_idx_seq[-1]:
                ms2_seq.append(ms2)
        
        self.ms2_seq = ms2_seq