# Author: Hauxu Yu

# A module for feature/peak detection

# Import modules
import numpy as np
from tqdm import tqdm
from scipy.signal import argrelextrema
from ms_entropy import calculate_entropy_similarity
import copy
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


def roi_finder(d):
    """
    A function to find the region of interest (ROI) in the MS data.

    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object that contains the MS data.
    """

    # A list to store the rois in progress
    rois = []
    # A list for finally selected ROIs
    final_rois = []

    # Initiate a set of rois using the first MS1 scan
    fs = d.scans[d.ms1_idx[0]]    # The first scan

    # Find the MS2 for this scan
    allocate_vec = loc_ms2_for_ms1_scan(d, d.ms1_idx[0])

    for i in range(len(fs.int_seq)):
        roi = Roi(scan_idx=d.ms1_idx[0], rt=fs.rt, mz=fs.mz_seq[i], intensity=fs.int_seq[i])
        if allocate_vec[i] > 0:
            roi.ms2_seq.append(d.scans[allocate_vec[i]])
        rois.append(roi)
    
    last_ms1_idx = d.ms1_idx[0]
    last_rt = fs.rt

    # Loop over all MS1 scans
    for ms1_idx in tqdm(d.ms1_idx[1:]):

        s = d.scans[ms1_idx]    # The current MS1 scan

        # Find the MS2 for this scan
        allocate_vec = loc_ms2_for_ms1_scan(d, ms1_idx)

        visited_idx = []    # A list to store the visited indices of ions in the current MS1 scan
        visited_rois_idx = []   # A list to store the visited indices of rois

        # Loop over all current rois
        for i, roi in enumerate(rois):
            
            mz_diff = np.abs(roi.mz - s.mz_seq)

            min_idx = np.argmin(mz_diff)

            if mz_diff[min_idx] < d.params.mz_tol_ms1:
                if min_idx not in visited_idx:
                    roi.extend_roi(scan_idx=ms1_idx, rt=s.rt, mz=s.mz_seq[min_idx], intensity=s.int_seq[min_idx])
                    
                    roi.gap_counter = 0

                    if allocate_vec[min_idx] > 0:
                        roi.ms2_seq.append(d.scans[allocate_vec[min_idx]])
                    visited_idx.append(min_idx)
                    visited_rois_idx.append(i)
            
        to_be_moved = []

        # Plus one to the gap counter of the rois that are not visited
        for i in range(len(rois)):
            if i not in visited_rois_idx:
                rois[i].extend_roi(scan_idx=ms1_idx, rt=s.rt, mz=np.nan, intensity=0)

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
                if allocate_vec[i] > 0:
                    roi.ms2_seq.append(d.scans[allocate_vec[i]])
                rois.append(roi)
        last_ms1_idx = ms1_idx
        last_rt = s.rt
           
    # Move all rois to final_rois
    for roi in rois:
        final_rois.append(roi)

    return final_rois


def find_roi_cut(roi, params):
    """
    A function to find place to cut an roi based on ion identity.
    An roi will be cut only if it has
    - params.min_ion_num or more non-zero intensities
    - two or more MS/MS spectra

    Parameters
    ----------------------------------------------------------
    roi: Roi object
        An Roi object that contains the roi.
    params: Params object
        A Params object that contains the parameters.
    """

    # counter the number of non-zero intensities in the roi
    non_zero_int = np.count_nonzero(roi.int_seq)

    if non_zero_int >= params.min_ion_num and len(roi.ms2_seq) >= 2:

        cut_positions = argrelextrema(np.array(roi.int_seq), np.less)[0]

        if len(cut_positions) != 0:
            final_cut_positions = []
            
            scan_for_cut = [roi.scan_idx_seq[i] for i in cut_positions]
            ms2_scan_number = [ms2.scan for ms2 in roi.ms2_seq]
            indices = np.searchsorted(np.array(ms2_scan_number), np.array(scan_for_cut))
            ms2_groups = [len(group) for group in np.split(ms2_scan_number, indices)]

            best_ms2s = []
            a = 0
            for i in ms2_groups:
                b = a + i
                best_ms2s.append(find_best_ms2(roi.ms2_seq[a:b]))
                a = b            

            ms2_ref = None
            for i in range(len(best_ms2s)):
                if best_ms2s[i] is None or best_ms2s[i].peaks.shape[0] == 0:
                    continue
                
                if ms2_ref is None:
                    ms2_ref = best_ms2s[i]
                else:
                    score = calculate_entropy_similarity(ms2_ref.peaks, best_ms2s[i].peaks)
                    if score < params.ms2_sim_tol:
                        final_cut_positions.append(cut_positions[i-1])
                        ms2_ref = best_ms2s[i]
                    else:
                        ms2_ref = find_best_ms2([ms2_ref, best_ms2s[i]])

            # cut the roi
            if len(final_cut_positions) != 0 and len(final_cut_positions) <= 5:
                return final_cut_positions
            else:
                return None
    else:
        return None


def roi_cutter(roi, positions):
    """
    A function to cut a long roi by given positions.

    Parameters
    ----------------------------------------------------------
    roi: Roi object
        An Roi object that contains the roi.
    positions: list
        A list of positions to cut the roi.
    """

    for i in positions:
        roi.int_seq[i] = roi.int_seq[i] / 2

    # add a zero and len(int_seq) to positions
    positions = np.insert(np.array([0, len(roi.int_seq)-1]), 1, positions)

    rois = []

    for i in range(len(positions)-1):
        fst = positions[i]
        snd = positions[i+1]+1
        temp = copy.deepcopy(roi)
        temp.subset_roi(fst, snd)
        rois.append(temp)

    return rois
    

def loc_ms2_for_ms1_scan(d, ms1_idx):
    """
    A function to allocate MS2 scans for the ions in a given MS1 scan.

    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object that contains the MS data.
    ms1_idx: int
        The index of the MS1 scan.
    """

    allocate_vec = [0] * len(d.scans[ms1_idx].mz_seq)
    mz_vec = d.scans[ms1_idx].mz_seq

    for i in range(ms1_idx+1, len(d.scans)):
        if d.scans[i].level == 1:
            break
        if d.scans[i].level == 2:
            mz_diff = np.abs(mz_vec - d.scans[i].precursor_mz)
            if np.min(mz_diff) < 0.01:
                allocate_vec[np.argmin(mz_diff)] = i

    return allocate_vec


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
        self.scan_number = -1
        self.peak_area = np.nan
        self.peak_height = np.nan
        self.top_average = np.nan
        self.best_ms2 = None
        self.length = 0
        self.apexes = []
        self.merged = False
        self.gaussian_similarity = 0.0

        # Ceature attributes for roi evaluation
        self.quality = None
        
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
    
    def roi_mz_error(self):
        """
        Function to calculate the m/z error (standard deviation of m/z) of the ROI.
        """

        return np.nanstd(self.mz_seq)


    def find_apex(self):
        """
        Function to find the retention time of the ROI.
        """
        
        tmp = max(range(len(self.int_seq)), key=self.int_seq.__getitem__)

        self.rt = self.rt_seq[tmp]
        self.scan_number = self.scan_idx_seq[tmp]
        self.peak_height = int(self.int_seq[tmp])
        self.mz = self.mz_seq[tmp]


    def find_roi_area(self):
        """
        Function to find the peak area of the ROI using trapzoidal rule.

        """
        
        self.peak_area = int(np.trapz(y=self.int_seq, x=self.rt_seq) * 60) # use seconds to calculate area
    

    def find_roi_top_average(self, num=3):
        """
        Function to find the peak height of the ROI by averaging
        the heighest three intensities.
        """

        d = np.sort(self.int_seq)[-num:]
        # calculate mean of non-zero values
        d = d[d != 0]
        self.top_average = np.mean(d, dtype=np.int64)
    

    def sum_roi(self):
        """
        Function to summarize the ROI to generate attributes.
        """

        end_idx = len(self.int_seq)-1

        while self.int_seq[end_idx] == 0 and self.int_seq[end_idx-1] == 0:
            end_idx -= 1

        end_idx += 1

        # keep one zero in the end of ROI
        self.mz_seq = self.mz_seq[:end_idx]
        self.int_seq = self.int_seq[:end_idx]
        self.rt_seq = self.rt_seq[:end_idx]
        self.scan_idx_seq = self.scan_idx_seq[:end_idx]
        
        self.find_apex()
        self.find_roi_area()
        self.find_roi_top_average()
        self.find_best_ms2()

        tmp = 0
        if self.int_seq[0] == 0:
            tmp += 1
        if self.int_seq[-1] == 0:
            tmp += 1
        
        if self.int_seq[0] == 0:
            self.mz_seq[0] = np.nan

        self.length = len(self.int_seq) - tmp
    

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


    def find_best_ms2(self):
        """
        Function to find the best MS2 spectrum of the ROI.
        """

        self.best_ms2 = find_best_ms2(self.ms2_seq)


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


def apex_detection(retention_times, intensities, merge_peak_rt_tol=0.02, retuen_idx=False, size=30, fold=1.5, find_peak_threh=0.2):
    """
    Detection of the apexes within a Region of Interest (ROI).s

    Parameters
    ----------
    retention_times : numpy.ndarray
        Retention times of the data points.
    intensities : numpy.ndarray
        Intensities of the data points.
    """

    # Apply a minimum filter to estimate the baseline
    baseline = uniform_filter1d(intensities, size=size)*fold
    substracted = intensities - baseline
    substracted[substracted < 0] = 0
    top = np.max(substracted)

    # Detect peaks
    peaks, _ = find_peaks(substracted, height=top*find_peak_threh)

    if len(peaks)==0:
        return []

    peak_rts = [retention_times[i] for i in peaks]
    peak_ints = [intensities[i] for i in peaks]

    # merge peaks that are too close and keep the higher one
    if merge_peak_rt_tol is not None:
        merged_rts = []
        merged_ints = []
        merged_idx = []

        merged_rts.append(peak_rts[0])
        merged_ints.append(peak_ints[0])
        merged_idx.append(peaks[0])

        for i in range(1, len(peak_rts)):
            if peak_rts[i] - merged_rts[-1] > merge_peak_rt_tol:
                merged_rts.append(peak_rts[i])
                merged_ints.append(peak_ints[i])
                merged_idx.append(peaks[i])
            else:
                if peak_ints[i] > merged_ints[-1]:
                    merged_ints[-1] = peak_ints[i]
                    merged_rts[-1] = peak_rts[i]
                    merged_idx[-1] = peaks[i]

        peak_rts = merged_rts
        peak_ints = merged_ints
        peaks = merged_idx

    if retuen_idx:
        return peaks
    else:
        return [[retention_times[i], intensities[i]] for i in peaks]