# Author: Huaxu Yu

# A module for feature detection

# imports
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
import bisect

from .feature_evaluation import calculate_noise_score, calculate_gaussian_similarity, calculate_asymmetry_factor, squared_error_to_smoothed_curve


"""
Classes
------------------------------------------------------------------------------------------------------------------------
"""

class Feature:
    """
    A class to store a feature characterized by a unique pair of m/z and retention time.
    """

    def __init__(self):

        # chromatographic peak
        self.rt_seq = []                     # retention time sequence
        self.signals = []                    # signal sequence (m/z, intensity)
        self.scan_idx_seq = []               # scan index sequence
        self.ms2_seq = []                    # MS2 spectra
        self.gap_counter = 0                 # count the number of consecutive zeros in the end of the peak

        # summary
        self.id = None                       # feature id
        self.feature_group_id = None         # peak group id
        self.mz = None                       # m/z
        self.rt = None                       # retention time
        self.scan_idx = None                 # scan index of the peak apex
        self.peak_height = None              # peak height
        self.peak_area = None                # peak area
        self.top_average = None              # average of the highest three intensities
        self.ms2 = None                      # representative MS2 spectrum (default: highest total intensity)
        self.length = 0                      # number of valid scans in the feature
        self.gaussian_similarity = 0.0       # Gaussian similarity
        self.noise_score = 0.0               # noise score
        self.asymmetry_factor = 0.0          # asymmetry factor
        self.sse = 0.0                       # squared error to the smoothed curve
        self.is_segmented = False            # whether the feature is segmented from a larger feature
        self.is_isotope = None               # whether the feature is an isotope
        self.charge_state = 1                # charge state of the feature
        self.isotope_signals = []            # isotope signals [[mz, intensity], ...]
        self.is_in_source_fragment = None    # whether the feature is an in-source fragment
        self.adduct_type = None              # adduct type

        self.annotation_algorithm = None     # annotation algorithm. Not used now.
        self.search_mode = None              # 'identity search', 'fuzzy search', or 'mzrt_search'
        self.similarity = None               # similarity score (0-1)
        self.annotation = None               # name of annotated compound
        self.formula = None                  # molecular formula
        self.matched_peak_number = None      # number of matched peaks
        self.smiles = None                   # SMILES
        self.inchikey = None                 # InChIKey
        self.matched_precursor_mz = None     # matched precursor m/z
        self.matched_ms2 = None              # matched ms2 spectra
        self.matched_adduct_type = None      # matched adduct type


    def extend(self, rt, signal, scan_idx):
        """
        Function to extend the chromatographic peak by providing the retention time, 
        signal and scan index.

        Parameters
        ----------
        rt: float
            Retention time.
        signal: list or numpy array
            [mz, intensity] of the signal.
        scan_idx: int
            Scan index.
        """

        self.rt_seq.append(rt)
        self.signals.append(signal)
        self.scan_idx_seq.append(scan_idx)


    def get_mz_error(self):
        """
        Function to calculate the 3*sigma error of the feature's m/z.

        Returns
        -------
        float
            The 3*sigma error of the feature's m/z.
        """

        return 3 * np.std(self.signals[:, 0])
    

    def get_rt_error(self):
        """
        Function to calculate the 3*sigma error of the feature's retention time.

        Returns
        -------
        float
            The 3*sigma error of the feature's retention time.
        """

        return 3 * np.std(self.rt_seq)


    def summarize(self, ph=True, pa=True, ta=True, g_score=True, n_score=True, a_score=True):
        """
        Summarize the feature by calculating the summary statistics.

        Parameters
        ----------
        ph: bool
            Whether to calculate the peak height.
        pa: bool
            Whether to calculate the peak area.
        ta: bool
            Whether to calculate the top average.
        g_score: bool
            Whether to calculate the Gaussian similarity.
        n_score: bool
            Whether to calculate the noise score.
        a_score: bool
            Whether to calculate the asymmetry factor.
        """

        self.signals = np.array(self.signals, dtype=np.float32)
        first,last = _trim_signals(self.signals)
        self.signals = self.signals[first:last]
        self.rt_seq = self.rt_seq[first:last]
        self.scan_idx_seq = self.scan_idx_seq[first:last]
        
        apx = np.argmax(self.signals[:, 1])
        self.mz = self.signals[apx, 0]
        self.rt = self.rt_seq[apx]
        self.scan_idx = self.scan_idx_seq[apx]
        self.length = np.sum(self.signals[:, 1] > 0)

        if ph:
            self.peak_height = int(self.signals[apx, 1])
        if pa:
            self.peak_area = int(np.trapz(y=self.signals[:, 1], x=self.rt_seq) * 60)
        if ta:
            self.top_average = int(np.mean(np.sort(self.signals[:, 1])[-3:]))
        if n_score:
            self.noise_score = calculate_noise_score(self.signals[:, 1])
        if g_score:
            self.gaussian_similarity = calculate_gaussian_similarity(self.rt_seq, self.signals[:, 1])
        if a_score:
            self.asymmetry_factor = calculate_asymmetry_factor(self.signals[:, 1])


    def subset(self, start, end, summarize=True):
        """
        Keep the subset of the feature by providing the start and end positions. Note that the 
        summary statistics will be recalculated in this function by default.

        Parameters
        ----------
        start: int
            The start position.
        end: int
            The end position. The data point at the end position is not included.
        summarize: bool
            Whether to recalculate the summary statistics.
        """
        
        self.rt_seq = self.rt_seq[start:end]
        self.signals = self.signals[start:end]
        self.scan_idx_seq = self.scan_idx_seq[start:end]
        self.ms2_seq = [ms2 for ms2 in self.ms2_seq if ms2.id > self.scan_idx_seq[0] and ms2.id < self.scan_idx_seq[-1]]

        if summarize:
            self.summarize(pa=False, ta=False, g_score=False, a_score=False)


"""
Functions
------------------------------------------------------------------------------------------------------------------------
"""

def detect_features(d):
    """
    Detect features in the MS data.

    Parameters
    ----------
    d: MSData object
        An MSData object that contains the MS data.

    Returns
    -------
    final_features: list
        A list of detected features.
    """

    # A list to store the rois in progress
    features = []
    # A list for the finally detected ROIs
    final_features = []

    # Initiate a set of rois using the first MS1 scan
    s = d.scans[d.ms1_idx[0]]    # The first scan

    for i in range(len(s.signals)):
        feature = Feature()
        feature.extend(rt=s.time, signal=s.signals[i], scan_idx=d.ms1_idx[0])
        features.append(feature)

    # Loop over all MS1 scans
    for ms1_idx in d.ms1_idx[1:]:

        s = d.scans[ms1_idx]                                  # The current MS1 scan
        if len(s.signals) == 0:
            continue
        avlb_signals = np.ones(len(s.signals), dtype=bool)    # available signals to assign to features
        avlb_features = np.ones(len(features), dtype=bool)    # available features to take new signals
        to_be_moved = []                                      # features to be moved to final_features
        
        for i, feature in enumerate(features):
            min_idx = _find_closest_index_ordered(array=s.signals[:,0], target=feature.signals[-1][0], 
                                                  tol=d.params.mz_tol_ms1)
            if min_idx is not None and avlb_signals[min_idx]:
                feature.extend(rt=s.time, signal=s.signals[min_idx], scan_idx=ms1_idx)
                feature.gap_counter = 0
                avlb_signals[min_idx] = False
                avlb_features[i] = False
            else:
                feature.extend(rt=s.time, signal=[feature.signals[-1][0], 0], scan_idx=ms1_idx)
                feature.gap_counter = feature.gap_counter + 1
                if feature.gap_counter > d.params.feature_gap_tol:
                    to_be_moved.append(i)

        # Move the features that have not been visited for a long time to final_features
        for i in to_be_moved[::-1]:
            final_features.append(features.pop(i))
        
        # Create new rois for the remaining signals
        for i, signal in enumerate(s.signals):
            if avlb_signals[i]:
                feature = Feature()
                feature.extend(rt=s.time, signal=signal, scan_idx=ms1_idx)
                features.append(feature)

        features.sort(key=lambda x: x.signals[-1][1], reverse=True)

    # Move all features to final_features
    for feature in features:
        final_features.append(feature)
    
    # summarize features
    for feature in final_features:
        feature.summarize(pa=False, g_score=False, a_score=False)
    
    # sort by m/z
    final_features.sort(key=lambda x: x.mz)

    return final_features


def segment_feature(feature, sigma=1.2, prominence_ratio=0.05, distance=5, peak_height_tol=1000,
                    length_tol=5, sse_tol=0.3):
    """
    Function to segment a feature into multiple features based on the edge detection.

    Parameters
    ----------
    sigma: float
        The sigma value for Gaussian filter.
    prominence_ratio: float
        The prominence ratio for finding peaks. prom = np.max(y)*prominence_ratio
    distance: int
        The minimum distance between peaks.
    peak_height_tol: float
        The peak height tolerance for segmentation.
    length_tol: int
        The length tolerance for segmentation.
    sse_tol: float
        The squared error tolerance for segmentation.

    Returns
    -------
    segmented_features: list
        A list of segmented features.
    """

    # if peak height is too low or the length is too short, skip segmentation
    peak_tmp = feature.signals[:,1]
    dp = np.sum(peak_tmp > peak_height_tol)
    if feature.peak_height < peak_height_tol or dp < length_tol:
        return [feature]
    
    # add zero to the front and the end of the signal to facilitate the edge detection
    peak_tmp = np.concatenate(([0], peak_tmp, [0]))
    ss = gaussian_filter1d(peak_tmp, sigma=sigma)
    feature.sse = squared_error_to_smoothed_curve(original_signal=peak_tmp, fit_signal=ss)
    if feature.sse > sse_tol:
        return [feature]

    # correction of prominence ratio and sigma based on noise level and data points
    prominence_ratio = np.clip(0.03, prominence_ratio * feature.sse * 20, 0.1)
    prominence = np.max(ss)*prominence_ratio
    sigma = np.clip(0.5, sigma * dp / 33, 1.2)
    ss = gaussian_filter1d(peak_tmp, sigma=sigma)   # recalculate the smoothed signal

    peaks, _ = find_peaks(ss, prominence=prominence, distance=distance)

    # baseline filter
    peaks = peaks - 1   # correct the index
    
    baseline = np.median(peak_tmp)

    # the resulting peaks should have a height larger than baseline
    peaks = peaks[feature.signals[peaks,1] > baseline]

    if len(peaks) < 2:
        return [feature]
    
    positions = [0]
    for i in range(len(peaks)-1):
        lowest_int = 1e10
        for j in range(peaks[i], peaks[i+1]):
            if feature.signals[:,1][j] < lowest_int:
                lowest_int = feature.signals[j,1]
                lowest_int_idx = j
        positions.append(lowest_int_idx)
    positions.append(len(feature.signals)-1)
    
    segmented_features = []
    for i in range(len(positions)-1):
        tmp = deepcopy(feature)
        tmp.subset(start=positions[i], end=positions[i+1]+1)
        tmp.is_segmented = True
        segmented_features.append(tmp)

    return segmented_features


"""
Helper functions
------------------------------------------------------------------------------------------------------------------------
"""

def _find_closest_index_ordered(array, target, tol=0.01):
    """
    Function to find the index of the closest value in an ordered array.

    Parameters
    ----------
    array: list or numpy array
        An ordered array.
    target: float
        The target value.
    tol: float
        The tolerance for the closest value.

    Returns
    -------
    idx: int
        The index of the closest value.
    """

    idx = bisect.bisect_left(array, target)
    
    if idx == 0:
        if array[idx] - target < tol:
            return 0
        else:
            return None
    if idx == len(array):
        if target - array[idx - 1] < tol:
            return len(array) - 1
        else:
            return None
    
    before = array[idx - 1]
    after = array[idx]
    
    if after - target < target - before and after - target < tol:
        return idx
    elif after - target > target - before and target - before < tol:
        return idx - 1
    else:
        return None


def _trim_signals(signals):
    """
    Function to trim the signals by removing the zeros in the beginning and the end.

    Parameters
    ----------
    signals: 2D numpy array
        A numpy array of signals as [[mz, intensity], ...]
    """

    first = 0
    for i in signals[:, 1]:
        if i != 0.:
            break
        else:
            first = first + 1
    last = len(signals)
    for i in signals[::-1, 1]:
        if i != 0.:
            break
        else:
            last = last - 1

    return first, last