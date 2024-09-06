# Author: Hauxu Yu

# A module to align metabolic features from different samples
# Isotopes and in-source fragments are not considered in the alignment

# Import modules
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.stats import zscore
import pickle

from .raw_data_utils import read_raw_file_to_obj


def feature_alignment(path, parameters, drop_by_fill_pct_ratio=0.1):
    """
    A function to align the features from individual files (.txt).

    Parameters
    ----------------------------------------------------------
    path: str
        The path to the feature tables.
    parameters: Params object
        The parameters for alignment.

    Returns
    ----------------------------------------------------------
    feature_table: DataFrame
        The aligned feature table.
    """

    # STEP 1: get names of individual files (.txt)
    txt_file_names = parameters.sample_names
    txt_file_names = [f + ".txt" for f in txt_file_names]
    txt_file_names = [os.path.join(path, name) for name in txt_file_names]

    # STEP 2: initiate aligned features and parameters
    features = []
    mz_tol = parameters.align_mz_tol
    rt_tol = parameters.align_rt_tol
    file_quality_arr = np.ones(len(parameters.sample_names), dtype=bool)
    # select anchors for retention time correction
    if parameters.run_rt_correction:
        mz_ref, rt_ref = rt_anchor_selection(txt_file_names[:20])
        rt_cor_functions = {}
    
    # STEP 3: read individual feature tables and align features
    for i, file_name in enumerate(tqdm(parameters.sample_names)):
        # check if the file exists, if not, add the file to problematic_files
        if not os.path.exists(txt_file_names[i]):
            file_quality_arr[i] = False
            continue

        # read feature table
        current_table = pd.read_csv(txt_file_names[i], low_memory=False, sep="\t")
        current_table = current_table[current_table["MS2"].notna()|(current_table["length"]>=parameters.min_scan_num_for_alignment)]
        # sort current table by peak height from high to low
        current_table = current_table.sort_values(by="peak_height", ascending=False)
        current_table.index = range(len(current_table))
        avail_roi = np.ones(len(current_table), dtype=bool)
        # retention time correction
        if parameters.run_rt_correction and parameters.individual_sample_groups[i] != 'blank':
            rt_arr = current_table["RT"].values
            rt_max = np.max(rt_arr)
            rt_arr, model = retention_time_correction(mz_ref, rt_ref, current_table["m/z"].values, rt_arr, rt_max=rt_max, return_model=True)
            current_table["RT"] = rt_arr
            rt_cor_functions[file_name] = model

        if len(features) > 0:
            for f in features:
                # search mz and rt in the current feature table
                v = np.logical_and(np.abs(f.mz - current_table["m/z"].values) < mz_tol, np.abs(f.rt - current_table["RT"].values) < rt_tol)
                v = np.logical_and(v, avail_roi)
                v = np.where(v)[0]
                
                if len(v) > 0:
                    v = v[0]
                    f.mz_seq[i] = current_table.loc[v, "m/z"]
                    f.rt_seq[i] = current_table.loc[v, "RT"]
                    f.peak_height_seq[i] = current_table.loc[v, "peak_height"]
                    f.peak_area_seq[i] = current_table.loc[v, "peak_area"]
                    f.ms2_seq.append(current_table.loc[v, "MS2"])
                    f.detected_seq[i] = True
                    f.roi_id_seq[i] = current_table.loc[v, "ID"]
                    avail_roi[v] = False
                    # check if this file can be the reference file
                    if current_table.loc[v, "peak_height"] > f.highest_intensity:
                        f.highest_intensity = current_table.loc[v, "peak_height"]
                        f.gaussian_similarity = current_table.loc[v, "Gaussian_similarity"]
                        f.noise_level = current_table.loc[v, "noise_level"]
                        f.asymmetry_factor = current_table.loc[v, "asymmetry_factor"]
                        f.charge_state = current_table.loc[v, "charge"]
                        f.is_isotope = current_table.loc[v, "is_isotope"]
                        f.isotopes = current_table.loc[v, "isotopes"]
                        f.is_in_source_fragment = current_table.loc[v, "is_in_source_fragment"]
                        f.adduct_type = current_table.loc[v, "adduct"]
                        f.reference_file = file_name

        # if an ROI is not detected in the previous files, add it to the features
        if np.sum(avail_roi) > 0:
            v = np.where(avail_roi)[0]
            for j in v:
                f = Feature(file_number=len(parameters.sample_names))
                f.mz = current_table.loc[j, "m/z"]
                f.rt = current_table.loc[j, "RT"]
                f.mz_seq[i] = current_table.loc[j, "m/z"]
                f.rt_seq[i] = current_table.loc[j, "RT"]
                f.peak_height_seq[i] = current_table.loc[j, "peak_height"]
                f.peak_area_seq[i] = current_table.loc[j, "peak_area"]
                f.ms2_seq.append(current_table.loc[j, "MS2"])
                f.detected_seq[i] = True
                f.roi_id_seq[i] = current_table.loc[j, "ID"]
                f.highest_intensity = current_table.loc[j, "peak_height"]
                f.gaussian_similarity = current_table.loc[j, "Gaussian_similarity"]
                f.noise_level = current_table.loc[j, "noise_level"]
                f.asymmetry_factor = current_table.loc[j, "asymmetry_factor"]
                f.charge_state = current_table.loc[j, "charge"]
                f.is_isotope = current_table.loc[j, "is_isotope"]
                f.isotopes = current_table.loc[j, "isotopes"]
                f.is_in_source_fragment = current_table.loc[j, "is_in_source_fragment"]
                f.adduct_type = current_table.loc[j, "adduct"]
                f.reference_file = file_name
                features.append(f)

        # summarize (calculate the average mz and rt) and reorder the features
        features = sorted(features, key=lambda x: x.highest_intensity, reverse=True)
        for f in features:
            f.calculate_mzrt()

    # STEP 4: drop the problematic files and index the features
    if np.sum(file_quality_arr) > 0:
        parameters.sample_names = [parameters.sample_names[i] for i in range(len(parameters.sample_names)) if file_quality_arr[i]]
        parameters.individual_sample_groups = [parameters.individual_sample_groups[i] for i in range(len(parameters.individual_sample_groups)) if file_quality_arr[i]]
        for f in features:
            f.mz_seq = f.mz_seq[file_quality_arr]
            f.rt_seq = f.rt_seq[file_quality_arr]
            f.peak_height_seq = f.peak_height_seq[file_quality_arr]
            f.peak_area_seq = f.peak_area_seq[file_quality_arr]
            f.detected_seq = f.detected_seq[file_quality_arr]
            # remove nan in f.ms2_seq
            f.ms2_seq = [ms2 for ms2 in f.ms2_seq if ms2 == ms2]
    
    # STEP 5: calculate the fill percentage and remove the features with fill percentage less than 0.1
    blank_num = len([x for x in parameters.individual_sample_groups if 'blank' in x])
    if blank_num > 0:
        features = [f for f in features if np.sum(f.detected_seq[:-blank_num]) > drop_by_fill_pct_ratio*(len(parameters.sample_names)-blank_num)]
    else:
        features = [f for f in features if np.sum(f.detected_seq) > drop_by_fill_pct_ratio*len(parameters.sample_names)]
    
    # STEP 6: clean the feature list to remove the features with almost the same mz and rt
    if parameters.clean_feature_table:
        features = sorted(features, key=lambda x: x.mz)
        mz_groups = []
        mz_groups_tmp = [0]
        for i, f in enumerate(features):
            if i == 0:
                continue
            if np.abs(f.mz - features[i-1].mz) < mz_tol:
                mz_groups_tmp.append(i)
            else:
                mz_groups.append(mz_groups_tmp)
                mz_groups_tmp = [i]
        mz_groups.append(mz_groups_tmp)

        cleaned_idx = []
        for group in mz_groups:
            if len(group) == 1:
                cleaned_idx.append(group[0])
            else:
                group = sorted(group, key=lambda x: features[x].rt)
                rt_groups = []
                rt_groups_tmp = [group[0]]
                for i in range(1, len(group)):
                    if np.abs(features[group[i]].rt - features[group[i-1]].rt) < 0.1:
                        rt_groups_tmp.append(group[i])
                    else:
                        rt_groups.append(rt_groups_tmp)
                        rt_groups_tmp = [group[i]]
                rt_groups.append(rt_groups_tmp)
                for rt_group in rt_groups:
                    rt_group = sorted(rt_group, key=lambda x: features[x].highest_intensity, reverse=True)
                    cleaned_idx.append(rt_group[0])
        features = [features[i] for i in cleaned_idx]

    features = sorted(features, key=lambda x: x.highest_intensity, reverse=True)

    for i in range(len(features)):
        features[i].id = i
    
    # output the models to pickle files
    if parameters.run_rt_correction:
        with open(os.path.join(parameters.project_dir, "rt_correction_models.pkl"), "wb") as f:
            pickle.dump(rt_cor_functions, f)

    return features


def gap_filling(features, parameters, mode='forced_peak_picking'):
    """
    A function to fill the gaps in the aligned feature table.

    Parameters
    ----------------------------------------------------------
    features: list
        The aligned features.
    parameters: Params object
        The parameters for gap filling.
    mode: str
        The mode for gap filling.
        'forced_peak_picking': fill the gaps by forced peak picking.
        '0.1_min_intensity': fill the gaps by the minimum intensity * 0.1 (no available yet)

    Returns
    ----------------------------------------------------------
    features: list
        The aligned features with filled gaps.
    """

    # fill the gaps by forced peak picking
    if mode == 'forced_peak_picking':
        raw_file_names = os.listdir(parameters.sample_dir)
        raw_file_names = [f for f in raw_file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
        raw_file_names = [f for f in raw_file_names if not f.startswith(".")]
        raw_file_names = [os.path.join(parameters.sample_dir, f) for f in raw_file_names]

        # if retention time correction is applied, read the model
        if parameters.run_rt_correction and os.path.exists(os.path.join(parameters.project_dir, "rt_correction_models.pkl")):
            with open(os.path.join(parameters.project_dir, "rt_correction_models.pkl"), "rb") as f:
                rt_cor_functions = pickle.load(f)
        else:
            rt_cor_functions = None

        for i, file_name in enumerate(tqdm(parameters.sample_names)):
            matched_raw_file_name = [f for f in raw_file_names if file_name in f]
            if len(matched_raw_file_name) == 0:
                continue
            else:
                matched_raw_file_name = matched_raw_file_name[0]
                d = read_raw_file_to_obj(matched_raw_file_name, int_tol=parameters.int_tol, read_ms2=False)
                
                # correct retention time if model is available
                if rt_cor_functions is not None and file_name in rt_cor_functions.keys():
                    f = rt_cor_functions[file_name]
                    if f is not None:
                        d.correct_retention_time(f)

                for f in features:
                    if not f.detected_seq[i]:
                        _, eic_int, _, _ = d.get_eic_data(f.mz, f.rt, parameters.align_mz_tol, 0.05)
                        if len(eic_int) > 0:
                            f.peak_height_seq[i] = np.max(eic_int)

    # calculate the fill percentage after gap filling (blank samples are not included)
    blank_num = len([x for x in parameters.individual_sample_groups if 'blank' in x])
    if blank_num > 0:
        for f in features:
            f.fill_percentage = np.sum(f.detected_seq[:-blank_num]) / (len(parameters.sample_names)-blank_num) * 100
    else:
        for f in features:
            f.fill_percentage = np.sum(f.detected_seq) / len(parameters.sample_names) * 100
    return features


def output_feature_table(feature_table, output_path):
    """
    A function to output the aligned feature table.

    Parameters
    ----------------------------------------------------------
    feature_table: DataFrame
        The aligned feature table.
    output_path: str
        The path to save the aligned feature table.
    """

    # keep four digits for the m/z column and three digits for the RT column
    feature_table["m/z"] = feature_table["m/z"].apply(lambda x: round(x, 4))
    feature_table["RT"] = feature_table["RT"].apply(lambda x: round(x, 3))
    feature_table['fill_percentage'] = feature_table['fill_percentage'].apply(lambda x: round(x, 2))
    feature_table['similarity'] = feature_table['similarity'].astype(float)
    feature_table['similarity'] = feature_table['similarity'].apply(lambda x: round(x, 4))

    feature_table.to_csv(output_path, index=False, sep="\t")


def retention_time_correction(mz_ref, rt_ref, mz_arr, rt_arr, rt_max=50, mode='linear_interpolation', mz_tol=0.015, 
                              rt_tol=2.0, found_marker_ratio=0.4, return_model=False):
    """
    To correct retention times for feature alignment.

    There are three steps:
    1. Find the selected anchors in the given data.
    2. Create a model to correct retention times.
    3. Correct retention times.
    
    Parameters
    ----------
    mz_ref: np.array
        The m/z values of the reference features.
    rt_ref: np.array
        The retention times of the reference features.
    mz_arr: np.array
        The m/z values of the features to be corrected.
    rt_arr: np.array
        The retention times of the features to be corrected.
    mode: str
        The mode for retention time correction.
        'linear_interpolation': linear interpolation for retention time correction.
    mz_tol: float
        The m/z tolerance for selecting anchors.
    rt_tol: float
        The retention time tolerance for selecting anchors.
    
    Returns
    -------
    rt_corr: np.array
        The corrected retention times.
    """

    mz_matched = []
    rt_matched = []
    idx_matched = []

    for i in range(len(mz_ref)):
        v = np.logical_and(np.abs(mz_arr - mz_ref[i]) < mz_tol, np.abs(rt_arr - rt_ref[i]) < rt_tol)
        v = np.where(v)[0]
        if len(v) > 0:
            mz_matched.append(mz_arr[v[0]])
            rt_matched.append(rt_arr[v[0]])
            idx_matched.append(i)
    rt_ref = rt_ref[idx_matched]
    
    if len(idx_matched) < found_marker_ratio*len(mz_ref):
        if return_model:
            return rt_arr, None
        else:
            return rt_arr
    
    # remove outliers
    v = rt_ref - np.array(rt_matched)
    z = zscore(v)
    outliers = np.where(np.logical_and(np.abs(z) > 1, np.abs(v) > 0.1))[0]
    if len(outliers) > 0:
        rt_ref = np.delete(rt_ref, outliers)
        rt_matched = np.delete(rt_matched, outliers)

    if len(rt_matched) < 3:
        if return_model:
            return rt_arr, None
        else:
            return rt_arr

    if mode == 'linear_interpolation':
        # add zero and rt_max to the beginning and the end
        rt_matched = np.concatenate(([0], rt_matched, [rt_max+rt_matched[-1]-rt_ref[-1]]))
        rt_ref = np.concatenate(([0], rt_ref, [rt_max]))
        f = interp1d(rt_matched, rt_ref, fill_value='extrapolate')
        if return_model:
            return f(rt_arr), f
        else:
            return f(rt_arr)


def rt_anchor_selection(data_list, num=50, noise_tol=0.3, mz_tol=0.01, return_all_anchor=False):
    """
    To select anchors for retention time correction. The anchors are commonly detected in the provided
    data, of high intensity, with good peak shape, and equally distributed in the analysis time.

    The number of anchors

    Parameters
    ----------
    data_list: list
        A list of MSData objects or file names of the ouput .txt files.
    num: int
        The number of anchors to be selected.
    noise_tol: float
        The noise level for the anchors. Suggestions: 0.3 or lower.

    Returns
    -------
    anchors: list
        A list of anchors (dict) for retention time correction.
    """

    if isinstance(data_list[0], str):
        # check files and choose the one with the highest total intensity as reference files
        total_int = []
        for file in data_list:
            table = pd.read_csv(file, sep="\t", low_memory=False)
            total_int.append(np.sum(table["peak_height"]))
        ref_idx = np.argmax(total_int)

        table = pd.read_csv(data_list[ref_idx], sep="\t", low_memory=False)
        table = table.sort_values(by="m/z")
        mzs = table["m/z"].values
        n_scores = table["noise_level"].values
        v = [False]
        for i in range(1, len(mzs)-1):
            if mzs[i]-mzs[i-1] > mz_tol and mzs[i+1] - mzs[i] > mz_tol and n_scores[i] < noise_tol:
                v.append(True)
            else:
                v.append(False)
        v.append(False)
        valid_mzs = mzs[v]
        valid_rts = table["RT"].values[v]
        valid_ints = table["peak_height"].values[v]
        valid_mzs = valid_mzs[np.argsort(valid_ints)[-num:]]
        valid_rts = valid_rts[np.argsort(valid_ints)[-num:]]
        # sort the results by retention time
        valid_mzs = valid_mzs[np.argsort(valid_rts)]
        valid_rts = valid_rts[np.argsort(valid_rts)]


        train_idx, test_idx = _split_to_train_test(valid_rts)
        if return_all_anchor:
            return valid_mzs, valid_rts, train_idx, test_idx
        else:
            return valid_mzs[train_idx], valid_rts[train_idx]


def _split_to_train_test(array, interval=0.3):
    """
    To split the selected anchors into training and testing sets.

    Parameters
    ----------
    array: numpy.ndarray
        The retention times of the selected anchors.
    interval: float
        The time interval for splitting the anchors.

    Returns
    -------
    train_idx: list
        The indices of the training set.
    test_idx: list
        The indices of the testing set.
    """

    train_idx = [0]
    test_idx = []
    for i in range(1, len(array)):
        if array[i] - array[train_idx[-1]] < interval:
            test_idx.append(i)
        else:
            train_idx.append(i)

    return train_idx, test_idx


class Feature:
    """
    A class to model a feature in mass spectrometry data. Generally, a feature is defined as 
    a unique pair of m/z and retention time.
    """

    def __init__(self, file_number=1):
        """
        Define the attributes of a aligned feature.
        """

        # summarized information
        self.id = None                              # index of the feature
        self.reference_file = None                  # the reference file
        self.mz = 0.0                               # m/z
        self.rt = 0.0                               # retention time
        self.highest_intensity = 0.0                # the highest peak height from individual files (which is the reference file)
        self.best_ms2 = None                        # the best MS2
        self.gaussian_similarity = 0.0              # Gaussian similarity
        self.noise_level = 0.0                      # noise level
        self.asymmetry_factor = 0.0                 # asymmetry factor
        self.fill_percentage = 0.0                  # fill percentage, in %
        self.charge_state = 1                       # charge state
        self.is_isotope = False                     # is isotope
        self.isotopes = None                        # isotopes
        self.is_in_source_fragment = False          # is in-source fragment
        self.adduct_type = None                     # adduct type

        self.annotation = None                      # annotation
        self.search_mode = None                     # 'identity search', 'hybrid search', or 'mzrt_search'
        self.formula = None                         # molecular formula
        self.similarity = None                      # similarity score (0-1)
        self.matched_peak_number = None             # number of matched peaks
        self.smiles = None                          # SMILES
        self.inchikey = None                        # InChIKey
        self.matched_ms2 = None                     # matched MS2

        # individual files
        self.mz_seq = np.zeros(file_number)                     # m/z values from individual files
        self.rt_seq = np.zeros(file_number)                     # retention time values from individual files
        self.peak_height_seq = np.zeros(file_number)            # peak height from individual files
        self.peak_area_seq = np.zeros(file_number)              # peak area from individual files 
        self.ms2_seq = []                                       # best MS2 from individual files
        self.detected_seq = np.zeros(file_number, dtype=bool)   # whether the feature is detected in individual files
        self.roi_id_seq = -np.ones(file_number, dtype=int)      # ROI ID from individual files (-1 if not detected or gap filled)

        # statistical analysis
        self.fold_change = None             # fold change
        self.t_test_p = None                # t-test p-value
        self.adjusted_t_test_p = None       # adjusted t-test p-value
    
    def calculate_mzrt(self):
        """
        Calculate the m/z and retention time of the feature by averaging the values.
        """

        self.mz = np.mean(self.mz_seq[self.detected_seq])
        self.rt = np.mean(self.rt_seq[self.detected_seq])