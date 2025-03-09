# Author: Huaxu Yu

# A module to align features (characterized by unique m/z and retention time) from different files. 

# imports
import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
import pickle
from copy import deepcopy

from .raw_data_utils import read_raw_file_to_obj
from .params import Params
from .utils_functions import convert_signals_to_string, extract_signals_from_string


"""
Classes
------------------------------------------------------------------------------------------------------------------------
"""

class AlignedFeature:
    """
    A class to model a feature in mass spectrometry data. Generally, a feature is defined as 
    a unique pair of m/z and retention time.
    """

    def __init__(self, file_number=1):
        """
        Define the attributes of a aligned feature.

        Parameters
        ----------
        file_number: int
            The number of files.
        """

        # individual files
        self.feature_id_arr = -np.ones(file_number, dtype=int)      # feature ID from individual files (-1 if not detected or gap filled)
        self.mz_arr = np.zeros(file_number)                         # m/z
        self.rt_arr = np.zeros(file_number)                         # retention time
        self.scan_idx_arr = np.zeros(file_number, dtype=int)        # scan index of the peak apex
        self.peak_height_arr = np.zeros(file_number)                # peak height
        self.peak_area_arr = np.zeros(file_number)                  # peak area
        self.top_average_arr = np.zeros(file_number)                # average of the highest three intensities
        self.ms2_seq = []                                           # representative MS2 spectrum from each file (default: highest total intensity)
        self.length_arr = np.zeros(file_number, dtype=int)          # length (i.e. non-zero scans in the peak)
        self.gaussian_similarity_arr = np.zeros(file_number)        # Gaussian similarity
        self.noise_score_arr = np.zeros(file_number)                # noise score
        self.asymmetry_factor_arr = np.zeros(file_number)           # asymmetry factor
        self.sse_arr = np.zeros(file_number)                        # squared error to the smoothed curve
        self.is_segmented_arr = np.zeros(file_number, dtype=bool)   # whether the peak is segmented

        # summary
        self.id = None                              # index of the feature
        self.feature_group_id = None                # feature group ID
        self.mz = 0.0                               # m/z
        self.rt = 0.0                               # retention time
        self.reference_file = None                  # the reference file with the highest peak height
        self.reference_scan_idx = None              # the scan index of the peak apex from the reference file
        self.highest_intensity = 0.0                # the highest peak height from individual files (which is the reference file)
        self.ms2 = None                             # representative MS2 spectrum
        self.ms2_reference_file = None              # the reference file for the representative MS2 spectrum
        self.gaussian_similarity = 0.0              # Gaussian similarity from the reference file
        self.noise_score = 0.0                      # noise level from the reference file
        self.asymmetry_factor = 0.0                 # asymmetry factor from the reference file
        self.detection_rate = 0.0                   # number of detected files / total number of files (blank not included)
        self.detection_rate_gap_filled = 0.0        # number of detected files after gap filling / total number of files (blank not included)
        self.charge_state = 1                       # charge state
        self.is_isotope = False                     # whether it is an isotope
        self.isotope_signals = None                 # isotope signals [[m/z, intensity], ...]
        self.is_in_source_fragment = False          # whether it is an in-source fragment
        self.adduct_type = None                     # adduct type

        self.annotation_algorithm = None            # annotation algorithm. Not used now.
        self.search_mode = None                     # 'identity search', 'fuzzy search', or 'mzrt_search'
        self.similarity = None                      # similarity score (0-1)
        self.annotation = None                      # name of annotated compound
        self.formula = None                         # molecular formula
        self.matched_peak_number = None             # number of matched peaks
        self.smiles = None                          # SMILES
        self.inchikey = None                        # InChIKey
        self.matched_precursor_mz = None            # matched precursor m/z
        self.matched_retention_time = None          # matched retention time
        self.matched_adduct_type = None             # matched adduct type
        self.matched_ms2 = None                     # matched ms2 spectra


"""
Functions
------------------------------------------------------------------------------------------------------------------------
"""

def feature_alignment(path: str, params: Params):
    """
    Align the features from multiple processed single files as .txt format.

    Parameters
    ----------
    path: str
        The path to the feature tables.
    params: Params object
        The parameters for alignment including sample names and sample groups.

    Returns
    -------
    features: list
        A list of AlignedFeature objects.
    """

    # STEP 1: preparation
    features = []
    names = [f + ".txt" for f in params.sample_names]
    names = [os.path.join(path, name) for name in names]
    not_found_files = [params.sample_names[i] for i in range(len(names)) if not os.path.exists(names[i])]
    if len(not_found_files) > 0:
        for f in not_found_files:
            params.problematic_files[f] = "file does not exist"
        # remove those files from names
        names = [names[i] for i in range(len(names)) if params.sample_names[i] not in not_found_files]
        # remove those files from sample metadata
        params.sample_metadata = params.sample_metadata[~params.sample_metadata.iloc[:,1].isin(not_found_files)]
        params.sample_names = list(params.sample_metadata.iloc[:, 0])
    
    # find anchors for retention time correction
    if params.correct_rt:
        intensities = []
        for n in names:
            df = pd.read_csv(n, sep="\t", low_memory=False)
            intensities.append(np.sum(df["peak_height"]))
        # use the file with median total intensity as the reference file
        anchor_selection_name = names[np.argsort(intensities)[len(intensities)//2]]
        mz_ref, rt_ref = rt_anchor_selection(anchor_selection_name, num=100)
        rt_cor_functions = {}
    
    # STEP 2: read individual feature tables and align features
    for i, file_name in enumerate(tqdm(params.sample_names)):
        
        if not os.path.exists(names[i]):
            continue

        # read feature table
        current_table = pd.read_csv(names[i], low_memory=False, sep="\t")
        current_table = current_table[current_table["MS2"].notna()|(current_table["total_scans"]>params.scan_number_cutoff)]
        
        # sort current table by peak height from high to low
        current_table = current_table.sort_values(by="peak_height", ascending=False)
        current_table.index = range(len(current_table))
        tmp_table = current_table[current_table["peak_height"] > params.ms1_abs_int_tol * 5]

        availible_features = np.ones(len(current_table), dtype=bool)

        # retention time correction
        if params.correct_rt:
            _, model = retention_time_correction(mz_ref, rt_ref, tmp_table["m/z"].values, tmp_table["RT"].values, rt_tol=params.rt_tol_rt_correction)
            if model is not None:
                current_table["RT"] = model(current_table["RT"].values)
            rt_cor_functions[file_name] = model

        for f in features:
            v1 = np.abs(f.mz - current_table["m/z"].values) < params.mz_tol_alignment
            v2 = np.abs(f.rt - current_table["RT"].values) < params.rt_tol_alignment
            idx = np.where(v1 & v2 & availible_features)[0]
            
            if len(idx) > 0:
                _assign_value_to_feature(f=f, df=current_table, i=i, p=idx[0], file_name=file_name)
                availible_features[idx[0]] = False
                # check if this file can be the reference file
                if current_table.loc[idx[0], "peak_height"] > f.highest_intensity:
                    _assign_reference_values(f=f, df=current_table, i=i, p=idx[0], file_name=file_name)

        # if an feature is not detected in the previous files, add it to the features
        for j, b in enumerate(availible_features):
            if b:
                f = AlignedFeature(file_number=len(params.sample_names))
                _assign_value_to_feature(f=f, df=current_table, i=i, p=j, file_name=file_name)
                _assign_reference_values(f=f, df=current_table, i=i, p=j, file_name=file_name)
                features.append(f)

        # summarize (calculate the average mz and rt) and reorder the features
        features = sorted(features, key=lambda x: x.highest_intensity, reverse=True)
    
    # save the retention time correction models
    if params.correct_rt:
        with open(os.path.join(params.project_file_dir, "rt_correction_models.pkl"), "wb") as f:
            pickle.dump(rt_cor_functions, f)
    
    # choose the best ms2
    for f in features:
        if len(f.ms2_seq) == 0:
            continue
        parsed_ms2 = []
        for file_name, ms2 in f.ms2_seq:
            signals = extract_signals_from_string(ms2)
            parsed_ms2.append([file_name, signals])
        # sort parsed ms2 by summed intensity
        parsed_ms2.sort(key=lambda x: np.sum(x[1][:, 1]), reverse=True)
        f.ms2_reference_file = parsed_ms2[0][0]
        f.ms2 = convert_signals_to_string(parsed_ms2[0][1])
    
    # STEP 3: calculate the detection rate and drop features using the detection rate cutoff
    v = ~params.sample_metadata['is_blank']
    for f in features:
        f.detection_rate = np.sum(f.feature_id_arr[v] != -1) / np.sum(v)
    features = [f for f in features if f.detection_rate > params.detection_rate_cutoff]
    
    # STEP 4: clean features by merging the features with almost the same m/z and retention time
    if params.merge_features:
        features = merge_features(features, params)

    # STEP 5: gap filling
    print("\tFilling gaps...")
    if params.fill_gaps:
        features = gap_filling(features, params)
    
    # STEP 6: index the features
    features.sort(key=lambda x: x.highest_intensity, reverse=True)
    for i, f in enumerate(features):
        f.id = i

    return features


def gap_filling(features, params: Params):
    """
    Fill the gaps for aligned features.

    Parameters
    ----------------------------------------------------------
    features: list
        The aligned features.
    params: Params object
        The parameters for gap filling.

    Returns
    ----------------------------------------------------------
    features: list
        The aligned features with filled gaps.
    """

    # fill the gaps by forced peak picking (local maximum)
    if params.gap_filling_method == 'local_maximum':

        # if retention time correction is applied, read the model
        if params.correct_rt and os.path.exists(os.path.join(params.project_dir, "rt_correction_models.pkl")):
            with open(os.path.join(params.project_dir, "rt_correction_models.pkl"), "rb") as f:
                rt_cor_functions = pickle.load(f)
        else:
            rt_cor_functions = None

        for i, file_name in enumerate(tqdm(params.sample_names)):
            fn = os.path.join(params.tmp_file_dir, file_name + ".mzpkl")
            if os.path.exists(fn):
                d = read_raw_file_to_obj(fn)
                # correct retention time if model is available
                if rt_cor_functions is not None and file_name in rt_cor_functions.keys():
                    f = rt_cor_functions[file_name]
                    if f is not None:
                        d.correct_retention_time(f)

                for f in features:
                    if f.feature_id_arr[i] == -1:
                        eic_time_arr, eic_signals, _ = d.get_eic_data(f.mz, f.rt, params.mz_tol_alignment, params.gap_filling_rt_window)
                        if len(eic_signals) > 0:
                            f.peak_height_arr[i] = np.max(eic_signals[:, 1])
                            f.peak_area_arr[i] = int(np.trapz(y=eic_signals[:, 1], x=eic_time_arr))
                            f.top_average_arr[i] = np.mean(np.sort(eic_signals[:, 1])[-3:])
    
    # calculate the detection rate after gap filling (blank samples are not included)
    v = ~params.sample_metadata['is_blank']
    for f in features:
        f.detection_rate_gap_filled = np.sum(f.peak_height_arr[v] > 0) / np.sum(v)
    
    return features


def merge_features(features: list, params: Params):
    """
    Clean features by merging features with almost the same m/z and retention time.

    Parameters
    ----------
    features: list
        A list of AlignedFeature objects.
    params: Params object
        The parameters for feature cleaning.

    Returns
    -------
    features: list
        A list of cleaned AlignedFeature objects.
    """

    features = sorted(features, key=lambda x: x.mz)
    features_cleaned = []

    tmp = [features[0]]
    features_rest = []
    for i in range(1, len(features)):
        if features[i].mz - features[i-1].mz < params.mz_tol_merge_features:
            tmp.append(features[i])
        else:
            if len(tmp) == 1:
                features_cleaned.append(tmp[0])
            else:
                features_rest.append(tmp)
            tmp = [features[i]]
    if len(tmp) == 1:
        features_cleaned.append(tmp[0])
    else:
        features_rest.append(tmp)

    for g in features_rest:
        g = sorted(g, key=lambda x: x.rt)
        tmp = [g[0]]
        for i in range(1, len(g)):
            if g[i].rt - g[i-1].rt < params.rt_tol_merge_features:
                tmp.append(g[i])
            else:
                if len(tmp) == 1:
                    features_cleaned.append(tmp[0])
                else:
                    tmp.sort(key=lambda x: x.highest_intensity, reverse=True)
                    merged_f = deepcopy(tmp[0])
                    merged_f.peak_height_arr = np.max([f.peak_height_arr for f in tmp], axis=0)
                    merged_f.peak_area_arr = np.max([f.peak_area_arr for f in tmp], axis=0)
                    merged_f.top_average_arr = np.max([f.top_average_arr for f in tmp], axis=0)
                    features_cleaned.append(merged_f)
                tmp = [g[i]]
        if len(tmp) == 1:
            features_cleaned.append(tmp[0])
        else:
            tmp.sort(key=lambda x: x.highest_intensity, reverse=True)
            merged_f = deepcopy(tmp[0])
            merged_f.peak_height_arr = np.max([f.peak_height_arr for f in tmp], axis=0)
            merged_f.peak_area_arr = np.max([f.peak_area_arr for f in tmp], axis=0)
            merged_f.top_average_arr = np.max([f.top_average_arr for f in tmp], axis=0)
            features_cleaned.append(merged_f)

    features_cleaned.sort(key=lambda x: x.highest_intensity, reverse=True)

    return features_cleaned


def convert_features_to_df(features, sample_names, quant_method="peak_height"):
    """
    Convert the aligned features to a DataFrame.

    Parameters
    ----------
    features : list
        list of features
    sample_names : list
        list of sample names
    quant_method : str
        quantification method, "peak_height", "peak_area" or "top_average"

    Returns
    -------
    feature_table : pd.DataFrame
        feature DataFrame
    """

    results = []
    sample_names = list(sample_names)
    columns=["group_ID", "feature_ID", "m/z", "RT", "adduct", "is_isotope", "is_in_source_fragment", "Gaussian_similarity", "noise_score", 
             "asymmetry_factor", "detection_rate", "detection_rate_gap_filled", "alignment_reference_file", "charge", "isotopes", "MS2_reference_file", "MS2", "matched_MS2", 
             "search_mode", "annotation", "formula", "similarity", "matched_peak_number", "SMILES", "InChIKey"] + sample_names

    for f in features:
        if quant_method == "peak_height":
            quant = list(f.peak_height_arr)
        elif quant_method == "peak_area":
            quant = list(f.peak_area_arr)
        elif quant_method == "top_average":
            quant = list(f.top_average_arr)
        
        quant = [int(x) for x in quant]
        
        results.append([f.feature_group_id, f.id, f.mz, f.rt, f.adduct_type, f.is_isotope, f.is_in_source_fragment, f.gaussian_similarity, f.noise_score,
                        f.asymmetry_factor, f.detection_rate, f.detection_rate_gap_filled, f.reference_file, f.charge_state, f.isotope_signals, f.ms2_reference_file,
                        f.ms2, f.matched_ms2, f.search_mode, f.annotation, f.formula, f.similarity, f.matched_peak_number, f.smiles, f.inchikey] + quant)
        
    feature_table = pd.DataFrame(results, columns=columns)
    
    return feature_table


def output_feature_to_msp(feature_table, output_path):
    """
    A function to output MS2 spectra to MSP format.

    Parameters
    ----------
    feature_table : pandas.DataFrame
        A DataFrame containing MS2 spectra.
    output_path : str
        The path to the output MSP file.
    """
    
    # check the output path to make sure it is a .msp file and it esists
    if not output_path.lower().endswith(".msp"):
        raise ValueError("The output path must be a .msp file.")

    with open(output_path, "w") as f:
        for i in range(len(feature_table)):
            f.write("ID: " + str(feature_table['feature_ID'][i]) + "\n")
            if feature_table['MS2'][i] is None or feature_table['MS2'][i]!=feature_table['MS2'][i]:
                f.write("NAME: Unknown\n")
                f.write("PRECURSORMZ: " + str(feature_table['m/z'][i]) + "\n")
                f.write("PRECURSORTYPE: " + str(feature_table['adduct'][i]) + "\n")
                f.write("RETENTIONTIME: " + str(feature_table['RT'][i]) + "\n")
                f.write("Num Peaks: " + "0\n")
                f.write("\n")
                continue

            if feature_table['annotation'][i] is None:
                name = "Unknown"
            else:
                name = str(feature_table['annotation'][i])

            peaks = re.findall(r"\d+\.\d+", feature_table['MS2'][i])
            f.write("NAME: " + name + "\n")
            f.write("PRECURSORMZ: " + str(feature_table['m/z'][i]) + "\n")
            f.write("PRECURSORTYPE: " + str(feature_table['adduct'][i]) + "\n")
            f.write("RETENTIONTIME: " + str(feature_table['RT'][i]) + "\n")
            f.write("SEARCHMODE: " + str(feature_table['search_mode'][i]) + "\n")
            f.write("FORMULA: " + str(feature_table['formula'][i]) + "\n")
            f.write("INCHIKEY: " + str(feature_table['InChIKey'][i]) + "\n")
            f.write("SMILES: " + str(feature_table['SMILES'][i]) + "\n")
            f.write("Num Peaks: " + str(int(len(peaks)/2)) + "\n")
            for j in range(len(peaks)//2):
                f.write(str(peaks[2*j]) + "\t" + str(peaks[2*j+1]) + "\n")
            f.write("\n")


def output_feature_table(feature_table, output_path):
    """
    Output the aligned feature table.

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
    feature_table['detection_rate'] = feature_table['detection_rate'].apply(lambda x: round(x, 2))
    feature_table['detection_rate_gap_filled'] = feature_table['detection_rate_gap_filled'].apply(lambda x: round(x, 2))
    feature_table['similarity'] = feature_table['similarity'].astype(float)
    feature_table['similarity'] = feature_table['similarity'].apply(lambda x: round(x, 4))

    feature_table.to_csv(output_path, index=False, sep="\t")


def retention_time_correction(mz_ref, rt_ref, mz_arr, rt_arr, mz_tol=0.01, rt_tol=0.5, mode='linear_interpolation', 
                              rt_max=None):
    """
    To correct retention times for feature alignment.

    There are three steps:
    1. Find the selected anchors in the given data.
    2. Create a model to correct retention times.
    3. Correct retention times.
    
    Parameters
    ----------
    mz_ref: np.array
        The m/z values of the selected anchors from another reference file.
    rt_ref: np.array
        The retention times of the selected anchors from another reference file.
    mz_arr: np.array
        Feature m/z values in the current file.
    rt_arr: np.array
        Feature retention times in the current file.
    mz_tol: float
        The m/z tolerance for selecting anchors.
    rt_tol: float
        The retention time tolerance for selecting anchors.
    mode: str
        The mode for retention time correction. Not used now.
        'linear_interpolation': linear interpolation for retention time correction.
    rt_max: float
        End of the retention time range.
    return_model: bool
        Whether to return the model for retention time correction.
    
    Returns
    -------
    rt_arr: np.array
        The corrected retention times.
    f: interp1d
        The model for retention time correction.
    """

    rt_matched = []
    idx_matched = []

    for i in range(len(mz_ref)):
        v = np.logical_and(np.abs(mz_arr - mz_ref[i]) < mz_tol, np.abs(rt_arr - rt_ref[i]) < rt_tol)
        v = np.where(v)[0]
        if len(v) == 1:
            rt_matched.append(rt_arr[v[0]])
            idx_matched.append(i)
    rt_ref = rt_ref[idx_matched]
    
    if len(idx_matched) < 5:
        return rt_arr, None
    
    # remove outliers
    v = rt_ref - np.array(rt_matched)
    k = np.abs(v - np.mean(v)) < np.std(v)
    rt_ref = rt_ref[k]
    rt_matched = np.array(rt_matched)[k]

    if len(rt_matched) < 5:
        return rt_arr, None
    
    x = [0]
    y = [0]
    for i in range(len(rt_matched)):
        if rt_matched[i] - x[-1] > 0.1:
            x.append(rt_matched[i])
            y.append(rt_ref[i])

    f = interp1d(x, y, fill_value='extrapolate')
    
    return f(rt_arr), f


def rt_anchor_selection(data_path, num=50, noise_score_tol=0.1, mz_tol=0.01):
    """
    Retention time anchors have unique m/z values and low noise scores. From all candidate features, 
    the top *num* features with the highest peak heights are selected as anchors.

    Parameters
    ----------
    data_path : str
        Absolute directory to the feature tables.
    num : int
        The number of anchors to be selected.
    noise_tol : float
        The noise level for the anchors. Suggestions: 0.3 or lower.
    mz_tol : float
        The m/z tolerance for selecting anchors.

    Returns
    -------
    anchors: list
        A list of anchors (dict) for retention time correction.
    """

    df = pd.read_csv(data_path, sep="\t", low_memory=False)
    # sort by m/z
    df = df.sort_values(by="m/z")
    df.index = range(len(df))
    mzs = df["m/z"].values
    candidates = []
    diff = np.diff(mzs)
    for i in range(1, len(mzs)-1):
        if diff[i-1] > mz_tol and diff[i] > mz_tol and df["noise_score"][i] < noise_score_tol:
            candidates.append(i)
    candidates = np.array(candidates)
    candidates = candidates[np.argsort(df["peak_height"].values[candidates])[-num:]]
    # reverse the order
    candidates = candidates[::-1]
    valid_mzs = mzs[candidates]
    valid_rts = df["RT"].values[candidates]
    
    return valid_mzs, valid_rts


"""
Internal Functions
------------------------------------------------------------------------------------------------------------------------
"""

def split_to_train_test(array, interval=0.1):
    """
    Split the selected anchors into training and testing sets.

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

    train_idx = [0, len(array)-1]
    for i in range(1, len(array)):
        if array[i] - array[train_idx[-1]] > interval:
            train_idx.append(i)
    train_idx.sort()
    test_idx = [i for i in range(len(array)) if i not in train_idx]

    return train_idx, test_idx


def _assign_value_to_feature(f, df, i, p, file_name):
    """
    Assign the values from individual files to the aligned feature.

    Parameters
    ----------
    f: AlignedFeature
        The aligned feature.
    df: DataFrame
        The feature table from the individual file.
    i: int
        The file index among all files to be aligned.
    p: int
        The row index of the feature in the current individual file.
    file_name: str
        The name of the current file.
    """

    f.feature_id_arr[i] = df.loc[p, "feature_ID"]
    f.mz_arr[i] = df.loc[p, "m/z"]
    f.rt_arr[i] = df.loc[p, "RT"]
    f.peak_height_arr[i] = df.loc[p, "peak_height"]
    f.peak_area_arr[i] = df.loc[p, "peak_area"]
    f.top_average_arr[i] = df.loc[p, "top_average"]
    f.length_arr[i] = df.loc[p, "total_scans"]
    f.gaussian_similarity_arr[i] = df.loc[p, "Gaussian_similarity"]
    f.noise_score_arr[i] = df.loc[p, "noise_score"]
    f.asymmetry_factor_arr[i] = df.loc[p, "asymmetry_factor"]
    f.scan_idx_arr[i] = df.loc[p, "scan_idx"]
    if df.loc[p, "MS2"] == df.loc[p, "MS2"]:
        f.ms2_seq.append([file_name, df.loc[p, "MS2"]])


def _assign_reference_values(f, df, i, p, file_name):
    """
    Assign the reference values to the aligned feature.

    Parameters
    ----------
    f: AlignedFeature
        The aligned feature.
    df: DataFrame
        The feature table from the individual file.
    i: int
        The file index among all files to be aligned.
    p: int
        The row index of the feature in the current individual file.
    file_name: str
        The name of the reference file
    """

    f.mz = df.loc[p, "m/z"]
    f.rt = df.loc[p, "RT"]
    f.reference_file = file_name
    f.reference_scan_idx = df.loc[p, "scan_idx"]
    f.highest_intensity = df.loc[p, "peak_height"]
    f.gaussian_similarity = df.loc[p, "Gaussian_similarity"]
    f.noise_score = df.loc[p, "noise_score"]
    f.asymmetry_factor = df.loc[p, "asymmetry_factor"]