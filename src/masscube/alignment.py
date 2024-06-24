# Author: Hauxu Yu

# A module to align metabolic features from different samples
# Isotopes and in-source fragments are not considered in the alignment

# Import modules
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from .raw_data_utils import read_raw_file_to_obj
from time import time

def feature_alignment(path, parameters):
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
    
    # STEP 3: read individual feature tables and align features
    for i, file_name in enumerate(tqdm(parameters.sample_names)):
        # check if the file exists, if not, add the file to problematic_files
        if not os.path.exists(txt_file_names[i]):
            file_quality_arr[i] = False
            continue

        # read feature table
        current_table = pd.read_csv(txt_file_names[i], low_memory=False, sep="\t")
        current_table = current_table[current_table["MS2"].notna()|(current_table["length"] > 5)]
        # sort current table by peak height from high to low
        current_table = current_table.sort_values(by="peak_height", ascending=False)
        current_table.index = range(len(current_table))
        avail_roi = np.ones(len(current_table), dtype=bool)

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
    features = [f for f in features if (np.sum(f.detected_seq)-blank_num) > 0.1*(len(parameters.sample_names)-blank_num)]

    for i in range(len(features)):
        features[i].id = i

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

        for i, file_name in enumerate(tqdm(parameters.sample_names)):
            matched_raw_file_name = [f for f in raw_file_names if file_name in f]
            if len(matched_raw_file_name) == 0:
                continue
            else:
                matched_raw_file_name = matched_raw_file_name[0]
                d = read_raw_file_to_obj(matched_raw_file_name, int_tol=parameters.int_tol, read_ms2=False)
                for f in features:
                    if not f.detected_seq[i]:
                        _, eic_int, _, _ = d.get_eic_data(f.mz, f.rt, parameters.align_mz_tol, 0.05)
                        f.peak_height_seq[i] = np.max(eic_int)
    
    # calculate the fill percentage after gap filling (blank samples are not included)
    blank_num = len([x for x in parameters.individual_sample_groups if 'blank' in x])
    for f in features:
        f.fill_percentage = (np.sum(f.detected_seq)-blank_num) / (len(parameters.sample_names)-blank_num) * 100
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
    feature_table['similarity'] = feature_table['similarity'].apply(lambda x: round(x, 4))

    feature_table.to_csv(output_path, index=False, sep="\t")


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
        self.mz_seq = np.zeros(file_number)         # m/z values from individual files
        self.rt_seq = np.zeros(file_number)         # retention time values from individual files
        self.peak_height_seq = np.zeros(file_number)    # peak height from individual files
        self.peak_area_seq = np.zeros(file_number)  # peak area from individual files 
        self.ms2_seq = []                           # best MS2 from individual files
        self.detected_seq = np.zeros(file_number, dtype=bool)   # whether the feature is detected in individual files

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


# generate an empty feature table with 100000 rows
def _init_feature_table(rows=5000, sample_names=[]):
    tmp = pd.DataFrame(
        columns=["ID", "m/z", "RT", "adduct", "is_isotope", "is_in_source_fragment", "Gaussian_similarity", "noise_level", "asymmetry_factor",
                 "charge", "isotopes", "MS2", 
                "matched_MS2", "search_mode", "annotation", "formula", "similarity", "matched_peak_number", "SMILES", "InChIKey", 
                "fill_percentage", "alignment_reference"] + sample_names,
        index=range(rows)
    )
    return tmp