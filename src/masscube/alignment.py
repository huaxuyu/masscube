# Author: Hauxu Yu

# A module to align metabolic features from different samples
# Isotopes and in-source fragments are not considered in the alignment

# Import modules
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from .visualization import mirror_ms2_db
from .raw_data_utils import read_raw_file_to_obj


def feature_alignment(path, parameters):
    """
    A function to align the features from individual files.

    Parameters
    ----------------------------------------------------------
    d_list: list
        A list of MS data to be aligned.
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

    # STEP 2: initiate aligned features
    feature_table = _init_feature_table(sample_names=parameters.sample_names)

    mz_tol = parameters.align_mz_tol
    rt_tol = parameters.align_rt_tol
    # STEP 3: read individual feature tables and align features
    last_file_feature_idx = 0
    int_seq = np.zeros(len(feature_table))
    problematic_files = []
    for i, file_name in enumerate(tqdm(parameters.sample_names)):

        # check if the file exists
        if not os.path.exists(txt_file_names[i]):
            problematic_files.append(file_name)
            continue
        
        # read feature table
        current_table = pd.read_csv(txt_file_names[i], low_memory=False, sep="\t")
        current_table = current_table[current_table["MS2"].notna()|(current_table["length"] > 5)]
        # sort current table by peak height from high to low
        current_table = current_table.sort_values(by="peak_height", ascending=False)
        current_table.index = range(len(current_table))
        new_feature_idx = []

        mz_seq = np.array(feature_table["m/z"], dtype=np.float64)
        rt_seq = np.array(feature_table["RT"], dtype=np.float64)
        mz_to_be_matched = np.array(current_table["m/z"], dtype=np.float64)
        rt_to_be_matched = np.array(current_table["RT"], dtype=np.float64)
        labeled_v = np.ones(len(feature_table), dtype=bool)
        
        # compare the m/z and RT of the features
        for j in range(len(current_table)):
            v = np.logical_and(np.abs(mz_seq - mz_to_be_matched[j]) < mz_tol, np.abs(rt_seq - rt_to_be_matched[j]) < rt_tol)
            # check if the feature is already in the aligned feature table using tolerance
            # including parameters.align_mz_tol and parameters.align_rt_tol
            v = np.logical_and(v, labeled_v)
            v = np.where(v)[0]
            # if the feature is already in the aligned feature table, update the feature
            if len(v) > 0:
                if len(v) > 1:
                    idx = v[np.argmax(int_seq[v])]
                else:
                    idx = v[0]
                labeled_v[idx] = False
                feature_table.loc[idx, file_name] = current_table.loc[j, "peak_height"]
                # update the m/z and RT of the feature by averaging the values
                feature_table.loc[idx, "m/z"] = (feature_table.loc[idx, "m/z"]*i + current_table.loc[j, "m/z"])/(i+1)
                feature_table.loc[idx, "RT"] = (feature_table.loc[idx, "RT"]*i + current_table.loc[j, "RT"])/(i+1)
                # if the intensity is the largest, update the reference file
                if current_table.loc[j, "peak_height"] > int_seq[idx]:
                    feature_table.iloc[idx, 3:11] = [current_table.loc[j, "adduct"], current_table.loc[j, "is_isotope"], 
                                                     current_table.loc[j, "is_in_source_fragment"], current_table.loc[j, "Gaussian_similarity"], 
                                                     current_table.loc[j, "noise_level"], current_table.loc[j, "asymmetry_factor"],
                                                     current_table.loc[j, "charge"], current_table.loc[j, "isotopes"]]

                    # only overwrite the MS2 if the current MS2 is not nan
                    if current_table.loc[j, "MS2"] == current_table.loc[j, "MS2"]:
                        feature_table.loc[idx,"MS2"] = current_table.loc[j, "MS2"]
                    feature_table.loc[idx, "alignment_reference"] = file_name
                    int_seq[idx] = current_table.loc[j, "peak_height"]

            # if the feature is not in the aligned feature table, add the feature to the table
            elif len(v) == 0:
                new_feature_idx.append(j)
        
        # add new features to the aligned feature table
        # check if the aligned feature table is full
        if last_file_feature_idx + len(new_feature_idx) > len(feature_table):
            empty_row = pd.DataFrame(np.nan, index=range(len(new_feature_idx)), columns=feature_table.columns)
            feature_table = pd.concat([feature_table, empty_row], ignore_index=True)
            int_seq = np.append(int_seq, np.zeros(len(new_feature_idx)))
        
        new_feature_table = current_table.loc[new_feature_idx]
        new_feature_table.index = range(last_file_feature_idx, last_file_feature_idx+len(new_feature_table))
        a = last_file_feature_idx+len(new_feature_table)-1
        feature_table.loc[last_file_feature_idx:a, "m/z"] = new_feature_table["m/z"]
        feature_table.loc[last_file_feature_idx:a, "RT"] = new_feature_table["RT"]
        feature_table.loc[last_file_feature_idx:a, "adduct"] = new_feature_table["adduct"]
        feature_table.loc[last_file_feature_idx:a, "noise_level"] = new_feature_table["noise_level"]
        feature_table.loc[last_file_feature_idx:a, "Gaussian_similarity"] = new_feature_table["Gaussian_similarity"]
        feature_table.loc[last_file_feature_idx:a, "asymmetry_factor"] = new_feature_table["asymmetry_factor"]
        feature_table.loc[last_file_feature_idx:a, "charge"] = new_feature_table["charge"]
        feature_table.loc[last_file_feature_idx:a, "isotopes"] = new_feature_table["isotopes"]
        feature_table.loc[last_file_feature_idx:a, "MS2"] = new_feature_table["MS2"]
        feature_table.loc[last_file_feature_idx:a, "alignment_reference"] = file_name
        feature_table.loc[last_file_feature_idx:a, "is_isotope"] = new_feature_table["is_isotope"]
        feature_table.loc[last_file_feature_idx:a, "is_in_source_fragment"] = new_feature_table["is_in_source_fragment"]
        feature_table.loc[last_file_feature_idx:a, file_name] = new_feature_table["peak_height"]
        int_seq[last_file_feature_idx:last_file_feature_idx+len(new_feature_table)] = new_feature_table["peak_height"]
        last_file_feature_idx += len(new_feature_table)

    # STEP 4: drop the empty rows and index the feature table
    feature_table.dropna(subset=["m/z"], inplace=True)
    feature_table['ID'] = range(1, len(feature_table)+1)
    # remove the problematic files from parameters.sample_names and parameters.individual_sample_groups
    problematic_idx = [parameters.sample_names.index(f) for f in problematic_files]
    parameters.sample_names = [parameters.sample_names[i] for i in range(len(parameters.sample_names)) if i not in problematic_idx]
    parameters.individual_sample_groups = [parameters.individual_sample_groups[i] for i in range(len(parameters.individual_sample_groups)) if i not in problematic_idx]

    # remove the problematic files from the feature table
    feature_table = feature_table.drop(columns=problematic_files)
    return feature_table


def gap_filling(feature_table, parameters, mode='forced_peak_picking', fill_percetange=0.1):
    """
    A function to fill the gaps in the aligned feature table.

    Parameters
    ----------------------------------------------------------
    feature_table: DataFrame
        The aligned feature table.
    parameters: Params object
        The parameters for gap filling.
    mode: str
        The mode for gap filling.
        'forced_peak_picking': fill the gaps by forced peak picking.
        '0.1_min_intensity': fill the gaps by the minimum intensity * 0.1 (no available yet)

    Returns
    ----------------------------------------------------------
    feature_table: DataFrame
        The aligned feature table with filled gaps.
    """
    # calculate the number of na values in each row
    blank_number = len([x for x in parameters.individual_sample_groups if 'blank' in x])
    total_number = len(parameters.individual_sample_groups)
    if blank_number == 0:
        percent = feature_table.iloc[:,-total_number:].notna().sum(axis=1) / (total_number - blank_number)
    else:
        percent = feature_table.iloc[:,-total_number:-blank_number].notna().sum(axis=1) / (total_number - blank_number)
    feature_table = feature_table[percent > fill_percetange]
    feature_table.index = range(len(feature_table))

    # fill the gaps by forced peak picking
    if mode == 'forced_peak_picking':
        raw_file_names = os.listdir(parameters.sample_dir)
        raw_file_names = [f for f in raw_file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
        raw_file_names = [os.path.join(parameters.sample_dir, f) for f in raw_file_names]

        for file_name in tqdm(parameters.sample_names):
            matched_raw_file_name = [f for f in raw_file_names if file_name in f]
            if len(matched_raw_file_name) == 0:
                print("No raw file found for ", file_name)
                continue
            else:
                matched_raw_file_name = matched_raw_file_name[0]
                d = read_raw_file_to_obj(matched_raw_file_name, int_tol=parameters.int_tol)
                for i in range(len(feature_table)):
                    if pd.isna(feature_table.loc[i, file_name]):
                        _, eic_int, _, _ = d.get_eic_data(feature_table.loc[i, "m/z"], feature_table.loc[i, "RT"], parameters.align_mz_tol, 0.05)
                        feature_table.loc[i, file_name] = np.max(eic_int)
    return feature_table

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

    feature_table.to_csv(output_path, index=False, sep="\t")


class AlignedFeature:
    """
    A class to model an aligned feature from different files.
    """

    def __init__(self):
        """
        Define the attributes of a aligned feature.
        """

        self.id = None                      # index of the feature
        self.mz = 0.0                       # m/z
        self.rt = 0.0                       # retention time
        self.mz_seq = []                    # m/z values from different files
        self.rt_seq = []                    # retention time values from different files
        self.peak_height_seq = []           # peak height from different files
        self.peak_area_seq = []             # peak area from different files 
        self.top_average_seq = []           # top average from different files
        self.ms2_seq = []                   # MS2 from different files
        self.best_ms2 = None                # the best MS2
        self.reference_file = None          # the reference file
        self.highest_roi_intensity = 0.0    # the highest peak height

        # annotation by MS2 matching
        self.annotation = None              # annotated compound name
        self.annotation_mode = None         # 'identity search' or 'hybrid search'
        self.similarity = None              # similarity score (0-1)
        self.matched_peak_number = None     # number of matched peaks
        self.smiles = None                  # SMILES
        self.inchikey = None                # InChIKey
        self.matched_precursor_mz = None    # matched precursor m/z
        self.matched_peaks = None           # matched peaks
        self.formula = None                 # molecular formula

        # isotope, in-source fragment, and adduct information
        self.charge_state = 1               # charge state
        self.isotopes = []                  # isotopes
        self.adduct_type = None             # adduct type

        # statistical analysis
        self.fold_change = None             # fold change
        self.t_test_p = None                # t-test p-value
        self.adjusted_t_test_p = None       # adjusted t-test p-value


    def extend_feat(self, roi, front_zeros=[]):
        """
        A function to extend the feature with a new ROI.

        Parameters
        ----------------------------------------------------------
        roi: ROI object
            The new ROI to be added.
        zeros: list
            A list of zeros to be added to the feature.
        """
        
        if len(self.mz_seq) == 0:
            set_init_mzrt = True
        else:
            set_init_mzrt = False

        if len(front_zeros) > 0:
            self.mz_seq.extend(front_zeros)
            self.rt_seq.extend(front_zeros)
            self.peak_height_seq.extend(front_zeros)
            self.peak_area_seq.extend(front_zeros)
            self.top_average_seq.extend(front_zeros)
            self.ms2_seq.extend([None] * len(front_zeros))

        if roi is not None:
            self.mz_seq.append(roi.mz)
            self.rt_seq.append(roi.rt)
            self.peak_height_seq.append(roi.peak_height)
            self.peak_area_seq.append(roi.peak_area)
            self.top_average_seq.append(roi.top_average)
            self.ms2_seq.append(roi.best_ms2)

            if roi.peak_height > self.highest_roi_intensity:
                self.highest_roi_intensity = roi.peak_height
                self.highest_roi = roi
            
            if set_init_mzrt:
                self.mz = roi.mz
                self.rt = roi.rt

        else:
            self.mz_seq.append(0.0)
            self.rt_seq.append(0.0)
            self.peak_height_seq.append(0.0)
            self.peak_area_seq.append(0.0)
            self.top_average_seq.append(0.0)
            self.ms2_seq.append(None)
    

    def extend_feat_from_a_row(self, row, front_zeros=[]):
        """
        A function to extend the feature with a new ROI.

        Parameters
        ----------------------------------------------------------
        row: DataFrame row
            The row to be added.
        zeros: list
            A list of zeros to be added to the feature.
        """

        if len(self.mz_seq) == 0:
            self.mz = row["m/z"]
            self.rt = row["RT"]

        # for new features, add zeros to the feature
        if len(front_zeros) > 0:
            self.mz_seq.extend(front_zeros)
            self.rt_seq.extend(front_zeros)
            self.peak_height_seq.extend(front_zeros)
            self.peak_area_seq.extend(front_zeros)
            self.top_average_seq.extend(front_zeros)
            self.ms2_seq.extend([None] * len(front_zeros))

        self.mz_seq.append(row["m/z"])
        self.rt_seq.append(row["RT"])
        self.peak_height_seq.append(row["peak_height"])
        self.peak_area_seq.append(row["peak_area"])
        self.top_average_seq.append(row["top_average"])
        self.ms2_seq.append(row["MS2"])

        if row["peak_height"] > self.highest_roi_intensity:
            self.highest_roi_intensity = row["peak_height"]
            self.highest_roi = row

    def choose_best_ms2(self):
        """
        A function to choose the best MS2 for the feature. 
        The best MS2 is the one with the highest summed intensity.
        """

        total_ints = []
        for ms2 in self.ms2_seq:
            if ms2 is not None:
                total_ints.append(np.sum(ms2.peaks[:,1]))
            else:
                total_ints.append(0.0)
        self.best_ms2 = self.ms2_seq[np.argmax(total_ints)]
    

    def plot_match_result(self, output=False):

        if self.matched_peaks is not None:
            mirror_ms2_db(self, output=output)
        else:
            print("No matched MS/MS spectrum found.")
    

    def sum_feature(self, id=None):
        self.id = id
        self.mz_seq = np.array(self.mz_seq)
        self.rt_seq = np.array(self.rt_seq)
        self.mz = np.mean(self.mz_seq[self.mz_seq > 0])
        self.rt = np.mean(self.rt_seq[self.rt_seq > 0])

        self.choose_best_ms2()

        self.charge_state = self.highest_roi.charge_state
        self.is_isotope = self.highest_roi.is_isotope
        self.isotope_mz_seq = self.highest_roi.isotope_mz_seq
        self.isotope_int_seq = self.highest_roi.isotope_int_seq
        self.is_in_source_fragment = self.highest_roi.is_in_source_fragment
        self.adduct_type = self.highest_roi.adduct_type


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