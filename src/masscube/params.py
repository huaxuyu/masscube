# Author: Huaxu Yu

# A module to define and estimate the parameters

# import modules
import pandas as pd
import os
import json
from importlib.metadata import version
import numpy as np

from .utils_functions import get_start_time


class Params:
    """
    Parameters for the project and individual files.
    """

    def __init__(self):
        """
        Function to initiate Params.
        ----------------------------
        """

        # project
        self.sample_names = None            # sample names without extension, list of strings
        self.sample_abs_paths = None        # absolute paths of the raw MS data, list of strings
        self.sample_metadata = None         # sample metadata, pandas DataFrame
        self.project_dir = None             # project directory, string
        self.sample_dir = None              # directory for the raw MS data, string
        self.single_file_dir = None         # directory for the single file output, string
        self.tmp_file_dir = None            # directory for the intermediate file output, string
        self.ms2_matching_dir = None        # directory for the MS2 matching output, string
        self.bpc_dir = None                 # directory for the base peak chromatogram output, string
        self.project_file_dir = None        # directory for the project files, string
        self.normalization_dir = None       # directory for the normalization output, string
        self.statistics_dir = None          # directory for the statistical analysis output, string
        self.problematic_files = {}         # problematic files, dictionary: {file_name: error_message}

        # raw data reading and cleaning
        self.file_name = None               # file name of the raw data, string
        self.file_path = None               # absolute path of the raw data, string
        self.ion_mode = "positive"          # MS ion mode, "positive" or "negative", string
        self.ms_type = None                 # type of MS, "orbitrap", "qtof", "tripletof" or "others", string
        self.is_centroid = True             # whether the raw data is centroid data, boolean
        self.file_format = None             # file type in lower case, 'mzml', 'mzxml', 'mzjson' or 'mzjson.gz', string
        self.time = None                    # when the data file was acquired, datetime.datetime object
        self.scan_time_unit = "minute"      # time unit of the scan time, "minute" or "second", string
        self.mz_lower_limit = 0.0           # lower limit of m/z in Da, float
        self.mz_upper_limit = 100000.0      # upper limit of m/z in Da, float
        self.rt_lower_limit = 0.0           # lower limit of RT in minutes, float
        self.rt_upper_limit = 10000.0       # upper limit of RT in minutes, float
        self.scan_levels = [1,2]            # scan levels to be read, list of integers
        self.centroid_mz_tol = 0.005        # m/z tolerance for centroiding, default is 0.005. set to None to disable centroiding
        self.ms1_abs_int_tol = 1000.0       # absolute intensity threshold for MS1, recommend 30000 for Orbitrap and 1000 for QTOF
        self.ms2_abs_int_tol = 500          # absolute intensity threshold for MS2, recommend 10000 for Orbitrap and 500 for QTOF
        self.ms2_rel_int_tol = 0.01         # relative intensity threshold to base peak for MS2, default is 0.01
        self.precursor_mz_offset = 2.0      # offset for MS2 m/z range in Da. The m/z upper limit is precursor_mz - precursor_mz_offset.

        # feature detection
        self.mz_tol_ms1 = 0.01              # m/z tolerance for MS1, default is 0.01
        self.mz_tol_ms2 = 0.015             # m/z tolerance for MS2, default is 0.015
        self.feature_gap_tol = 10           # gap tolerance within a feature, default is 10 (i.e. 10 consecutive scans without signal), integer
        self.batch_size = 100               # batch size for parallel processing, default is 100, integer
        self.percent_cpu_to_use = 0.8       # percentage of CPU to use, default is 0.8, float
        
        # feature grouping
        self.group_features_single_file = False     # whether to group features in a single file, default is False
        self.scan_scan_cor_tol = 0.7                # scan-to-scan correlation tolerance for feature grouping, default is 0.7
        self.mz_tol_feature_grouping = 0.015        # m/z tolerance for feature grouping, default is 0.01
        self.rt_tol_feature_grouping = 0.1          # RT tolerance for feature grouping, default is 0.1
        self.valid_charge_states = [1]              # valid charge states for feature grouping, list of integers

        # feature alignment
        self.mz_tol_alignment = 0.01                # m/z tolerance for alignment, default is 0.01
        self.rt_tol_alignment = 0.2                 # RT tolerance for alignment, default is 0.2
        self.rt_tol_rt_correction = 0.5             # Expected maximum RT shift for RT correction, default is 0.5 minutes
        self.correct_rt = True                      # whether to perform RT correction, default is True
        self.scan_number_cutoff = 5                 # feature with non-zero scan number greater than the cutoff will be aligned, default is 5
        self.detection_rate_cutoff = 0.1            # features detected need to be >rate*(qc+sample), default rate is 0.1
        self.merge_features = True                  # whether to merge features with almost the same m/z and RT, default is True
        self.mz_tol_merge_features = 0.01           # m/z tolerance for merging features, default is 0.01
        self.rt_tol_merge_features = 0.02           # RT tolerance for merging features, default is 0.02
        self.group_features_after_alignment = True  # whether to group features after alignment, default is False
        self.fill_gaps = True                       # whether to fill the gaps in the aligned features, default is True
        self.gap_filling_method = "local_maximum"   # method for gap filling, default is "  local_maximum", string
        self.gap_filling_rt_window = 0.05           # RT window for finding local maximum, default is 0.05 minutes
        self.isotope_rel_int_limit = 1.0            # intensity upper limit of isotopes cannot exceed the base peak intensity * isotope_rel_int_limit, default is 1.5

        # feature annotation
        self.ms2_library_path = None        # path to the MS2 library (.msp or .pickle), character string
        self.fuzzy_search = False           # whether to perform fuzzy search, default is False
        self.consider_rt = False            # whether to consider RT in MS2 matching, default is False.
        self.rt_tol_annotation = 0.2        # RT tolerance for MS2 annotation, default is 0.2
        self.ms2_sim_tol = 0.7              # MS2 similarity tolerance, default is 0.7
        
        # normalization
        self.sample_normalization = False   # whether to normalize the data based on total sample amount/concentration, default is False
        self.sample_norm_method = "pqn"     # sample normalization method, default is "pqn" (probabilistic quotient normalization), character string
        self.signal_normalization = False   # whether to run feature-wised normalization to correct systematic signal drift, default is False
        self.signal_norm_method = "lowess"  # normalization method for signal drift, default is "loess" (local polynomial regression fitting), character string

        # statistical analysis
        self.run_statistics = False         # whether to perform statistical analysis

        # visualization
        self.plot_bpc = False               # whether to plot base peak chromatograms
        self.plot_ms2 = False               # whether to plot mirror plots for MS2 matching
        self.plot_normalization = False     # whether to plot the normalization results

        # classifier building
        self.by_group_name = None           # only used for building classification model: group name for classifier building, string

        # output
        self.output_single_file = False     # whether to output the processed individual files to a csv file
        self.output_ms1_scans = False       # whether to output all MS1 scans to a pickle file for faster data reloading (only used in untargted metabolomics workflow)
        self.output_aligned_file = False    # whether to output aligned features to a csv file
        self.quant_method = "peak_height"   # value for quantification and output, "peak_height", "peak_area" or "top_average", string
    

    def read_parameters_from_csv(self, path):
        """
        Function to read parameters from a csv file.
        --------------------------------------------
        path: character string
            The path to the csv file.
        """

        # Read the csv file
        df = pd.read_csv(path)

        # Read the parameters
        for i in range(df.shape[0]):
            # check if the value can be converted to a float, if yes, convert it to a float
            try:
                value = float(df.iloc[i, 1])
            except:
                value = df.iloc[i, 1]
                if value.lower() == "true" or value.lower() == "yes":
                    value = True
                elif value.lower() == "false" or value.lower() == "no":
                    value = False
            setattr(self, df.iloc[i, 0], value)

        # check if the parameters are correct
        self.check_parameters()


    def read_sample_metadata(self, path):
        """
        Read the sample metadata from a csv file.
        
        Parameters
        ----------
        path : str
            The path to the csv file.
        """

        df = pd.read_csv(path)
        self.sample_names = list(df.iloc[:, 0])
        df.columns = [col.lower() if col.lower() in ['is_qc', 'is_blank'] else col for col in df.columns]

        if 'is_qc' in df.columns and type(df['is_qc'][0]) == str:
            df['is_qc'] = df['is_qc'].apply(lambda x: True if x.lower() == 'yes' else False)
        else:
            df['is_qc'] = False
        if 'is_blank' in df.columns and type(df['is_blank'][0]) == str:
            df['is_blank'] = df['is_blank'].apply(lambda x: True if x.lower() == 'yes' else False)
        else:
            df['is_blank'] = False
        
        # move all qc samples to the front and all blank samples to the end
        df = df.sort_values(by=['is_qc', 'is_blank'], ascending=[False, True])
        df = df.reset_index(drop=True)
        
        self.sample_metadata = df


    def _untargeted_metabolomics_workflow_preparation(self):
        """
        Prepare the parameters for the untargeted metabolomics workflow.
        """
    
        # STEP 1: check if the project directory exists
        if not os.path.exists(self.project_dir):
            raise ValueError("The project directory does not exist. Please create the directory first.")
        
        self.sample_dir = os.path.join(self.project_dir, "data")
        self.single_file_dir = os.path.join(self.project_dir, "single_files")
        self.tmp_file_dir = os.path.join(self.project_dir, "tmp")
        self.ms2_matching_dir = os.path.join(self.project_dir, "ms2_matching")
        self.bpc_dir = os.path.join(self.project_dir, "chromatograms")
        self.project_file_dir = os.path.join(self.project_dir, "project_files")
        self.statistics_dir = os.path.join(self.project_dir, "statistical_analysis")
        self.normalization_dir = os.path.join(self.project_dir, "normalization_results")
        
        # STEP 2: check if the required files are prepared
        #         three items are required: raw MS data, sample table and parameter file
        if not os.path.exists(self.sample_dir) or len(os.listdir(self.sample_dir)) == 0:
            raise ValueError("No raw MS data is found in the project directory.")
        if not os.path.exists(os.path.join(self.project_dir, "sample_table.csv")):
            print("No sample table is found in the project directory. Normalization and statistical analysis will NOT be performed.")
            self.run_statistics = False
            self.sample_normalization = False
            self.signal_normalization = False
        if not os.path.exists(os.path.join(self.project_dir, "parameters.csv")):
            print("No parameter file is found in the project directory. Default parameters will be used.")
            print("To perform feature annotation, please specify the path of MS/MS library in the parameter file.")

        # STEP 3: create the output directories if not exist
        if not os.path.exists(self.single_file_dir):
            os.makedirs(self.single_file_dir)
        if not os.path.exists(self.tmp_file_dir):
            os.makedirs(self.tmp_file_dir)
        if not os.path.exists(self.ms2_matching_dir):
            os.makedirs(self.ms2_matching_dir)
        if not os.path.exists(self.bpc_dir):
            os.makedirs(self.bpc_dir)
        if not os.path.exists(self.project_file_dir):
            os.makedirs(self.project_file_dir)
        
        # STEP 4: read the parameters from csv file or use default values
        if os.path.exists(os.path.join(self.project_dir, "parameters.csv")):
            self.read_parameters_from_csv(os.path.join(self.project_dir, "parameters.csv"))
        else:
            print("Using default parameters...")
            # determine the type of MS and ion mode
            file_names = os.listdir(self.sample_dir)
            file_names = [f for f in file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
            file_name = os.path.join(self.sample_dir, file_names[0])
            ms_type, ion_mode, _ = find_ms_info(file_name)
            self.set_default(ms_type, ion_mode)
            self.plot_bpc = True
        
        if not os.path.exists(self.statistics_dir) and self.run_statistics:
            os.makedirs(self.statistics_dir)
        if not os.path.exists(self.normalization_dir) and (self.sample_normalization or self.signal_normalization):
            os.makedirs(self.normalization_dir)

        # STEP 5: read the sample names and sample metadata from the sample table
        if os.path.exists(os.path.join(self.project_dir, "sample_table.csv")):
            self.read_sample_metadata(os.path.join(self.project_dir, "sample_table.csv"))
            # find the absolute paths of the raw MS data in order
            self.sample_abs_paths, good_idx = _find_files_in_data_folder(self.sample_dir, list(self.sample_metadata.iloc[:, 0]))
            self.sample_metadata = self.sample_metadata.iloc[good_idx, :]
            self.sample_names = list(self.sample_metadata.iloc[:, 0])

            # find the start time of the raw MS data
            self.sample_metadata['time'] = [get_start_time(path) for path in self.sample_abs_paths]
            self.sample_metadata['analytical_order'] = 0
            for rank, idx in enumerate(np.argsort(self.sample_metadata['time'])):
                self.sample_metadata.loc[idx, 'analytical_order'] = rank + 1
        else:
            self.sample_names = [f for f in os.listdir(self.sample_dir) if not f.startswith(".") and 
                                 (f.lower().endswith(".mzml") or f.lower().endswith(".mzxml"))]
            self.sample_abs_paths = [os.path.join(self.sample_dir, f) for f in self.sample_names]
            self.sample_names = [f.split(".")[0] for f in self.sample_names]
            self.sample_metadata = pd.DataFrame({'sample_name': self.sample_names, 'is_qc': False, 'is_blank': False})

        # STEP 6: set output
        self.output_single_file = True      # output the processed individual files to a txt file
        self.output_ms1_scans = True        # for faster data reloading in gap filling
        self.output_aligned_file = True     # output the aligned features to a txt file


    def set_default(self, ms_type, ion_mode):
        """
        Set the parameters by the type of MS.
        --------------------------------------
        ms_type: character string
            The type of MS, "orbitrap" or "qtof".
        ion_mode: character string
            The ionization mode, "positive" or "negative".
        """

        if ms_type == "orbitrap":
            self.ms1_abs_int_tol = 30000
            self.ms2_abs_int_tol = 10000
        else:
            self.ms1_abs_int_tol = 1000
            self.ms2_abs_int_tol = 500
        
        self.ion_mode = ion_mode
    

    def check_parameters(self):
        """
        Check if the parameters are correct using PARAMETER_RAGEES.
        ------------------------------------
        """

        for key, value in PARAMETER_RAGES.items():
            if not value[0] <= getattr(self, key) <= value[1]:
                print(f"Parameter {key} is not out of range. The value is set to the default value.")
                setattr(self, key, PARAMETER_DEFAULT[key])
        if not os.path.exists(str(self.ms2_library_path)):
            self.ms2_library_path = None
        self.batch_size = int(self.batch_size)


    def output_parameters(self, path, format="json"):
        """
        Output the parameters to a file.
        ---------------------------------

        Parameters
        ----------
        path : str
            The path to the output file.
        format : str
            The format of the output file. "json" is only supported for now. 
        """

        if format == "json":
            parameters = {}
            # obtain the version of the package
            parameters["MassCube_version"] = version("masscube")

            for key, value in self.__dict__.items():
                if key != "project_dir":
                    parameters[key] = value
            with open(path, 'w') as f:
                json.dump(parameters, f)
        else:
            raise ValueError("The output format is not supported.")


def find_ms_info(file_name):
    """
    Find the type of MS and ion mode from the raw file.

    Parameters
    ----------
    file_name : str
        The file name of the raw file.

    Returns
    -------
    ms_type : str
        The type of MS, "orbitrap", "qtof", "tripletof" or "others".
    ion_mode : str
        The ion mode, "positive" or "negative".
    centroid : bool
        Whether the data is centroid data.
    """

    ms_type = None
    ion_mode = None
    centroid = False

    # for mzml and mzxml
    if file_name.lower().endswith('.mzml') or file_name.lower().endswith('.mzxml'):
        text = ""
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                text += line
                if i > 200:
                    break
        text = text.lower()
        if 'orbitrap' in text or 'q exactive' in text:
            ms_type = 'orbitrap'
        elif 'tripletof' in text:
            ms_type = 'tripletof'
        elif 'tof' in text:
            ms_type = 'qtof'
        
        if 'positive' in text:
            ion_mode = 'positive'
        elif 'negative' in text:
            ion_mode = 'negative'

        if "centroid spectrum" in text or 'centroided="1"' in text:
            centroid = True

    return ms_type, ion_mode, centroid


def _find_files_in_data_folder(folder_path, filenames):
    found_files = []
    
    # Get a set of all files in the folder (without extensions)
    folder_files = {os.path.splitext(f)[0]: f for f in os.listdir(folder_path) if not f.startswith(".") and
                    (f.lower().endswith(".mzml") or f.lower().endswith(".mzxml"))}
    
    good_idx = []
    for i, name in enumerate(filenames):
        if name in folder_files:
            found_files.append(os.path.abspath(os.path.join(folder_path, folder_files[name])))
            good_idx.append(i)
    
    return found_files, good_idx


PARAMETER_RAGES = {
    "mz_lower_limit": (0.0, 100000.0),
    "mz_upper_limit": (0.0, 100000.0),
    "rt_lower_limit": (0.0, 10000.0),
    "rt_upper_limit": (0.0, 10000.0),
    "centroid_mz_tol": (0.0, 0.1),
    "ms1_abs_int_tol": (0, 1e10),
    "ms2_abs_int_tol": (0, 1e10),
    "ms2_rel_int_tol": (0.0, 1.0),
    "precursor_mz_offset": (0.0, 100000.0),
    "mz_tol_ms1": (0.0, 0.02),
    "mz_tol_ms2": (0.0, 0.02),
    "feature_gap_tol": (0, 100),
    "scan_scan_cor_tol": (0.0, 1.0),
    "mz_tol_alignment": (0.0, 0.02),
    "rt_tol_alignment": (0.0, 2.0),
    "scan_number_cutoff": (0, 100),
    "detection_rate_cutoff": (0.0, 1.0),
    "mz_tol_merge_features": (0.0, 0.02),
    "rt_tol_merge_features": (0.0, 0.5),
    "ms2_sim_tol": (0.0, 1.0)
}

PARAMETER_DEFAULT = {
    "mz_lower_limit": 0.0,
    "mz_upper_limit": 100000.0,
    "rt_lower_limit": 0.0,
    "rt_upper_limit": 10000.0,
    "centroid_mz_tol": 0.005,
    "ms1_abs_int_tol": 1000.0,
    "ms2_abs_int_tol": 500,
    "ms2_rel_int_tol": 0.01,
    "precursor_mz_offset": 2.0,
    "mz_tol_ms1": 0.01,
    "mz_tol_ms2": 0.015,
    "feature_gap_tol": 30,
    "scan_scan_cor_tol": 0.7,
    "mz_tol_alignment": 0.01,
    "rt_tol_alignment": 0.2,
    "scan_number_cutoff": 5,
    "detection_rate_cutoff": 0.1,
    "mz_tol_merge_features": 0.01,
    "rt_tol_merge_features": 0.05,
    "ms2_sim_tol": 0.7
}