# Author: Hauxu Yu

# A module to define and estimate the parameters

# import modules
import pandas as pd
import os
import json
import gzip
from importlib.metadata import version

# Define a class to store the parameters
class Params:
    """
    Parameters for the project and individual files.
    """

    def __init__(self):
        """
        Function to initiate Params.
        ----------------------------
        """

        # The project
        self.project_dir = None             # Project directory, character string
        self.sample_names = None            # Absolute paths to the raw files, without extension, list of character strings
        self.sample_groups = None           # Sample groups, list of character strings
        self.sample_group_num = None        # Number of sample groups, integer
        self.sample_dir = None              # Directory for the sample information, character string
        self.single_file_dir = None         # Directory for the single file output, character string
        self.annotation_dir = None          # Directory for the annotation output, character string
        self.chromatogram_dir = None        # Directory for the chromatogram output, character string
        # self.network_dir = None             # Directory for the network output, character string
        self.statistics_dir = None          # Directory for the statistical analysis output, character string

        # MS data acquisition
        self.rt_range = [0.0, 1000.0]       # RT range in minutes, list of two floats
        self.ion_mode = "positive"          # Ionization mode, "positive" or "negative", character string

        # Feature detection
        self.mz_tol_ms1 = 0.01              # m/z tolerance for MS1, default is 0.01
        self.mz_tol_ms2 = 0.015             # m/z tolerance for MS2, default is 0.015
        self.int_tol = None                   # Intensity tolerance, recommend 30000 for Orbitrap and 1000 for QTOF, integer
        self.roi_gap = 30                   # Gap within a feature, default is 30 (i.e. 30 consecutive scans without signal), integer
        self.ppr = 0.7                      # Peak peak correlation threshold for feature grouping, default is 0.7

        # Parameters for feature alignment
        self.align_mz_tol = 0.01            # m/z tolerance for MS1, default is 0.01
        self.align_rt_tol = 0.2             # RT tolerance, default is 0.2
        self.run_rt_correction = True       # Whether to perform RT correction, default is True
        self.min_scan_num_for_alignment = 6    # Minimum scan number a feature to be aligned, default is 6
        self.clean_feature_table = True     # Whether to clean the feature table, default is True

        # Parameters for feature annotation
        self.msms_library = None            # Path to the MS/MS library (.msp or .pickle), character string
        self.ms2_sim_tol = 0.7              # MS2 similarity tolerance, default is 0.7

        # Parameters for normalization
        self.run_normalization = False      # Whether to normalize the data, default is False
        self.normalization_method = "pqn"   # Normalization method, default is "pqn" (probabilistic quotient normalization), character string

        # Parameters for output
        self.output_single_file = False     # Whether to output the processed individual files to a csv file
        self.output_aligned_file = False    # Whether to output aligned features to a csv file

        # Statistical analysis
        self.run_statistics = False         # Whether to perform statistical analysis

        # # Network analysis
        # self.run_network = False            # Whether to perform network analysis

        # Visualization
        self.plot_bpc = False               # Whether to plot base peak chromatogram
        self.plot_ms2 = False               # Whether to plot mirror plots for MS2 matching
    

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
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False

            setattr(self, df.iloc[i, 0], value)
        
        self.rt_range = [self.rt_start, self.rt_end]

        # check if the parameters are correct
        self.check_parameters()
    

    def _untargeted_metabolomics_workflow_preparation(self):
        """
        Prepare the parameters for the untargeted metabolomics workflow.
        """
    
        # STEP 1: check if the project directory exists
        if not os.path.exists(self.project_dir):
            raise ValueError("The project directory does not exist. Please create the directory first.")
        
        self.sample_dir = os.path.join(self.project_dir, "data")
        self.single_file_dir = os.path.join(self.project_dir, "single_files")
        self.ms2_matching_dir = os.path.join(self.project_dir, "ms2_matching")
        self.bpc_dir = os.path.join(self.project_dir, "chromatogram")
        # self.network_dir = os.path.join(self.project_dir, "network")
        self.statistics_dir = os.path.join(self.project_dir, "statistics")
        
        # STEP 2: check if the required files are prepared
        #         three items are required: raw MS data, sample table, parameter file
        if not os.path.exists(self.sample_dir) or len(os.listdir(self.sample_dir)) == 0:
            raise ValueError("No raw MS data is found in the project directory.")
        if not os.path.exists(os.path.join(self.project_dir, "sample_table.csv")):
            print("No sample table is found in the project directory. No statistical analysis and sample normalization will be performed.")
        if not os.path.exists(os.path.join(self.project_dir, "parameters.csv")):
            print("No parameter file is found in the project directory. Default parameters will be used.")
            print("To perform feature annotation, please specify the path of MS/MS library in the parameter file.")

        # STEP 3: create the output directories if not exist
        if not os.path.exists(self.single_file_dir):
            os.makedirs(self.single_file_dir)
        if not os.path.exists(self.ms2_matching_dir):
            os.makedirs(self.ms2_matching_dir)
        if not os.path.exists(self.bpc_dir):
            os.makedirs(self.bpc_dir)
        # if not os.path.exists(self.network_dir):
        #     os.makedirs(self.network_dir)
        if not os.path.exists(self.statistics_dir):
            os.makedirs(self.statistics_dir)
        
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
            self.run_statistics = True
            self.plot_bpc = True
            # annotation will not be performed if no MS/MS library is provided
            self.plot_ms2 = False

        # STEP 5: read the sample table and allocate the sample groups
        #         reorder the samples by qc, sample, and blank
        if not os.path.exists(os.path.join(self.project_dir, "sample_table.csv")):
            self.sample_names = [f for f in os.listdir(self.sample_dir) if not f.startswith(".")] # for macOS
            self.sample_names = [f for f in self.sample_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
            self.sample_names = [f.split(".")[0] for f in self.sample_names]
            self.sample_groups = ["sample"] * len(self.sample_names)
            self.individual_sample_groups = ["sample"] * len(self.sample_names)
            self.sample_group_num = 1
            # skip the normalization and statistics if no sample table is provided
            self.run_normalization = False
            self.run_statistics = False
        else:
            sample_table = pd.read_csv(os.path.join(self.project_dir, "sample_table.csv"))
            try:
                sample_table.iloc[:, 1] = sample_table.iloc[:, 1].str.lower()
            except:
                raise ValueError("The second column of the sample table is not correct.")
            sample_groups_pre = list(set(sample_table.iloc[:, 1]))
            sample_groups = [i for i in sample_groups_pre if i not in ["qc", "blank"]]
            self.sample_group_num = len(sample_groups)
            if "qc" in sample_groups_pre:
                sample_groups = ["qc"] + sample_groups
            if "blank" in sample_groups_pre:
                sample_groups = sample_groups + ["blank"]

            sample_table_new = pd.DataFrame(columns=sample_table.columns)
            for i in range(len(sample_groups)):
                sample_table_new = pd.concat([sample_table_new, sample_table[sample_table.iloc[:, 1].str.lower() == sample_groups[i]]])
            self.sample_names = list(sample_table_new.iloc[:, 0])
            self.sample_groups = sample_groups
            self.individual_sample_groups = []
            for name in self.sample_names:
                self.individual_sample_groups.append(sample_table_new[sample_table_new.iloc[:, 0] == name].iloc[0, 1])

        # STEP 6: set output
        self.output_single_file = True
        self.output_aligned_file = True


    def _batch_processing_preparation(self):
        """
        Prepare the parameters for the batch processing.
        """
    
        # STEP 1: check if the project directory exists
        if not os.path.exists(self.project_dir):
            raise ValueError("The project directory does not exist. Please create the directory first.")
        
        self.sample_dir = os.path.join(self.project_dir, "data")
        self.single_file_dir = os.path.join(self.project_dir, "single_files")
        self.bpc_dir = os.path.join(self.project_dir, "chromatogram")

        # STEP 2: check if the required files are prepared
        #         three items are required: raw MS data, sample table, parameter file
        if not os.path.exists(self.sample_dir) or len(os.listdir(self.sample_dir)) == 0:
            raise ValueError("No raw MS data is found in the project directory.")
        if not os.path.exists(os.path.join(self.project_dir, "parameters.csv")):
            print("No parameter file is found in the project directory. Default parameters will be used.")

        # STEP 3: create the output directories if not exist
        if not os.path.exists(self.single_file_dir):
            os.makedirs(self.single_file_dir)
        if not os.path.exists(self.bpc_dir):
            os.makedirs(self.bpc_dir)
        
        # STEP 4: read the parameters from csv file or use default values
        if os.path.exists(os.path.join(self.project_dir, "parameters.csv")):
            self.read_parameters_from_csv(os.path.join(self.project_dir, "parameters.csv"))
        else:
            print("Using default parameters...")
            self.plot_bpc = False

        self.output_single_file = True


    def set_default(self, ms_type, ion_mode):
        """
        Set the parameters by the type of MS.
        --------------------------------------
        ms_type: character string
            The type of MS, "orbitrap" or "tof".
        ion_mode: character string
            The ionization mode, "positive" or "negative".
        """

        if ms_type == "orbitrap":
            self.int_tol = 30000
        elif ms_type == "tof":
            self.int_tol = 1000
        if ion_mode == "positive":
            self.ion_mode = "positive"
        elif ion_mode == "negative":
            self.ion_mode = "negative"
    

    def check_parameters(self):
        """
        Check if the parameters are correct using PARAMETER_RAGEES.
        ------------------------------------
        """

        for key, value in PARAMETER_RAGES.items():
            if not value[0] <= getattr(self, key) <= value[1]:
                print(f"Parameter {key} is not out of range. The value is set to the default value.")
                setattr(self, key, PARAMETER_DEFAULT[key])
        
        if self.msms_library != self.msms_library:
            self.msms_library = None
    

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
        The type of MS.
    ion_mode : str
        The ion mode.
    """

    ms_type = 'tof'
    ion_mode = 'positive'
    centroid = False

    ext = os.path.splitext(file_name)[1].lower()

    # if mzML of mzXML
    if ext == '.mzml' or ext == '.mzxml':
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                if 'orbitrap' in line.lower():
                    ms_type = 'orbitrap'
                if 'negative' in line.lower():
                    ion_mode = 'negative'
                if "centroid spectrum" in line.lower() or 'centroided="1"' in line.lower():
                    centroid = True
                if i > 200:
                    break
    
    # if mzjson or compressed mzjson
    elif ext == '.mzjson' or ext == '.gz':
        if ext.lower() == ".mzjson":
            with open(file_name, 'r') as f:
                data = json.load(f)
        else:
            with gzip.open(file_name, 'rt') as f:
                data = json.load(f)
        ms_type = data["metadata"]["instrument_type"]
        ion_mode = data["metadata"]["ion_mode"]
        if "centroid" in data["metadata"]:
            centroid = data["metadata"]["centroid"]
        else:
            centroid = True

    return ms_type, ion_mode, centroid


PARAMETER_RAGES = {
    "rt_start": [0.0, 1000.0],
    "rt_end": [0.0, 1000.0],
    "mz_tol_ms1": [0.0, 0.02],
    "mz_tol_ms2": [0.0, 0.02],
    "int_tol": [0, 1e10],
    "roi_gap": [0, 50],
    "min_scan_num_for_alignment": [0, 50],
    "align_mz_tol": [0.0, 0.02],
    "align_rt_tol": [0.0, 2.0],
    "ppr": [0.5, 1.0],
    "ms2_sim_tol": [0.0, 1.0]
}

PARAMETER_DEFAULT = {
    "rt_start": 0.0,
    "rt_end": 1000.0,
    "mz_tol_ms1": 0.01,
    "mz_tol_ms2": 0.015,
    "int_tol": 30000,
    "roi_gap": 10,
    "min_scan_num_for_alignment": 5,
    "align_mz_tol": 0.01,
    "align_rt_tol": 0.2,
    "ppr": 0.7,
    "ms2_sim_tol": 0.7
}