# Author: Hauxu Yu

# A module to define and estimate the parameters

# import modules
import pandas as pd
import os

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
        self.file_names = None              # Absolute paths to the raw files, without extension, list of character strings
        self.sample_groups = None           # Sample groups, list of character strings
        self.sample_group_num = None        # Number of sample groups, integer
        self.sample_dir = None              # Directory for the sample information, character string
        self.single_file_dir = None         # Directory for the single file output, character string
        self.annotation_dir = None          # Directory for the annotation output, character string
        self.chromatogram_dir = None        # Directory for the chromatogram output, character string
        self.network_dir = None             # DirectoTry for the network output, character string
        self.statistics_dir = None          # Directory for the statistical analysis output, character string

        # MS data acquisition
        self.rt_range = [0.0, 1000.0]       # RT range in minutes, list of two floats
        self.ion_mode = "positive"          # Ionization mode, "positive" or "negative", character string

        # Feature detection
        self.mz_tol_ms1 = 0.01              # m/z tolerance for MS1, default is 0.01
        self.mz_tol_ms2 = 0.015             # m/z tolerance for MS2, default is 0.015
        self.int_tol = 1000                 # Intensity tolerance, default is 30000 for Orbitrap and 1000 for other instruments, integer
        self.roi_gap = 50                   # Gap within a feature, default is 2 (i.e. 2 consecutive scans without signal), integer
        self.min_ion_num = 10               # Minimum scan number a feature, default is 10, integer

        # Parameters for feature alignment
        self.align_mz_tol = 0.01            # m/z tolerance for MS1, default is 0.01
        self.align_rt_tol = 0.2             # RT tolerance, default is 0.2

        # Parameters for feature annotation
        self.msms_library = None            # Path to the MS/MS library (.msp or .pickle), character string
        self.ppr = 0.7                      # Peak peak correlation threshold for feature grouping, default is 0.7
        self.ms2_sim_tol = 0.7              # MS2 similarity tolerance, default is 0.7

        # Parameters for normalization
        self.run_normalization = False      # Whether to normalize the data, default is False
        self.normalization_method = "pqn"   # Normalization method, default is "pqn" (probabilistic quotient normalization), character string

        # Parameters for output
        self.output_single_file = False     # Whether to output the processed individual files to a csv file
        self.output_aligned_file = False    # Whether to output aligned features to a csv file

        # Statistical analysis
        self.run_statistics = False         # Whether to perform statistical analysis

        # Network analysis
        self.run_network = False            # Whether to perform network analysis

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
        self.network_dir = os.path.join(self.project_dir, "network")
        self.statistics_dir = os.path.join(self.project_dir, "statistics")
        
        # STEP 2: check if the required files are prepared
        #         three items are required: raw MS data, sample table, parameter file
        if not os.path.exists(self.sample_dir) or len(os.listdir(self.sample_dir)) == 0:
            raise ValueError("No raw MS data is found in the project directory.")
        if not os.path.exists(os.path.join(self.project_dir, "sample_table.csv")):
            raise ValueError("No sample table is found in the project directory.")
        if not os.path.exists(os.path.join(self.project_dir, "parameters.csv")):
            raise ValueError("No parameter file is found in the project directory.")

        # STEP 3: create the output directories if not exist
        if not os.path.exists(self.single_file_dir):
            os.makedirs(self.single_file_dir)
        if not os.path.exists(self.ms2_matching_dir):
            os.makedirs(self.ms2_matching_dir)
        if not os.path.exists(self.bpc_dir):
            os.makedirs(self.bpc_dir)
        if not os.path.exists(self.network_dir):
            os.makedirs(self.network_dir)
        if not os.path.exists(self.statistics_dir):
            os.makedirs(self.statistics_dir)

        # STEP 4: read the sample table and allocate the sample groups
        #         reorder the samples by qc, sample, and blank
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
            sample_table_new = pd.concat([sample_table_new, sample_table[sample_table.iloc[:, 1].str.contains(sample_groups[i])]])
        self.sample_names = list(sample_table_new.iloc[:, 0])
        self.sample_groups = sample_groups
        self.individual_sample_groups = []
        for name in self.sample_names:
            self.individual_sample_groups.append(sample_table_new[sample_table_new.iloc[:, 0] == name].iloc[0, 1])

        # STEP 5: read the parameters from csv file or use default values
        if os.path.exists(os.path.join(self.project_dir, "parameters.csv")):
            self.read_parameters_from_csv(os.path.join(self.project_dir, "parameters.csv"))
        else:
            print("Using default parameters...")
            # determine the type of MS and ion mode
            file_names = os.listdir(self.sample_dir)
            file_names = [f for f in file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
            file_name = os.path.join(self.sample_dir, file_names[0])
            ms_type, ion_mode = find_ms_info(file_name)
            self.set_default(ms_type, ion_mode)
            self.run_statistics = True
            self.plot_bpc = True
            self.plot_ms2 = False

        # STEP 6: set output
        self.output_single_file = True
        self.output_aligned_file = True
    

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

    ms_type = None
    ion_mode = None

    with open(file_name, 'r') as f:
        for line in f:
            if 'orbitrap' in line.lower():
                ms_type = 'orbitrap'
            if 'tof' in line.lower():
                ms_type = 'tof'
            if 'positive' in line.lower():
                ion_mode = 'positive'
            if 'negative' in line.lower():
                ion_mode = 'negative'
            if ms_type is not None and ion_mode is not None:
                break
    return ms_type, ion_mode