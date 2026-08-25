# Author: Huaxu Yu

# A module to define and estimate the parameters

# import modules
import pandas as pd
import os
import json
from importlib.metadata import version
import numpy as np


PROJECT_CONTEXT_FIELDS = frozenset(
    {
        "sample_metadata",
        "project_dir",
        "sample_dir",
        "single_file_dir",
        "tmp_file_dir",
        "ms2_matching_dir",
        "bpc_dir",
        "project_file_dir",
        "normalization_dir",
        "statistics_dir",
        "problematic_files",
    }
)
SINGLE_FILE_METADATA_FIELDS = frozenset(
    {
        "file_name",
        "file_path",
        "file_format",
        "ms_type",
        "ion_mode",
        "is_centroid",
        "scan_time_unit",
    }
)


class Params:
    """
    User-configurable processing parameters.

    File metadata, sample metadata, project paths, and workflow runtime state
    deliberately live on their corresponding data/project objects.
    """

    def __init__(self):
        """
        Function to initiate Params.
        ----------------------------
        """

        # raw data reading and cleaning
        self.mz_lower_limit = 0.0           # lower limit of m/z in Da, float
        self.mz_upper_limit = 100000.0      # upper limit of m/z in Da, float
        self.rt_lower_limit = 0.0           # lower limit of RT in minutes, float
        self.rt_upper_limit = 10000.0       # upper limit of RT in minutes, float
        self.scan_levels = [1,2]            # scan levels to be read, list of integers
        self.centroid_mz_tol = 0.002        # m/z tolerance for centroiding, default is 0.002. set to None to disable centroiding
        self.ms1_abs_int_tol = 1000.0       # absolute intensity threshold for MS1, recommend 30000 for Orbitrap and 1000 for QTOF
        self.ms2_abs_int_tol = 500.0        # absolute intensity threshold for MS2, recommend 10000 for Orbitrap and 500 for QTOF
        self.ms2_rel_int_tol = 0.0          # relative intensity threshold to base peak for MS2, default is 0.01
        self.precursor_mz_offset = None     # offset for MS2 m/z range in Da. The m/z upper limit is precursor_mz - precursor_mz_offset. 
                                            # set to None to disable the m/z upper limit for MS2.

        # feature detection
        self.mz_tol_ms1 = 0.01              # m/z tolerance for MS1, default is 0.01
        self.mz_tol_ms2 = 0.015             # m/z tolerance for MS2, default is 0.015
        self.feature_gap_tol = 10           # gap tolerance within a feature, default is 10 (i.e. 10 consecutive scans without signal), integer
        self.segment_features = True        # whether to segment coeluting peaks within a feature trace
        self.batch_size = 100               # batch size for parallel processing, default is 100, integer
        self.percent_cpu_to_use = 0.8       # percentage of CPU to use, default is 0.8, float
        
        # feature grouping
        self.group_features_single_file = False     # whether to group features in a single file, default is False
        self.scan_scan_cor_tol = 0.9                # scan-to-scan correlation tolerance for feature grouping, default is 0.7
        self.mz_tol_feature_grouping = 0.01         # m/z tolerance for feature grouping, default is 0.01
        self.rt_tol_feature_grouping = 0.05          # RT tolerance for feature grouping, default is 0.1
        self.isotope_rel_int_limit = 1.5            # intensity upper limit of isotopes cannot exceed the base peak intensity * isotope_rel_int_limit, default is 1.5

        # feature alignment
        self.mz_tol_alignment = 0.01                # m/z tolerance for alignment, default is 0.01
        self.rt_tol_alignment = 0.2                 # RT tolerance for alignment, default is 0.2
        self.noise_tol = 2.0                        # noise score tolerance for alignment, default is 2.0
        self.gaussian_similarity_tol = 0.7          # Gaussian similarity tolerance for alignment, default is 0.6
        self.rt_tol_rt_correction = 0.5             # Expected maximum RT shift for RT correction, default is 0.5 minutes
        self.correct_rt = True                      # whether to perform RT correction, default is True
        self.scan_number_cutoff = 5                 # feature with non-zero scan number greater than the cutoff will be aligned, default is 5
        self.detection_rate_cutoff = 0.1            # features detected need to be >rate*(qc+sample), default rate is 0.1
        self.merge_features = True                  # whether to merge features with almost the same m/z and RT, default is True
        self.mz_tol_merge_features = 0.005           # m/z tolerance for merging features, default is 0.005
        self.rt_tol_merge_features = 0.03           # RT tolerance for merging features, default is 0.03
        self.group_features_after_alignment = True  # whether to group features after alignment, default is False
        self.fill_gaps = True                       # whether to fill the gaps in the aligned features, default is True
        self.gap_filling_method = "local_maximum"   # method for gap filling, default is "  local_maximum", string
        self.gap_filling_rt_window = 0.05           # RT window for finding local maximum, default is 0.05 minutes


        # feature annotation
        self.annotate_ms2 = False           # whether to annotate features using an MS2 library
        self.ms2_library_path = None        # path to the MS2 library (.msp or .pickle), character string
        self.fuzzy_search = False           # whether to perform fuzzy search, default is False
        self.consider_rt = False            # whether to consider RT in MS2 matching, default is False.
        self.rt_tol_annotation = 0.2        # RT tolerance for MS2 annotation, default is 0.2
        self.ms2_sim_tol = 0.7              # MS2 similarity tolerance, default is 0.7
        self.spectral_similarity_method = "unweighted_entropy"  # method for spectral similarity calculation
        
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
        self.output_peak_shape = False      # whether to embed feature RT/intensity traces in single-file txt output
        self.quant_method = "peak_height"   # value for quantification and output, "peak_height", "peak_area" or "top_average", string
    

    def read_parameters_from_csv(self, path, ignore_metadata_fields=False):
        """
        Function to read parameters from a csv file.
        --------------------------------------------
        path: character string
            The path to the csv file.
        ignore_metadata_fields: bool
            Ignore recognized legacy metadata rows instead of loading them as
            parameters. Used by project workflows when opening older projects.
        """

        # Read the csv file
        df = pd.read_csv(path)

        # Read the parameters
        for i in range(df.shape[0]):
            name = str(df.iloc[i, 0]).strip()
            if name in SINGLE_FILE_METADATA_FIELDS and ignore_metadata_fields:
                continue
            if (
                name in PROJECT_CONTEXT_FIELDS
                or name in SINGLE_FILE_METADATA_FIELDS
                or not hasattr(self, name)
            ):
                raise ValueError(f"Unknown or non-parameter field in parameter file: {name}")
            value = _coerce_parameter_value(df.iloc[i, 1], getattr(self, name), name)
            setattr(self, name, value)

        # check if the parameters are correct
        self.check_parameters()

    def set_default(self, ms_type, ion_mode=None):
        """
        Set the parameters by the type of MS.
        --------------------------------------
        ms_type: character string
            The type of MS, "orbitrap" or "qtof".
        ion_mode: character string, optional
            Retained only for backward-compatible calls. Ion mode is source
            metadata and does not become a Params field.
        """

        if ms_type == "orbitrap":
            self.ms1_abs_int_tol = 50000
            self.ms2_abs_int_tol = 10000
        else:
            self.ms1_abs_int_tol = 1000
            self.ms2_abs_int_tol = 500
        
        # ion_mode is accepted for backward compatibility. It describes the
        # data source and is stored on MSData.metadata, not in Params.


    def check_parameters(self):
        """
        Check if the parameters are correct using PARAMETER_RAGES.
        ------------------------------------
        """

        # Remove data facts and workflow state retained by Params objects
        # pickled by older MassCube versions.
        for name in SINGLE_FILE_METADATA_FIELDS | PROJECT_CONTEXT_FIELDS:
            self.__dict__.pop(name, None)

        for key, value in PARAMETER_RANGES.items():
            current = getattr(self, key)
            if key == "centroid_mz_tol" and current is None:
                continue
            if not value[0] <= current <= value[1]:
                raise ValueError(
                    f"Parameter {key} must be between {value[0]} and {value[1]}; "
                    f"received {current}."
                )
        self.batch_size = max(1, int(self.batch_size))
        self.feature_gap_tol = int(self.feature_gap_tol)
        self.scan_number_cutoff = int(self.scan_number_cutoff)

        for name in BOOLEAN_PARAMETER_FIELDS:
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"Parameter {name} must be boolean.")

        if sorted(set(int(level) for level in self.scan_levels)) != sorted(self.scan_levels):
            self.scan_levels = sorted(set(int(level) for level in self.scan_levels))
        if not self.scan_levels or any(level not in {1, 2} for level in self.scan_levels):
            raise ValueError("scan_levels must contain one or both of 1 and 2.")
        if self.mz_lower_limit > self.mz_upper_limit:
            raise ValueError("mz_lower_limit cannot exceed mz_upper_limit.")
        if self.rt_lower_limit > self.rt_upper_limit:
            raise ValueError("rt_lower_limit cannot exceed rt_upper_limit.")
        if self.precursor_mz_offset is not None and self.precursor_mz_offset < 0:
            raise ValueError("precursor_mz_offset must be non-negative or None.")
        if self.quant_method not in {"peak_height", "peak_area", "top_average"}:
            raise ValueError(f"Unsupported quant_method: {self.quant_method}")
        if self.gap_filling_method != "local_maximum":
            raise ValueError(f"Unsupported gap_filling_method: {self.gap_filling_method}")
        if self.spectral_similarity_method != "unweighted_entropy":
            raise ValueError(
                "Only spectral_similarity_method='unweighted_entropy' is currently supported."
            )


    def output_parameters(self, path, format="json"):
        """
        Output the parameters to a file.

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

            parameters.update(self.as_dict())
            with open(path, 'w') as f:
                json.dump(parameters, f)
        else:
            raise ValueError("The output format is not supported.")


    def as_dict(self):
        """Return user-configurable parameters without workflow runtime state."""

        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in PROJECT_CONTEXT_FIELDS
            and key not in SINGLE_FILE_METADATA_FIELDS
        }
PARAMETER_RANGES = {
    "mz_lower_limit": (0.0, 100000.0),
    "mz_upper_limit": (0.0, 100000.0),
    "rt_lower_limit": (0.0, 10000.0),
    "rt_upper_limit": (0.0, 10000.0),
    "centroid_mz_tol": (0.0, 0.1),
    "ms1_abs_int_tol": (0, 1e10),
    "ms2_abs_int_tol": (0, 1e10),
    "ms2_rel_int_tol": (0.0, 1.0),
    "mz_tol_ms1": (0.0, 5.0),
    "mz_tol_ms2": (0.0, 5.0),
    "feature_gap_tol": (0, 100),
    "batch_size": (1, 100000),
    "percent_cpu_to_use": (0.01, 1.0),
    "scan_scan_cor_tol": (0.0, 1.0),
    "mz_tol_feature_grouping": (0.0, 5.0),
    "rt_tol_feature_grouping": (0.0, 10.0),
    "isotope_rel_int_limit": (0.0, 100.0),
    "mz_tol_alignment": (0.0, 0.02),
    "rt_tol_alignment": (0.0, 2.0),
    "scan_number_cutoff": (0, 100),
    "detection_rate_cutoff": (0.0, 1.0),
    "mz_tol_merge_features": (0.0, 0.02),
    "rt_tol_merge_features": (0.0, 0.5),
    "ms2_sim_tol": (0.0, 1.0)
}

# Backward-compatible alias for the historical misspelling.
PARAMETER_RAGES = PARAMETER_RANGES

PARAMETER_DEFAULT = {
    "mz_lower_limit": 0.0,
    "mz_upper_limit": 100000.0,
    "rt_lower_limit": 0.0,
    "rt_upper_limit": 10000.0,
    "centroid_mz_tol": 0.002,
    "ms1_abs_int_tol": 1000.0,
    "ms2_abs_int_tol": 500,
    "ms2_rel_int_tol": 0.0,
    "precursor_mz_offset": 2.0,
    "mz_tol_ms1": 0.01,
    "mz_tol_ms2": 0.015,
    "feature_gap_tol": 10,
    "batch_size": 100,
    "percent_cpu_to_use": 0.8,
    "scan_scan_cor_tol": 0.9,
    "mz_tol_feature_grouping": 0.01,
    "rt_tol_feature_grouping": 0.05,
    "isotope_rel_int_limit": 1.5,
    "mz_tol_alignment": 0.01,
    "rt_tol_alignment": 0.2,
    "scan_number_cutoff": 5,
    "detection_rate_cutoff": 0.1,
    "mz_tol_merge_features": 0.005,
    "rt_tol_merge_features": 0.03,
    "ms2_sim_tol": 0.7
}


def _coerce_parameter_value(value, current, name):
    """Convert a CSV value according to the declared parameter's current type."""

    if pd.isna(value):
        return None
    if isinstance(current, bool):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
        if isinstance(value, (int, float)) and value in {0, 1}:
            return bool(value)
        raise ValueError(f"Parameter {name} must be boolean.")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(value))
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        if isinstance(value, str):
            text = value.strip().strip("[]")
            return [int(float(item.strip())) for item in text.split(",") if item.strip()]
        return list(value)
    if current is None:
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        if name == "precursor_mz_offset":
            return float(value)
        return os.fspath(value)
    return type(current)(value)


BOOLEAN_PARAMETER_FIELDS = (
    "segment_features",
    "group_features_single_file",
    "correct_rt",
    "merge_features",
    "group_features_after_alignment",
    "fill_gaps",
    "annotate_ms2",
    "fuzzy_search",
    "consider_rt",
    "sample_normalization",
    "signal_normalization",
    "run_statistics",
    "plot_bpc",
    "plot_ms2",
    "plot_normalization",
    "output_single_file",
    "output_peak_shape",
)
