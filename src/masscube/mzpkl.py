# Author: Huaxu Yu

"""
mzpkl.py - pickle utilities for MassCube

This module defines the pkl format the raw files in MassCube:

1. Structure of the pickle file format.
2. Convert the raw data to pickle file.
"""


# imports
import pickle


def convert_MSData_to_mzpkl(d, output_dir: str = None):
    """
    Convert the MSData object to pickle format.

    Parameters
    ----------
    d: MSData
        The MSData object.
    output_dir: str
        The path to the output directory.
    """

    # more keys can be added to the results dictionary if needed
    results = {
        "name": d.params.file_name,
        "ion_mode": d.params.ion_mode,
        "ms1_time_arr": d.ms1_time_arr,
        "ms1_idx": d.ms1_idx,
        "ms2_idx": d.ms2_idx,
        "scans": d.scans
    }

    if output_dir is not None:
        with open(output_dir, 'wb') as f:
            pickle.dump(results, f)
    else:
        return results


def read_mzpkl_to_MSData(d, file_path: str):
    """
    Read the pickle file to MSData object.

    Parameters
    ----------
    d: MSData
        The MSData object
    file_path: str
        The path to the pickle file.
    """

    with open(file_path, 'rb') as f:
        results = pickle.load(f)

    d.params.file_name = results["name"]
    d.params.ion_mode = results["ion_mode"]
    d.ms1_time_arr = results["ms1_time_arr"]
    d.ms1_idx = results["ms1_idx"]
    d.ms2_idx = results["ms2_idx"]
    d.scans = results["scans"]