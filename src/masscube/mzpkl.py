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
        "name": d.metadata.file_name,
        "ion_mode": d.metadata.ion_mode,
        "metadata": vars(d.metadata).copy(),
        "ms1_time_arr": d.ms1_time_arr,
        "ms1_idx_arr": d.ms1_idx_arr,
        "ms2_idx_arr": d.ms2_idx_arr,
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

    Returns
    -------
    d: MSData
        The populated MSData object.
    """

    with open(file_path, 'rb') as f:
        results = pickle.load(f)

    if not isinstance(results, dict):
        raise ValueError(f"Invalid mzpkl file: {file_path}")

    required_keys = {"ms1_time_arr", "ms1_idx_arr", "ms2_idx_arr", "scans"}
    missing_keys = required_keys.difference(results)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Invalid mzpkl file {file_path}: missing {missing}")

    metadata = results.get("metadata")
    if isinstance(metadata, dict):
        d.update_metadata(metadata)
    else:
        # Backward compatibility with mzpkl files created before the full
        # single-file metadata object was included in the payload.
        d.update_metadata({
            "file_name": results.get("name"),
            "ion_mode": results.get("ion_mode"),
        })

    d.ms1_time_arr = results["ms1_time_arr"]
    d.ms1_idx_arr = results["ms1_idx_arr"]
    d.ms2_idx_arr = results["ms2_idx_arr"]
    d.scans = results["scans"]

    # Older pickles stored a scan's source index as ``id``. Migrate those
    # objects in memory so current code can consistently use raw_file_id.
    for scan in d.scans:
        if getattr(scan, "raw_file_id", None) is None and hasattr(scan, "id"):
            scan.raw_file_id = scan.id

    return d


def raw_data_to_mzpkl(raw_data: str, output_dir: str = None):
    """
    Convert raw data to mzpkl format.

    Parameters
    ----------
    raw_data: 
        The raw MSData object.
    output_dir: str
        The path to the output directory.
    """

    return convert_MSData_to_mzpkl(raw_data, output_dir)
