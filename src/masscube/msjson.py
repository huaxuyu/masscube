"""
msjson.py - JSON utilities for MassCube

This module defines a new data format for mass spectrometry data, msjson.

It provides:

1. Structure of the msjson format.
2. Convert the mzML or mzXML file to msjson format.
3. Convert the MSData object to msjson format and vice versa.

msjson design:

1. smaller file size.
2. faster loading and parsing in Python.
3. human and machine readable.
"""

import json
import os

from . import read_raw_file_to_obj
from .params import find_ms_info

def rawdata_to_msjson(raw_data_file: str, msjson_file: str = None, int_tol_ms1: float = None,
                      int_tol_ms2: float = None):
    
    """
    Convert the MSData object to msjson format.

    params
    ------
    raw_data_file: str
        The path to the raw data file.
    msjson_file: str
        The path to the msjson file. If None, the msjson file will 
        be saved in the same directory as the raw data file.
    """

    # check file name
    if not os.path.exists(raw_data_file):
        print(f"Error: {raw_data_file} does not exist.")
        return None
    
    if not msjson_file.lower().endswith(".mzml") or not msjson_file.lower().endswith(".mzxml"):
        print(f"Error: {msjson_file} should be a mzML or mzXML file.")
        return None

    # initialize the msjson file
    msjson = {
        "metadata": {
            "name": None,
            "ion_mode": None,
            "rt_start": None,
            "rt_end": None,
            "instrument": None,
            "instrument_type": None,
            "collison_energy": None,
        },
        "scans": []
    }

    # find the metadata
    ms_type, ion_mode, centroid = find_ms_info(raw_data_file)

    if ms_type == "tof":
        int_tol_ms1 = 1000

    d = read_raw_file_to_obj(raw_data_file)
