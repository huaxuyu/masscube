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
import gzip

from . import read_raw_file_to_obj
from .params import find_ms_info

def rawdata_to_msjson(raw_data_file: str, msjson_file: str = None, int_tol_ms1: float = None,
                      int_tol_ms2: float = None, compression: bool = True):
    
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
    
    if not raw_data_file.lower().endswith(".mzml") and not raw_data_file.lower().endswith(".mzxml"):
        print(f"Error: {raw_data_file} should be a mzML or mzXML file.")
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

    if not centroid:
        print(f"Error: {raw_data_file} is not centroided.")
        return None

    if ms_type == "tof":
        int_tol_ms1 = 500
    elif ms_type == "orbitrap":
        int_tol_ms1 = 10000

    d = read_raw_file_to_obj(raw_data_file, int_tol = int_tol_ms1)

    msjson["metadata"]["name"] = d.file_name
    msjson["metadata"]["ion_mode"] = ion_mode
    msjson["metadata"]["rt_start"] = d.ms1_rt_seq[0]
    msjson["metadata"]["rt_end"] = d.ms1_rt_seq[-1]
    msjson["metadata"]["instrument_type"] = ms_type
    
    for i in d.scans:
        if i.level == 1:
            msjson["scans"].append({
                "level": i.level,
                "time": i.rt,
                "mz": list(i.mz_seq),
                "intensity": list(i.int_seq),
            })
        if i.level == 2:
            msjson["scans"].append({
                "level": i.level,
                "time": i.rt,
                "mz": list(i.peaks[:,0]),
                "intensity": list(i.peaks[:,1]),
                "precursor_mz": round(i.precursor_mz, 4)
            })
    for i in msjson["scans"]:
        i["mz"] = [round(x, 4) for x in i["mz"]]
        i["intensity"] = [f"{x:.3e}" for x in i["intensity"]]

    # compress the msjson file
    if compression:
        if msjson_file is None:
            msjson_file = os.path.join(os.path.dirname(raw_data_file), os.path.basename(raw_data_file).split(".")[0] + ".msjson.gz")
        with gzip.open(msjson_file, "wt") as f:
            json.dump(msjson, f)
    else:
        if msjson_file is None:
            msjson_file = os.path.join(os.path.dirname(raw_data_file), os.path.basename(raw_data_file).split(".")[0] + ".msjson")
        with open(msjson_file, "w") as f:
            json.dump(msjson, f)