# Author: Huaxu Yu

# A module for utility functions

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
import re
from collections import Counter


def generate_sample_table(path=None, output=True):
    """
    Generate a sample table from the mzML or mzXML files in the specified path.
    The stucture of the path should be:
    path
    ├── data
    │   ├── sample1.mzml
    │   ├── sample2.mzml
    │   └── ...
    └── ...

    Parameters
    ----------
    path : str
        Path to the main directory that contains a subdirectory 'data' with mzML or mzXML files.
    output : bool
        If True, output the sample table to a csv file.

    Return
    ------
    sample_table : pandas DataFrame
        A DataFrame with two columns: 'Sample' and 'Groups'.

    Output
    ------
    sample_table.csv : csv file
        A csv file with two columns: 'Sample' and 'Groups' in the specified path.
    """

    # if path is not specified, use the current working directory
    if path is None:
        path = os.getcwd()
    
    path_data = os.path.join(path, 'data')

    if os.path.exists(path_data):
        file_names = [os.path.splitext(f)[0] for f in os.listdir(path_data) if f.lower().endswith('.mzml') or f.lower().endswith('.mzxml')]
        file_names = [f for f in file_names if not f.startswith(".")]   # for Mac OS
        file_names = sorted(file_names)
        sample_table = pd.DataFrame({'sample_name': file_names, "is_qc": [None]*len(file_names), "is_blank": [None]*len(file_names)})
    else:
        raise FileNotFoundError(f"The path {path_data} does not exist.")
    
    if output:
        sample_table.to_csv(os.path.join(path, 'sample_table.csv'), index=False)
        return None
    else:
        return sample_table


def get_timestamps(path=None, output=True):
    """
    Get timestamps for individual files and sort the files by time.
    The stucture of the path should be:
    path
    ├── data
    │   ├── sample1.mzml
    │   ├── sample2.mzml
    │   └── ...
    └── ...

    Parameters
    ----------
    path : str
        Path to the main directory that contains a subdirectory 'data' with mzML or mzXML files.
    output : bool
        If True, output the timestamps to a txt file with two columns: 'file_name' and 'aquisition_time'.

    Return
    ------
    file_times : list
        A list of tuples with two elements: 'file_name' and 'aquisition_time'.

    Output
    ------
    timestamps.txt : txt file
        A txt file with two columns: 'file_name' and 'aquisition_time' in the specified path.
    """

    # if path is not specified, use the current working directory
    if path is None:
        path = os.getcwd()
    
    path_data = os.path.join(path, 'data')

    if os.path.exists(path_data):
        file_names = [f for f in os.listdir(path_data) if f.lower().endswith('.mzml') or f.lower().endswith('.mzxml')]
        file_names = [f for f in file_names if not f.startswith(".")]  # for Mac OS
        file_names = sorted(file_names)

    times = []
    print("Getting timestamps for individual files...")
    for f in tqdm(file_names):
        tmp = os.path.join(path_data, f)
        times.append(get_start_time(tmp))
    
    file_names = [f.split(".")[0] for f in file_names]
    
    # sort the files by time
    file_times = list(zip(file_names, times))
    file_times = sorted(file_times, key=lambda x: x[1])

    # output to a txt file using pandas

    df = pd.DataFrame(file_times, columns=["file_name", "aquisition_time"])
    if output:
        output_path = os.path.join(path, "timestamps.txt")
        df.to_csv(output_path, sep="\t", index=False)
    else:
        return df


def formula_to_mz(formula, adduct, charge):
    """
    Calculate the m/z value of a molecule given its chemical formula, adduct and charge.

    Parameters
    ----------
    formula : str
        Chemical formula of the molecule.
    adduct : str
        Adduct of the molecule. The first character should be '+' or '-'. In particular, 
        for adduct like [M-H-H2O]-, use '-H3O' or '-H2OH'.
    charge : int
        Charge of the molecule. Positive for cations and negative for anions.

    Returns
    -------
    mz : float
        The m/z value of the molecule.

    Examples
    --------
    >>> formula_to_mz("C6H12O6", "+H", 1)
    181.070665
    >>> formula_to_mz("C9H14N3O8P", "-H2OH", -1)
    304.034010
    """

    mz = 0

    # original molecule
    formula_matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    atom_counts = Counter()
    for element, count in formula_matches:
        atom_counts[element] += int(count) if count else 1

    mz += sum(ATOM_MASSES[element] * count for element, count in atom_counts.items())

    # adduct
    adduct_matches = re.findall(r'([A-Z][a-z]*)(\d*)', adduct[1:])
    adduct_counts = Counter()
    for element, count in adduct_matches:
        adduct_counts[element] += int(count) if count else 1

    if adduct[0] == '+':
        mz += sum(ATOM_MASSES[element] * count for element, count in adduct_counts.items())
    elif adduct[0] == '-':
        mz -= sum(ATOM_MASSES[element] * count for element, count in adduct_counts.items())

    # charge
    mz = (mz - ELECTRON_MASS * charge) / abs(charge)

    return mz


def get_start_time(file_name):
    """
    Function to get the start time of the raw data.

    Parameters
    ----------
    file_name : str
        Absolute path of the raw data.
    """

    if os.path.exists(str(file_name)):
        with open(file_name, "rb") as f:
            for l in f:
                l = str(l)
                if "startTimeStamp" in str(l):
                    t = l.split("startTimeStamp")[1].split('"')[1]
                    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")


def extract_signals_from_string(ms2):
    """
    Extract signals from MS2 spectrum in string format.

    Parameters
    ----------
    ms2 : str
        MS2 spectrum in string format. Format: "mz1;intensity1|mz2;intensity2|..."
        example: "100.0;1000.0|200.0;2000.0|300.0;3000.0|"
    
    returns
    ----------
    peaks : numpy.array
        Peaks in numpy array format.
    """
    
    # Use findall function to extract all numbers matching the pattern
    numbers = re.findall(r'\d+\.\d+', ms2)
    
    # Convert the extracted numbers from strings to floats
    numbers = [float(num) for num in numbers]
    
    numbers = np.array(numbers).reshape(-1, 2)

    return numbers


def convert_signals_to_string(signals):
    """
    Convert peaks to string format.

    Parameters
    ----------
    signals : numpy.array
        MS2 signals organized as [[mz1, intensity1], [mz2, intensity2], ...]

    Returns
    -------
    string : str
        Converted signals in string format. Format: "mz1;intensity1|mz2;intensity2|..."
    """

    if signals is None:
        return None
    
    string = ""
    for i in range(len(signals)):
        string += str(np.round(signals[i, 0], decimals=4)) + ";" + str(np.round(signals[i, 1], decimals=4)) + "|"
    string = string[:-1]
    
    return string


ATOM_MASSES = {
    'H': 1.00782503207,
    'D': 2.01410177812,
    'C': 12.0,
    'N': 14.0030740052,
    'O': 15.9949146221,
    'F': 18.998403163,
    'Na': 22.989769282,
    'Mg': 23.985041697,
    'P': 30.973761998,
    'S': 31.97207069,
    'Cl': 34.968852682,
    'K': 38.96370649,
    'Ca': 39.96259098,
    'Fe': 55.93493633,
    'Cu': 62.92959772,
    'Zn': 63.92914201,
    'Br': 78.9183376,
    'I': 126.904473,
}

ELECTRON_MASS = 0.00054857990946