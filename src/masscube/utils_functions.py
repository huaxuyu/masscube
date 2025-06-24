# Author: Huaxu Yu

# A module for utility functions

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from IsoSpecPy import IsoTotalProb


####################################################################################################
# Sample management functions
####################################################################################################

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


def label_batch_id(df, ratio_threshold=10):
    """
    Using the time difference between files to recognize different batches.

    Parameters
    ----------
    df : pandas DataFrame
        A DataFrame with at least one column called 'time'.

    Returns
    -------
    df : pandas DataFrame
        A DataFrame with an extra column 'batch_id' indicating the batch number for each file.
    ratio_threshold : float
        If the interval between two files is larger than ratio_threshold * the minimum interval,
        the two files are considered to be in different batches.
    """

    df = df.sort_values(by="time")
    time_diff = np.diff(df['time'])
    t0 = ratio_threshold * np.min(time_diff)
    v = [0]
    for i in range(len(time_diff)):
        if time_diff[i] > t0:
            v.append(v[-1] + 1)
        else:
            v.append(v[-1])
    df['batch_id'] = v
    df = df.sort_index()
    
    return df


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


####################################################################################################
# Chemical formula-related utility functions
####################################################################################################

def formula_to_mz(formula, adduct):
    """
    Calculate the m/z value of a molecule given its chemical formula, adduct and charge.

    Parameters
    ----------
    formula : str
        Chemical formula of the molecule.
    adduct : str
        Adduct of the molecule.

    Returns
    -------
    mz : float
        The m/z value of the molecule with the given adduct.

    Examples
    --------
    >>> formula_to_mz("C6H12O6", "+H", 1)
    181.070665
    >>> formula_to_mz("C9H14N3O8P", "-H2OH", -1)
    304.034010
    """

    # original molecule
    parsed_formula = parse_formula(formula)

    # combine with adduct
    parsed_formula, charge = _combine_formula_with_adduct(parsed_formula, adduct)

    sp = IsoTotalProb(formula=parsed_formula, prob_to_cover=0.999)
    mass = sp[0][0]
    mz = (mass - charge * ELECTRON_MASS) / abs(charge)

    return mz


def formula_to_isotope_distribution(formula, adduct, prob_to_cover=0.9999, delta_mass=0.005):
    """
    Calculate the isotope distribution of a molecule given its chemical formula and adduct.

    Parameters
    ----------
    formula : str
        Chemical formula of the molecule.
    adduct : str
        Adduct of the molecule.
    prob_to_cover : float
        Probability to cover the isotope distribution. Default is 0.999.
    delta_mass : float
        The minimum mass difference that can be distinguished, which is determined by the 
        resolution of the mass spectrometer. Default is 0.001 Da for high resolution mass spectrometers.
    
    Returns
    -------
    isotopes : numpy.array
        An array of isotopes with their m/z values and intensities.
        The first column is the m/z value and the second column is the intensity.
    
    """

    # original molecule
    parsed_formula = parse_formula(formula)

    # combine with adduct
    results = _combine_formula_with_adduct(parsed_formula, adduct)

    if results is None:
        return None
    else:
        parsed_formula, charge = results

    sp = IsoTotalProb(formula=parsed_formula, prob_to_cover=prob_to_cover)

    isotopes = []
    for m, i in sp:
        isotopes.append((m, i))
    isotopes = np.array(isotopes)
    isotopes[:, 0] = (isotopes[:, 0] - charge * ELECTRON_MASS) / abs(charge)
    isotopes = isotopes[isotopes[:, 0].argsort()]

    binned = centroid_signals(isotopes, delta_mass)

    return binned


def parse_formula(formula):
    """
    Parse a chemical formula into a dictionary of elements and their counts.
    
    Parameters
    ----------
    formula : str
        The chemical formula to parse. For example, "C6H12O6" or "C9H14N3O8P".

    Returns
    -------
    atom_counts : collections.Counter
        A Counter object containing the elements and their counts in the formula.
        For example, for "C6H12O6", it returns Counter({'C': 6, 'H': 12, 'O': 6}).
    """
    
    formula_matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    atom_counts = Counter()
    for element, count in formula_matches:
            atom_counts[element] += int(count) if count else 1
    
    return atom_counts


def bin_isotopes_by_mass(data, bin_width):
    """
    Bin isotopes by m/z values.

    Parameters
    ----------
    data : list or numpy.array
        A list of tuples or a numpy array with two columns: m/z values and their corresponding abundances.
        For example, [(100.0, 1000.0), (200.0, 2000.0), (300.0, 3000.0)].
    bin_width : float
        The width of the bins to use for binning the m/z values.

    Returns
    -------
    binned : numpy.array
        A numpy array with two columns: the average m/z value of each bin and the total abundance in that bin.
    """

    binned = defaultdict(list)
    for mass, abundance in data:
        # Compute bin index
        bin_index = int(mass // bin_width)
        binned[bin_index].append([mass, abundance])

    binned = dict(binned)

    for bin, values in binned.items():
        total_prob = sum(prob for _, prob in values)
        average_mass = sum(mass * prob for mass, prob in values) / total_prob
        binned[bin] = (average_mass, total_prob)
    
    binned = np.array(list(binned.values()))
    binned = binned[np.argsort(binned[:, 0])]

    return binned


def _combine_formula_with_adduct(parsed_formula, adduct):
    """
    Combine a chemical formula with an adduct to get the final formula.

    Parameters
    ----------
    parsed_formula : Counter
        The parsed chemical formula as a Counter object, like Counter({'C': 6, 'H': 12, 'O': 6}).
    adduct : str
        Accepted adduct forms by MassCube, e.g. '[M+H]+', '[M-H]-', etc.
    
    Returns
    -------
    parsed_formula : Counter
        The final formula as a Counter object, like Counter({'C': 6, 'H': 13, 'O': 6}).
    charge : int
        The charge of the molecule after adding the adduct.
    """

    # adduct
    if adduct in ADDUCTS.keys():
        tmp = ADDUCTS[adduct]
    else:
        print(f"Adduct {adduct} not found in the database. Please check the adduct name.")
        return None
    
    # cannot subtract atoms that are not in the original formula
    for k, v in tmp.modification.items():
        if v<0 and k not in parsed_formula:
            return None
    
    for k, v in parsed_formula.items():
        parsed_formula[k] = v * tmp.mol_multiplier
    
    parsed_formula = parsed_formula + tmp.modification

    return parsed_formula, tmp.charge


####################################################################################################
# Spectral utility functions
####################################################################################################

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


def centroid_signals(signals, mz_tol=0.005):
    """
    Function to centroid signals in a mass spectrum.

    Parameters
    ----------
    signals: numpy array
        MS signals for a scan as 2D numpy array in float32, organized as [[m/z, intensity], ...].
    mz_tol: float
        m/z tolerance for centroiding. Default is 0.005 Da.

    Returns
    -------
    signals: numpy array
        Centroided signals.
    """

    if mz_tol is None:
        return signals
    
    # sort signals by m/z
    signals = signals[signals[:, 0].argsort()]

    v = np.diff(signals[:, 0]) < mz_tol

    if np.sum(v) == 0:
        return signals
    
    b = np.zeros(len(signals), dtype=int)
    for i in range(len(v)):
        if v[i]:
            b[i+1] = b[i]
        else:
            b[i+1] = b[i] + 1
    
    # merge signals with m/z difference less than mz_tol
    merged_signals = np.zeros((b[-1]+1, 2), dtype=np.float32)

    for i in range(b[-1]+1):
        if np.sum(b == i) == 1:
            merged_signals[i, 0] = signals[b == i, 0][0]
            merged_signals[i, 1] = signals[b == i, 1][0]
        else:
            tmp = signals[b == i]
            merged_signals[i, 0] = np.average(tmp[:, 0], weights=tmp[:, 1])
            merged_signals[i, 1] = np.sum(tmp[:, 1])

    return merged_signals


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

@dataclass
class Adduct:
    """
    A class to represent an adduct.
    """
    name: str            # output form, e.g. '[M+H]+'
    mass_shift: float    # e.g. 1.007276
    modification: str    # formula for the modification, e.g. Counter({'H': 1}) for [M+H]+ and Counter({'H': -1}) for [M-H]+
    charge: int          # e.g. 1 for one positive charge, -1 for one negative charge
    mol_multiplier: int  # number of molecules, e.g. 1 for [M+H]+, 2 for [2M+H]+, 3 for [3M+H]+
    considered: bool     # whether the adduct is considered in the analysis


ADDUCTS = {
    # Positive adducts
    '[M+H]+': Adduct('[M+H]+', 1.00782503227, Counter({'H': 1}), 1, 1, True),
    '[M+NH4]+': Adduct('[M+NH4]+', 18.03437413328, Counter({'N': 1, 'H': 4}), 1, 1, True),
    '[M]+': Adduct('[M]+', 0, Counter({}), 1, 1, False), # rare adduct form
    '[M+H+CH3OH]+': Adduct('[M+H+CH3OH]+', 33.03403978155, Counter({'C': 1, 'H': 5, 'O': 1}), 1, 1, False), # only consider if mobile phase is methanol
    '[M+Na]+': Adduct('[M+Na]+', 22.989769282, Counter({'Na': 1}), 1, 1, True),
    '[M+K]+': Adduct('[M+K]+', 38.963706493, Counter({'K': 1}), 1, 1, True),
    '[M+H+CH3CN]+': Adduct('[M+H+CH3CN]+', 42.033826, Counter({'C': 2, 'H': 4, 'N': 1}), 1, 1, False), # only consider if mobile phase is acetonitrile
    '[M-H+2Na]+': Adduct('[M-H+2Na]+', 44.971165, Counter({'Na': 2, 'H': -1}), 1, 1, False),
    '[M+H-H2O]+': Adduct('[M+H-H2O]+', -17.002739652469998, Counter({'H': -1, 'O': -1}), 1, 1, True),
    '[M+H+CH3COOH]+': Adduct('[M+H+CH3COOH]+', 61.02895440175, Counter({'C': 2, 'H': 5, 'O': 2}), 1, 1, True),
    '[M+H+HCOOH]+': Adduct('[M+H+HCOOH]+', 47.01330433721, Counter({'C': 1, 'H': 3, 'O': 2}), 1, 1, True),
    '[2M+H]+': Adduct('[2M+H]+', 1.00782503227, Counter({'H': 1}), 1, 2, True),
    '[2M+NH4]+': Adduct('[2M+NH4]+', 18.03437413328, Counter({'N': 1, 'H': 4}), 1, 2, False), # rare adduct form
    '[2M+Na]+': Adduct('[2M+Na]+', 22.989769282, Counter({'Na': 1}), 1, 2, False), # rare adduct form
    '[2M+H-H2O]+': Adduct('[2M+H-H2O]+', -17.002739652469998, Counter({'H': -1, 'O': -1}), 1, 2, False), # rare adduct form 
    '[3M+H]+': Adduct('[3M+H]+', 1.00782503227, Counter({'H': 1}), 1, 3, True),
    '[3M+H-H2O]+': Adduct('[3M-H-H2O]+', -17.002739652469998, Counter({'H': -1, 'O': -1}), 1, 3, False), # rare adduct form
    '[M+2H]2+': Adduct('[M+2H]2+', 2.01565006454, Counter({'H': 2}), 2, 1, True),
    '[M+3H]3+': Adduct('[M+3H]3+', 3.02347509681, Counter({'H': 3}), 3, 1, True),
    '[M+Li]+': Adduct('[M+Li]+', 7.016003443, Counter({'Li': 1}), 1, 1, False),
    '[M+Ag]+': Adduct('[M+Ag]+', 106.905092, Counter({'Ag': 1}), 1, 1, False),
    '[M+Ca]2+': Adduct('[M+Ca]2+', 39.96259092, Counter({'Ca': 1}), 2, 1, False),
    '[M+Fe]2+': Adduct('[M+Fe]2+', 55.9349363, Counter({'Fe': 1}), 2, 1, False),
    
    # Negative adducts
    '[M-H]-': Adduct('[M-H]-', -1.00782503227, Counter({'H': -1}), -1, 1, True),
    '[M+Cl]-': Adduct('[M+Cl]-', 34.96885273, Counter({'Cl': 1}), -1, 1, True),
    '[M+HCOO]-': Adduct('[M+HCOO]-', 44.99765427267, Counter({'C': 1, 'H': 1, 'O': 2}), -1, 1, True),
    '[M+CH3COO]-': Adduct('[M+CH3COO]-', 59.01330433721, Counter({'C': 2, 'H': 3, 'O': 2}), -1, 1, True),
    '[M-H-H2O]-': Adduct('[M-H-H2O]-', -19.01838971701, Counter({'H': -3, 'O': -1}), -1, 1, True),
    '[2M-H]-': Adduct('[2M-H]-', -1.00782503227, Counter({'H': -1}), -1, 2, True),
    '[3M-H]-': Adduct('[3M-H]-', -1.00782503227, Counter({'H': -1}), -1, 3, True),
    '[M-2H]2-': Adduct('[M-2H]2-', -2.01565006454, Counter({'H': -2}), -2, 1, True),
    '[M-3H]3-': Adduct('[M-3H]3-', -3.02347509681, Counter({'H': -3}), -3, 1, True), 
    '[2M+Cl]-': Adduct('[2M+Cl]-', 34.96885273, Counter({'Cl': 1}), -1, 2, False),
    '[2M+HCOO]-': Adduct('[2M+HCOO]-', 44.99765427267, Counter({'C': 1, 'H': 1, 'O': 2}), -1, 2, False),
    '[2M+CH3COO]-': Adduct('[2M+CH3COO]-', 59.01330433721, Counter({'C': 2, 'H': 3, 'O': 2}), -1, 2, False),
    '[2M-H-H2O]-': Adduct('[2M-H-H2O]-', -19.01838971701, Counter({'H': -3, 'O': -1}), -1, 2, False),
}
