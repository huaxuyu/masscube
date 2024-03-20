from pyteomics import mass
import numpy as np
from pyteomics.mass.mass import isotopologues, most_probable_isotopic_composition, calculate_mass

import pandas as pd
import os


def cal_ion_mass(formula, adduct, charge):
    """
    A function to calculate the ion mass of a given formula, adduct and charge.

    Parameters
    ----------
    formula: str
        The chemical formula of the ion.
    adduct: str
        Adduct of the ion.
    charge: int
        Charge of the ion. 
        Use signs for specifying ion modes: +1 for positive mode and -1 for negative mode.

    Returns
    -------
    ion_mass: float
        The ion mass of the given formula, adduct and charge.
    """

    # if there is a D in the formula, and not followed by a lowercase letter, replace it with H[2]
    if 'D' in formula and not formula[formula.find('D') + 1].islower():
        formula = formula.replace('D', 'H[2]')

    # calculate the ion mass
    final_formula = formula + adduct
    ion_mass = (mass.calculate_mass(formula=final_formula) - charge * _ELECTRON_MASS) / abs(charge)
    return ion_mass

_ELECTRON_MASS = 0.00054858
    


def calculate_isotope_distribution(formula, mass_resolution=10000, intensity_threshold=0.001):

    mass = []
    abundance = []

    for i in isotopologues(formula, report_abundance=True, overall_threshold=intensity_threshold):
        mass.append(calculate_mass(i[0]))
        abundance.append(i[1])
    mass = np.array(mass)
    abundance = np.array(abundance)
    order = np.argsort(mass)
    mass = mass[order]
    abundance = abundance[order]

    mass_diffrence = mass[0] / mass_resolution
    # merge mass with difference lower than 0.01
    groups = []
    group = [0]
    for i in range(1, len(mass)):
        if mass[i] - mass[i-1] < mass_diffrence:
            group.append(i)
        else:
            groups.append(group)
            group = [i]
    groups.append(group)
    groups

    mass = [np.mean(mass[i]) for i in groups]
    abundance = [np.sum(abundance[i]) for i in groups]
    abundance = np.array(abundance) / np.max(abundance)

    return mass, abundance


def generate_sample_table(path=None):

    # if path is not specified, use the current working directory
    if path is None:
        path = os.getcwd()
    
    path_data = os.path.join(path, 'data')

    if os.path.exists(path):
        filenames = [os.path.splitext(f)[0] for f in os.listdir(path_data) if f.lower().endswith('.mzml') or f.lower().endswith('.mzxml')]
        filenames = sorted(filenames)
        sample_table = pd.DataFrame({'Sample': filenames, "Groups": [None]*len(filenames)})
        sample_table.to_csv(os.path.join(path, 'sample_table.csv'), index=False)
    else:
        raise FileNotFoundError(f"The path {path} does not exist.")



    
