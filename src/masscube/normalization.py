# Author: Hauxu Yu

# A module for normalization
# There are two types of normalization:
# 1. Sample normalization - to normalize samples with different total amounts/concentrations.
# 2. Signal normalization - to address the signal drifts in the mass spectrometry data.

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

"""
Sample normalization
====================

Provides
    1. Find the normalization factors.
    2. Find reference sample.
    3. Normalize samples by factors.
"""

def find_normalization_factors(array, method='pqn'):
    """ 
    A function to find the normalization factors for a data frame.

    Parameters
    ----------
    array : numpy array
        The data to be normalized.
    method : str
        The method to find the normalization factors.
        'pqn': probabilistic quotient normalization.

    Returns
    -------
    numpy array
        Normalization factor.
    """

    # find the reference sample
    ref_idx = find_reference_sample(array)
    ref_arr = array[:, ref_idx]

    factors = []
    if method == 'pqn':
        for i in range(array.shape[1]):
            a = array[:, i]
            common = np.logical_and(a > 0, ref_arr > 0)
            factors.append(np.median(a[common] / ref_arr[common]))
    return np.array(factors)
    

def sample_normalization_by_factors(array, v):
    """
    A function to normalize a data frame by a vector.

    Parameters
    ----------
    array : numpy array
        The data to be normalized.
    v : numpy array
        The normalization factor.
    """

    # change all zeros to ones
    v[v == 0] = 1

    return np.array(array / v, dtype=np.int64)


def find_reference_sample(array, method='median_intensity'):
    """
    A function to find the reference sample for normalization.
    Note, samples are in columns and features are in rows.

    Parameters
    ----------
    array : numpy array
        The data to be normalized.
    method : str
        The method to find the reference sample. 
        'number': the reference sample has the most detected features.
        'total_intensity': the reference sample has the highest total intensity.
        'median_intensity': the reference sample has the highest median intensity.

    Returns
    -------
    int
        The index of the reference sample.
    """

    if method == 'number':
        # find the reference sample with the most detected features
        return np.argmax(np.count_nonzero(array, axis=0))
    elif method == 'total_intensity':
        # find the reference sample with the highest total intensity
        return np.argmax(np.sum(array, axis=0))
    elif method == 'median_intensity':
        # find the reference sample with the highest median intensity
        return np.argmax(np.median(array, axis=0))
    

def sample_normalization(feature_table, individual_sample_groups, method='pqn'):
    """
    A function to normalize samples using a feature list.
    By default, the blank samples are not normalized.

    Parameters
    ----------
    feature_table : pandas DataFrame
        The feature table.
    sample_groups : list
        A list of groups of individual samples. Blank samples will be skipped.
    method : str
        The method to find the normalization factors.
        'pqn': probabilistic quotient normalization.

    Returns
    -------
    numpy array
        Normalization factors.
    """

    blank_number = len([x for x in individual_sample_groups if 'blank' in x])
    total_number = len(individual_sample_groups)
    if blank_number == 0:
        array = np.array(feature_table.iloc[:, -total_number:])
    else:
        array = np.array(feature_table.iloc[:, -total_number:-blank_number])

    v = find_normalization_factors(array, method=method)

    array = sample_normalization_by_factors(array, v)

    if blank_number == 0:
        feature_table.iloc[:, -total_number:] = array
    else:
        feature_table.iloc[:, -total_number:-total_number+array.shape[1]] = array

    return feature_table



"""
Signal normalization
====================

Provides
    1. Feature-wise normalization based on timestamp.
    2. Standard-free QC-based normalization.
"""


def qc_normalization(array, order, qc_idx, batch_idx=None, method='lowess'):
    """
    A function to normalize samples using quality control samples.

    Parameters
    ----------
    array : numpy array
        The data to be normalized. Samples are in columns and features are in rows.
    order : list
        The order of the samples. It should have the same length as the number of samples.
    qc_idx : list
        The index of the quality control samples.
    batch_idx : list
        The index of the batches. It should have the same length as the number of samples. Not used now.
    method : str
        The method to find the normalization factors.
        'lowess': locally weighted scatterplot smoothing.

    Returns
    -------
    numpy array
        Normalized data.
    """

    array = array[:, np.argsort(order)]
    qc_idx = np.array(qc_idx)

    data_corr = []
    for i, int_arr in enumerate(array):
        # build loess model using qc samples
        qc_arr = int_arr[qc_idx]
        model = lowess(qc_arr, qc_idx, frac=0.09, it=0)
        x_new = np.arange(len(int_arr))
        y_new = np.interp(x_new, model[:, 0], model[:, 1])
        y_new[y_new < 0] = np.min(y_new[y_new > 0])

        int_arr_corr = int_arr / y_new * np.min(y_new)
        data_corr.append(int_arr_corr)
    data_corr = np.array(data_corr)
    # sort the data back to the original order
    data_corr = data_corr[:, order]