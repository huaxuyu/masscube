# Author: Hauxu Yu

# A module for sample normalization

import numpy as np

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

    if method == 'pqn':
        array = array[array[:, ref_idx] != 0, :]
        # calculate the normalization factor
        return np.median(array / array[:, ref_idx][:, None], axis=0)
    

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

    return np.array(array / v, dtype=int)


def find_reference_sample(array, method='number'):
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