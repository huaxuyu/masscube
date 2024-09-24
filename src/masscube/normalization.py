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

Aims:

Sample normalization is to normalize samples with different total amounts/concentrations, 
such as urine samples, fecal samples, and tissue samples that are unweighted.

Modules:
    1. Select high-quality features for normalization using QC and blank samples.
    2. Find the reference sample.
    3. Find the normalization factors.
    4. Normalize samples by factors.
"""


def sample_normalization(feature_table, sample_groups=None, method='pqn', feature_selection=True):
    """
    A normalization function that takes a feature table as input and returns a normalized feature table.

    Parameters
    ----------
    feature_table : pandas DataFrame
        The feature table.
    sample_groups : list
        A list of groups of individual samples. Blank samples will be skipped.
    method : str
        The method to find the normalization factors.
        'pqn': probabilistic quotient normalization.
        'total_intensity': total intensity normalization.
        'median_intensity': median intensity normalization.
        'quantile': quantile normalization.
        'mdfc': maximum density fold change normalization (https://doi.org/10.1093/bioinformatics/btac355).
    feature_selection : bool
        Whether to select high-quality features for normalization. High-quality features have
        relative standard deviation (RSD) less than 25% in QC samples and average intensity in QC+biological samples
        greater than 2 fold of the average intensity in blank samples.

    Returns
    -------
    pandas DataFrame
        Normalized feature table.
    """

    # STEP 1: data preparation
    # 1. check the length of the sample groups
    if sample_groups is None:
        sample_groups = ['sample'] * (feature_table.shape[1]-22)
    if len(sample_groups) != (feature_table.shape[1]-22):
        print("Normalization failed: the length of the sample groups does not match the number of samples.")
        return feature_table
    sample_groups = np.array(sample_groups, dtype=str)
    # 2. separate the data array from the feature table
    data = feature_table.iloc[:, 22:].values

    # STEP 2: select high-quality features for normalization
    if feature_selection:
        hq_data = high_quality_feature_selection(data, sample_groups)
    else:
        hq_data = data
    
    # drop blank samples for normalization
    data = data[:, sample_groups!='blank']
    hq_data = hq_data[:, sample_groups!='blank']

    # STEP 3: find the normalization factors
    v = find_normalization_factors(hq_data, method=method)

    # STEP 4: normalize samples by factors
    data = sample_normalization_by_factors(data, v)

    # STEP 5: update the feature table
    feature_table.iloc[:, 22:(22+data.shape[1])] = data

    return feature_table


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


def high_quality_feature_selection(array, sample_groups, by_blank=True, by_qc=True,
                                   blank_ratio_tol=0.5, qc_rsd_tol=0.25, return_idx=False):
    """
    Select high-quality features based on provided criteria for normalization.
    High-quality features have (default):
        1. relative standard deviation (RSD) less than 25% in QC samples and 
        2. average intensity in QC and biological samples greater than 0.5 fold of 
        the average intensity in blank samples.

    Parameters
    ----------
    array : numpy array
        The data to be normalized. Samples are in columns and features are in rows.
    sample_groups : numpy array or list
        A list of groups of individual samples. 'blank' for blank samples, 'qc' for quality control samples.

    Returns
    -------
    numpy array
        High-quality features. Features are in rows and samples are in columns.
    """

    # 1. filter features based on blank samples
    sample_groups = list(sample_groups)
    if by_blank and 'blank' in sample_groups:
        blank_idx = np.array(sample_groups) == 'blank'
        blank_avg = np.mean(array[:, blank_idx], axis=1)
        sample_ave = np.mean(array[:, ~blank_idx], axis=1)
        # simple imputation for zero division
        sample_ave[sample_ave == 0] = 1
        blank_pass = blank_avg / sample_ave < blank_ratio_tol
    else:
        blank_pass = np.ones(array.shape[0], dtype=bool)

    # 2. filter features based on QC samples
    # at least three qc samples are required
    if by_qc and sample_groups.count('qc') > 2:
        qc_idx = np.array(sample_groups) == 'qc'
        sd = np.std(array[:, qc_idx], axis=1, ddof=1) 
        mean = np.mean(array[:, qc_idx], axis=1)
        rsd = np.array([s/m if m != 0 else 99 for s, m in zip(sd, mean)])
        qc_pass = rsd < qc_rsd_tol
    else:
        qc_pass = np.ones(array.shape[0], dtype=bool)

    print(str(np.sum(np.logical_and(blank_pass, qc_pass))) + " features are selected for determining normalization factors.")

    if return_idx:
        return array[np.logical_and(blank_pass, qc_pass)], np.logical_and(blank_pass, qc_pass)
    else:
        return array[np.logical_and(blank_pass, qc_pass)]


"""
Signal normalization
====================

Provides
    1. Feature-wise normalization based on timestamp.
    2. Standard-free QC-based normalization.
"""

def signal_normalization(feature_table, sample_groups, analytical_order, method='lowess',
                         batch_idx=None):
    """
    A function to normalize MS signal drifts based on analytical order.

    Parameters
    ----------
    feature_table : pandas DataFrame
        The feature table.
    sample_groups : list
        A list of groups of individual samples. Blank samples will be skipped.
    analytical_order : list
        The order of the samples. It should have the same length as the number of samples.
    method : str
        The method to find the normalization factors.
        'lowess': locally weighted scatterplot smoothing.
    batch_idx : list
        Not used now. The index of the batches. It should have the same length as the number of samples.
        e.g., [1, 1,..., 2, 2,..., 3, 3,...]
    
    Returns
    -------
    pandas DataFrame
        Normalized feature table.
    """

    # STEP 1: data preparation
    # 1. check the length of the sample groups
    if len(sample_groups) != (feature_table.shape[1]-22):
        print("Normalization failed: the length of the sample groups does not match the number of samples.")
        return feature_table
    sample_groups = np.array(sample_groups, dtype=str)
    # 2. separate the data array from the feature table
    data = feature_table.iloc[:, 22:].values

    # STEP 2: perform normalization
    if method == 'lowess':
        data_corr, _ = lowess_normalization(data, sample_groups, analytical_order, batch_idx)
    
    # STEP 3: update the feature table
    feature_table.iloc[:, 22:] = data_corr

    return feature_table


def lowess_normalization(array, sample_groups, analytical_order, batch_idx=None):
    """
    A function to normalize samples using quality control samples.

    Parameters
    ----------
    array : numpy array
        The data to be normalized. Samples are in columns and features are in rows.
    sample_groups : numpy array or list
        A list of groups of individual samples. 'blank' for blank samples, 'qc' for quality control samples.
    analytical_order : numpy array or list
        The order of the samples.
    batch_idx : numpy array or list
        The index of the batches. It should have the same length as the number of samples.
        e.g., [1, 1,..., 2, 2,..., 3, 3,...]

    Returns
    -------
    numpy array
        Normalized data.
    """

    # 1. reorder the data array
    array = array[:, np.argsort(analytical_order)]
    sample_groups = np.array(sample_groups)[np.argsort(analytical_order)]
    qc_idx = np.where(sample_groups == 'qc')[0]

    # 2. feature-wise normalization
    data_corr = []
    corrected_arr = []
    for int_arr in array:
        # build loess model using qc samples
        qc_arr = int_arr[qc_idx]
        # only keep the positive values
        qc_idx_tmp = qc_idx[qc_arr > 0]
        qc_arr = qc_arr[qc_arr > 0]

        if len(qc_arr) > 2:
            model = lowess(qc_arr, qc_idx_tmp, frac=0.09, it=0)
            x_new = np.arange(len(int_arr))
            y_new = np.interp(x_new, model[:, 0], model[:, 1])
            y_new[y_new < 0] = np.min(y_new[y_new > 0])
            int_arr_corr = int_arr / y_new * np.min(y_new)
            corrected_arr.append(True)
        else:
            int_arr_corr = int_arr
            corrected_arr.append(False)

        data_corr.append(int_arr_corr)

    data_corr = np.array(data_corr)
    # sort the data back to the original order
    data_corr = data_corr[:, analytical_order]

    return data_corr, corrected_arr