# Author: Huaxu Yu

# A module for normalization
# There are two types of normalization:
# 1. Sample normalization - to normalize samples with different total amounts/concentrations.
# 2. Signal normalization - to address the signal drifts in the mass spectrometry data.


# imports
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm

from .visualization import plot_lowess_normalization

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


def sample_normalization(feature_table, sample_metadata=None, method='pqn', feature_selection=True):
    """
    A normalization function that takes a feature table as input and returns a normalized feature table.

    Parameters
    ----------
    feature_table : pandas DataFrame
        The feature table.
    sample_metadata : pd.DataFrame
        DataFrame containing sample metadata. See params module for details.
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

    if sample_metadata is None:
        print("\tSample normalization failed: sample metadata is required.")
        return feature_table

    data = feature_table[sample_metadata.iloc[:, 0]].values

    if feature_selection:
        hq_data = high_quality_feature_selection(data, is_qc=sample_metadata['is_qc'], is_blank=sample_metadata['is_blank'])
    else:
        hq_data = data
    
    # drop blank samples for normalization
    data_to_norm = data[:, ~sample_metadata['is_blank']]
    hq_data_to_norm = hq_data[:, ~sample_metadata['is_blank']]

    # STEP 3: find the normalization factors
    v = find_normalization_factors(hq_data_to_norm, method=method)

    # STEP 4: normalize samples by factors
    data_to_norm = sample_normalization_by_factors(data_to_norm, v)
    data[:, ~sample_metadata['is_blank']] = data_to_norm

    # STEP 5: update the feature table
    feature_table[sample_metadata.iloc[:, 0]] = data

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
        'total_intensity': total intensity normalization.
        'median_intensity': median intensity normalization.
        'quantile': quantile normalization.
        'mdfc': maximum density fold change normalization.

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
    elif method == 'total_intensity':
        # to be implemented
        pass
    elif method == 'median_intensity':
        # to be implemented
        pass
    elif method == 'quantile':
        # to be implemented
        pass
    elif method == 'mdfc':
        # to be implemented
        pass

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

    Returns
    -------
    numpy array
        Normalized data.
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


def high_quality_feature_selection(array, is_qc=None, is_blank=None, blank_ratio_tol=0.5, qc_rsd_tol=0.25):
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
    is_qc : numpy array
        Boolean array indicating whether a sample is a quality control sample.
    is_blank : numpy array
        Boolean array indicating whether a sample is a blank sample.
    blank_ratio_tol : float
        The tolerance of the ratio of the average intensity in blank samples to the average intensity in QC and biological samples.
    qc_rsd_tol : float
        The tolerance of the relative standard deviation (RSD) in QC samples.

    Returns
    -------
    numpy array
        High-quality features. Features are in rows and samples are in columns.
    numpy array
        The index of the selected features.
    """

    # 1. filter features based on blank samples
    if is_blank is not None:
        blank_avg = np.mean(array[:, is_blank], axis=1)
        sample_ave = np.mean(array[:, ~is_blank], axis=1)
        sample_ave[sample_ave == 0] = 1     # avoid division by zero
        blank_pass = blank_avg / sample_ave < blank_ratio_tol
    else:
        blank_pass = np.ones(array.shape[0], dtype=bool)

    # 2. filter features based on QC samples (3 QC samples are required)
    if is_qc is not None and np.sum(is_qc) > 2:
        sd = np.std(array[:, is_qc], axis=1, ddof=1) 
        mean = np.mean(array[:, is_qc], axis=1)
        rsd = np.array([s/m if m != 0 else 99 for s, m in zip(sd, mean)])
        qc_pass = rsd < qc_rsd_tol
    else:
        qc_pass = np.ones(array.shape[0], dtype=bool)

    idxes = np.logical_and(blank_pass, qc_pass)

    return array[idxes]


"""
Signal normalization
====================

Provides
    1. Feature-wise normalization based on timestamp.
    2. Standard-free QC-based normalization.
"""

def signal_normalization(feature_table, sample_metadata, method='lowess', output_plot_path=None):
    """
    A function to normalize MS signal drifts based on analytical order.

    Parameters
    ----------
    feature_table : pandas DataFrame
        The feature table.
    sample_metadata : pd.DataFrame
        DataFrame containing sample metadata. See params module for details.
    method : str
        The method to find the normalization factors.
        'lowess': locally weighted scatterplot smoothing.
    output_plot_path : str
        The path to save the normalization plot. If none, no visualization will be generated.
    
    Returns
    -------
    pandas DataFrame
        Normalized feature table.
    """

    # STEP 1: check if 3 or more QC samples are available
    if np.sum(sample_metadata['is_qc']) < 3:
        print("\tSignal normalization failed: at least three quality control samples are required.")
        return feature_table
    
    # STEP 2: get analytical order
    tmp = sample_metadata.sort_values(by="analytical_order")
    samples = tmp.iloc[:, 0].values
    qc_idx = tmp['is_qc'].values
    sample_idx = ~tmp['is_blank'].values

    arr = feature_table.loc[:, samples].values
    n = len(samples)
    n_qc = np.sum(qc_idx)

    # STEP 3: normalize samples
    if method == 'lowess':
        print("\tSignal normalization is running: lowess normalization.")
        for id, a in enumerate(tqdm(arr)):
            r = lowess_normalization(a, qc_idx, frac=np.min([0.5, 8/n_qc]), it=3)
            if r['fit_curve'] is not None:
                # visualization
                if output_plot_path is not None:
                    plot_lowess_normalization(arr=a, fit_curve=r['fit_curve'], arr_new=r['normed_arr'], 
                                              sample_idx=sample_idx, qc_idx=qc_idx, n=n, id=id, output_dir=output_plot_path)
                arr[id] = r['normed_arr']

    feature_table.loc[:, samples] = arr

    return feature_table


def lowess_normalization(array, qc_idx, frac=0.07, it=3):
    """
    A function to normalize samples using quality control samples.

    Parameters
    ----------
    array : numpy array
        The data to be normalized.
    qc_idx : numpy array of bool
        Boolean array indicating whether a sample is a quality control sample. It's length 
        should be the same as the length of array.
    frac : float
        The fraction of the data used when estimating each y-value (used in lowess). See statsmodels package for details.
    it : int
        The number of residual-based reweightings to perform (used in lowess). See statsmodels package for details.

    Returns
    -------
    dict
        A dictionary containing the lowess model, the fit curve, and the normalized array.
        {'model': model, 'fit_curve': y, 'normed_arr': int_arr_corr}
    """

    # only use qc data > 0 for normalization
    valid_idx = qc_idx & (array > 0)
    qc_arr = array[valid_idx]

    # if there are more than 2 QC samples
    if len(qc_arr) > 2:
        x = np.arange(len(array))
        model = lowess(qc_arr, x[valid_idx], frac=frac, it=it)
        y = np.interp(x, model[:, 0], model[:, 1])
        y[y < 1] = 1    # gap filling
        int_arr_corr = array / y * np.max(y)
    else:
        int_arr_corr = array
        y = None
        model = None

    return {'model': model, 'fit_curve': y, 'normed_arr': int_arr_corr}