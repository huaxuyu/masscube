# A module for statistical analysis

# imports
from scipy.stats import ttest_ind, false_discovery_control, f_oneway
import numpy as np
import os

def statistical_analysis(feature_table, params, before_norm=False):
    """
    1. Univariate analysis (t-test and p-value adjustment for two groups; ANOVA and p-value adjustment for multiple groups)
    2. Multivariate analysis (PCA)

    Parameters
    ----------
    feature_table : pandas DataFrame
        The feature table.
    params: Params object
        The parameters for the experiment.

    Returns
    -------
    feature_table : pandas DataFrame
    """
    
    v = [params.sample_names[i] for i in range(len(params.individual_sample_groups)) if params.individual_sample_groups[i] not in ['qc', 'blank']]
    data_array = np.array(feature_table[v], dtype=int)

    s = len(params.sample_groups) - 2
    v = np.array([i for i in params.individual_sample_groups if i not in ['qc', 'blank']])

    if s == 2:
        p_values = t_test(data_array, v)
    elif s > 2:
        p_values = anova(data_array, v)

    elif s == 1 and before_norm==False:
        print("No statistical analysis is performed since only one group is found.")

    # for PCA analysis, the QC samples should also be included
    v = [params.sample_names[i] for i in range(len(params.individual_sample_groups)) if params.individual_sample_groups[i] not in ['blank']]
    data_array = feature_table[v].values
    v = np.array([i for i in params.individual_sample_groups if i not in ['blank']])

    pca_analysis(data_array, v, output_dir=params.statistics_dir, before_norm=before_norm)

    if s == 2:
        feature_table['t_test_p'] = p_values
    elif s > 2:
        feature_table['ANOVA_p'] = p_values
    
    return feature_table


def t_test(data_array, individual_sample_groups):
    """
    A function to perform t-test on a feature list.

    Parameters
    ----------
    feature_table : pandas DataFrame
        The feature table.
    individual_sample_groups : list
        A list of groups of individual samples.

    Returns
    -------
    p_values : list
    """

    # Perform t-test
    p_values = []
    sample_groups = list(set(individual_sample_groups))
    if len(sample_groups) != 2:
        print("The number of sample groups is not equal to 2.")
        return None
    
    v1 = individual_sample_groups == sample_groups[0]
    v2 = individual_sample_groups == sample_groups[1]

    for i in range(len(data_array)):
        # if all values are equal, the p-value will be 1
        if np.all(data_array[i] == data_array[i, 0]):
            p_values.append(1)
        else:
            p_values.append(ttest_ind(data_array[i, v1], data_array[i, v2]).pvalue)
    
    # adjusted_p_values = false_discovery_control(p_values)

    return p_values


def anova(data_array, individual_sample_groups):
    """
    A function to perform ANOVA on a feature list.

    Parameters
    ----------
    data_array : numpy array
        The feature intensities.
    individual_sample_groups : list
        A list of groups of individual samples.

    Returns
    -------
    p_values, adjusted_p_values : list
    """

    p_values = []
    sample_groups = list(set(individual_sample_groups))

    if len(sample_groups) < 2:
        print("The number of sample groups is less than 2.")
        return None
    
    for i in range(len(data_array)):
        if np.all(data_array[i] == data_array[i, 0]):
            p_values.append(1)
        else:
            p_values.append(f_oneway(*[data_array[i, individual_sample_groups == g] for g in sample_groups]).pvalue)
    
    # adjusted_p_values = false_discovery_control(p_values)

    return p_values


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from .visualization import plot_pca


def pca_analysis(data_array, individual_sample_groups, scaling=True, transformation=True, gapFillingRatio=0.2, output_dir=None, before_norm=False):
    """
    Principal component analysis (PCA) analysis.

    Parameters
    ----------
    data_array : numpy array
        The feature intensities.
    individual_sample_groups : list
        A list of groups of individual samples.
    scaling : bool
        Whether to scale the data.
    transformation : bool
        Whether to transform the data.
    gapFillingRatio : float
        The ratio for gap-filling.
    output_dir : str
        The output directory.
    """

    X = np.array(data_array, dtype=float)

    # drop the columns with all zeros
    X = X[~np.all(X == 0, axis=1), :]

    # Gap-filling
    for i, vec in enumerate(X):
        if not np.all(vec):
            X[i][vec == 0] = np.min(vec[vec!=0]) * gapFillingRatio

    # transformation by log10
    if transformation:
        X = np.log10(X)

    # scaling
    if scaling:
        X = (X - np.mean(X, axis=1).reshape(-1, 1)) / np.std(X, axis=1).reshape(-1, 1)
    
    # PCA analysis
    X = X.transpose()
    pca = PCA(n_components=2)
    pca.fit(X)
    var_PC1, var_PC2 = pca.explained_variance_ratio_
    vecPC1 = pca.transform(X)[:,0]
    vecPC2 = pca.transform(X)[:,1]

    if output_dir is not None:
        if before_norm:
            output_dir = os.path.join(output_dir, "PCA_before_normalization.png")
        else:
            output_dir = os.path.join(output_dir, "PCA.png")

    plot_pca(vecPC1, vecPC2, var_PC1, var_PC2, individual_sample_groups, output_dir)

    return vecPC1, vecPC2, var_PC1, var_PC2
