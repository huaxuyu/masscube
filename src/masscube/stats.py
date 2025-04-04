# Author: Huaxu Yu

# A module for statistical analysis

# imports

import numpy as np
import os
from scipy.stats import ttest_ind, f_oneway
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random

from .visualization import plot_pca


def full_statistical_analysis(feature_table, params, include_qc=False):
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
    
    # UMAP analysis
    umap_analysis(feature_table, params)
    
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


def pca_analysis(data_array, individual_sample_groups, scaling=True, transformation=True, gapFillingRatio=0.2, output_dir=None, before_norm=False):
    """
    Principal component analysis (PCA) analysis.

    Parameters
    ----------
    data_array : numpy array
        The feature intensities. Features are in rows and samples are in columns.
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

    X = np.array(data_array, dtype=np.float64)

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

    plot_pca(vecPC1, vecPC2, var_PC1, var_PC2, individual_sample_groups, output_dir=output_dir)

    return vecPC1, vecPC2, var_PC1, var_PC2


def umap_analysis(feature_table, params):
    """
    Perform UMAP analysis.

    Parameters
    ----------
    feature_table : pandas DataFrame
        The feature table.
    params : Params object
        The parameters for the experiment.
    include_qc : bool
        Whether to include the QC samples.
    """

    if params.sample_metadata is None:
        print("No sample metadata. UMAP analysis is not performed.")
        return None
    if params.statistics_dir is None:
        print("No statistics directory. UMAP analysis is not performed.")
        return None
    
    df = params.sample_metadata
    df = df[(~df['is_qc']) & (~df['is_blank'])]
    n = df.iloc[:, 0].values
    data_arr = feature_table[n].values  # samples in columns and features in rows
    keys = [i for i in df.columns[1:] if i not in ['is_qc', 'is_blank', 'analytical_order', 'time']]

    # UMAP analysis
    data_arr = data_arr.T
    data_arr = StandardScaler().fit_transform(data_arr)
    
    # set random seed
    n_samples = data_arr.shape[0]
    n_neighbors = min(15, n_samples - 1)
    reducer = UMAP(n_neighbors=n_neighbors, random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(data_arr)

    # by different metadata
    for color_by in keys:
        # convert the corresponding metadata to numerical values
        g = df[color_by].values
        ug = list(set(g))
        colors = generate_random_color(len(ug))
        metadata_to_color = {ug[i]: colors[i] for i in range(len(ug))}
        
        color_list = [metadata_to_color[i] for i in g]
        
        # plot the UMAP
        plt.figure(figsize=(10, 10))
        plt.rcParams['font.size'] = 20
        if 'Arial' in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = 'Arial'
        # remove frame, x and y axis
        plt.box(False)
        plt.xticks([])
        plt.yticks([])
        plt.scatter(embedding[:, 0], embedding[:, 1], c=color_list, s=100, edgecolors='black', linewidths=0.4)
        # set data point transparency
        plt.setp(plt.gca().collections, alpha=0.75)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f"UMAP colored by {color_by}")
        # show legend
        for i in range(len(ug)):
            plt.scatter([], [], c=colors[i], label=ug[i])
        plt.legend(loc='upper right', fontsize=20)
        plt.savefig(os.path.join(params.statistics_dir, f"UMAP_{color_by}.png"))
        plt.close()


def generate_random_color(num):
    """
    Randomly generate colors.
    
    Parameters
    ----------
    num : int
        The number of colors to generate.
    
    Returns
    -------
    colors : list
        A list of hex colors.
    """

    if num < 10:
        return COLORS[:num]
    
    else:
        colors = [c for c in COLORS]
        for _ in range(num - 10):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            # convert to hex
            colors.append('#%02X%02X%02X' % (r, g, b))

        return colors


COLORS = ["#FF5050", "#0078F0", "#00B050", "#FFC000", "#7030A0", "#FF00FF", "#00B0F0", "#FF0000", "#00FF00", "#0000FF"]
