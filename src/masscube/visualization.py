# Author: Hauxu Yu

# A module for data visualization.

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import numpy as np
import os
import re

from .annotation import _extract_peaks_from_string

def plot_bpcs(data_list=None, output=None, autocolor=False):
    """
    A function to plot the base peak chromatograms (overlapped) of a list of data.
    
    Parameters
    ----------
    data_list : list of MSData objects
        A list of data to be plotted.
    """

    if data_list is not None:
        if autocolor:
            color_list = _color_list
        else:
            color_list = ["black"] * len(data_list)

        plt.figure(figsize=(10, 4))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Arial'

        for i, d in enumerate(data_list):
            plt.plot(d.ms1_rt_seq, d.bpc_int, color=color_list[i], linewidth=0.5)
            plt.fill_between(d.ms1_rt_seq, d.bpc_int, color=color_list[i], alpha=0.05)
            plt.xlabel("Retention Time (min)", fontsize=18, fontname='Arial')
            plt.ylabel("Intensity", fontsize=18, fontname='Arial')
            plt.xticks(fontsize=14, fontname='Arial')
            plt.yticks(fontsize=14, fontname='Arial')

        if output:
            plt.savefig(output, dpi=600, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def random_color_generator():
    # set seed
    color = random.choice(list(mcolors.CSS4_COLORS.keys()))
    return color


_color_list = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]


def plot_roi(d, roi, mz_tol=0.005, rt_tol=1.0, output=False, break_scan=None):
    """
    Function to plot EIC of a target m/z.
    """
    rt_range = [roi.rt_seq[0]-rt_tol, roi.rt_seq[-1]+rt_tol]
    # get the eic data
    eic_rt, eic_int, _, eic_scan_idx = d.get_eic_data(target_mz=roi.mz, rt_range=rt_range, mz_tol=mz_tol)
    idx_start = np.where(eic_scan_idx == roi.scan_idx_seq[0])[0][0]
    idx_end = np.where(eic_scan_idx == roi.scan_idx_seq[-1])[0][0] + 1

    if break_scan is not None:
        idx_middle = np.where(eic_scan_idx == break_scan)[0][0]

    max_int = np.max(eic_int)

    plt.figure(figsize=(9, 3))
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.plot(eic_rt, eic_int, linewidth=0.5, color="black")
    plt.ylim(0, max_int*1.2)

    if break_scan is not None:
        plt.fill_between(eic_rt[idx_start:(idx_middle+1)], eic_int[idx_start:(idx_middle+1)], color="blue", alpha=0.2)
        plt.fill_between(eic_rt[idx_middle:idx_end], eic_int[idx_middle:idx_end], color="red", alpha=0.2)
    else:
        plt.fill_between(eic_rt[idx_start:idx_end], eic_int[idx_start:idx_end], color="black", alpha=0.2)
    plt.axvline(x = roi.rt, color = 'b', linestyle = '--', linewidth=1, ymax=0.8)
    # label the left and right of the ROI
    plt.axvline(x = roi.rt_seq[0], color = 'black', linestyle = '--', linewidth=0.5, ymax=0.8)
    plt.axvline(x = roi.rt_seq[-1], color = 'black', linestyle = '--', linewidth=0.5, ymax=0.8)
    plt.xlabel("Retention Time (min)", fontsize=18, fontname='Arial')
    plt.ylabel("Intensity", fontsize=18, fontname='Arial')
    plt.xticks(fontsize=14, fontname='Arial')
    plt.yticks(fontsize=14, fontname='Arial')
    plt.text(eic_rt[0], max_int*1.1, "m/z = {:.4f}".format(roi.mz), fontsize=11, fontname='Arial')
    plt.text(eic_rt[0]+(eic_rt[-1]-eic_rt[0])*0.2, max_int*1.1, "G-score = {:.2f}".format(roi.gaussian_similarity), fontsize=11, fontname='Arial', color="blue")
    plt.text(eic_rt[0]+(eic_rt[-1]-eic_rt[0])*0.4, max_int*1.1, "N-score = {:.2f}".format(roi.noise_level), fontsize=11, fontname='Arial', color="red")
    plt.text(eic_rt[0]+(eic_rt[-1]-eic_rt[0])*0.6,max_int*1.1, d.file_name, fontsize=11, fontname='Arial', color="gray")

    if output:
        plt.savefig(output, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_hist(arr, bins, x_label, y_label):

    plt.figure(figsize=(6, 3))
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.hist(arr, bins=bins, color='lightgrey', edgecolor='black', linewidth=0.5)
    plt.xlabel(x_label, fontsize=18, fontname='Arial')
    plt.ylabel(y_label, fontsize=18, fontname='Arial')
    plt.xticks(fontsize=14, fontname='Arial')
    plt.yticks(fontsize=14, fontname='Arial')

    plt.show()


def mirror_ms2_from_scans(scan1, scan2, output=False):
    """
    Plot a mirror image of two MS2 spectra for comparison.
    """

    if scan1.level == 2 and scan2.level == 2:
        mirror_ms2(precursor_mz1=scan1.precursor_mz, precursor_mz2=scan2.precursor_mz, peaks1=scan1.peaks, peaks2=scan2.peaks, output=output)


def mirror_ms2(precursor_mz1, precursor_mz2, peaks1, peaks2, annotation=None, score=None, output=False):

    plt.figure(figsize=(10, 3))
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    # plot precursor
    plt.vlines(x = precursor_mz1, ymin = 0, ymax = 1, color="cornflowerblue", linewidth=1.5, linestyles='dashed')
    plt.vlines(x = precursor_mz2, ymin = 0, ymax = -1, color="lightcoral", linewidth=1.5, linestyles='dashed')

    # plot fragment ions
    plt.vlines(x = peaks1[:, 0], ymin = 0, ymax = peaks1[:, 1] / np.max(peaks1[:, 1]), color="blue", linewidth=1.5)
    plt.vlines(x = peaks2[:, 0], ymin = 0, ymax = -peaks2[:, 1] / np.max(peaks2[:, 1]), color="red", linewidth=1.5)

    xmax = max([precursor_mz1, precursor_mz2])*1.2
    # plot zero line
    plt.hlines(y = 0, xmin = 0, xmax = xmax, color="black", linewidth=1.5)
    plt.xlabel("m/z, Dalton", fontsize=18, fontname='Arial')
    plt.ylabel("Intensity", fontsize=18, fontname='Arial')
    plt.xticks(fontsize=14, fontname='Arial')
    plt.yticks(fontsize=14, fontname='Arial')

    # note name and similarity score
    plt.text(xmax*0.9, 0.9, "Experiment", fontsize=12, fontname='Arial', color="grey")
    plt.text(xmax*0.9, -0.9, "Database", fontsize=12, fontname='Arial', color="grey")
    plt.text(0, 0.9, "similarity = {:.3f}".format(score), fontsize=12, fontname='Arial', color="blue")
    plt.text(0, -0.95, annotation, fontsize=12, fontname='Arial', color="black")

    if output:
        plt.savefig(output, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def mirror_ms2_db(f, output=False):

    precursor_mz1 = f.mz
    precursor_mz2 = f.matched_precursor_mz
    peaks1 = f.best_ms2.peaks
    peaks2 = f.matched_peaks

    plt.figure(figsize=(10, 3))
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Arial'
    # plot precursor
    plt.vlines(x = precursor_mz1, ymin = 0, ymax = 1, color="cornflowerblue", linewidth=1.5, linestyles='dashed')
    plt.vlines(x = precursor_mz2, ymin = 0, ymax = -1, color="lightcoral", linewidth=1.5, linestyles='dashed')

    # plot fragment ions
    plt.vlines(x = peaks1[:, 0], ymin = 0, ymax = peaks1[:, 1] / np.max(peaks1[:, 1]), color="blue", linewidth=1.5)
    plt.vlines(x = peaks2[:, 0], ymin = 0, ymax = -peaks2[:, 1] / np.max(peaks2[:, 1]), color="red", linewidth=1.5)

    xmax = max([precursor_mz1, precursor_mz2])*1.2
    # plot zero line
    plt.hlines(y = 0, xmin = 0, xmax = xmax, color="black", linewidth=1.5)
    plt.xlabel("m/z, Dalton", fontsize=18, fontname='Arial')
    plt.ylabel("Intensity", fontsize=18, fontname='Arial')
    plt.xticks(fontsize=14, fontname='Arial')
    plt.yticks(fontsize=14, fontname='Arial')

    # note name and similarity score
    plt.text(xmax*0.9, 0.9, "Experiment", fontsize=12, fontname='Arial', color="grey")
    plt.text(xmax*0.9, -0.9, "Database", fontsize=12, fontname='Arial', color="grey")
    plt.text(0, 0.9, "similarity = {:.3f}".format(f.similarity), fontsize=12, fontname='Arial', color="blue")
    plt.text(0, -0.95, f.annotation.lower(), fontsize=12, fontname='Arial', color="black")

    if output:
        plt.savefig(output, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_ms2_matching_from_feature_table(feature_table, params=None, output_dir=None):
    """
    Plot the mirror image of MS2 spectra from a feature table.
    
    Parameters
    ----------
    feature_table : pandas DataFrame
        The feature table.
    params : Params object
        The parameters for the workflow.
    output_dir : str
        The output directory.
    """

    sub_feature_table = feature_table[feature_table["search_mode"] == "identity_search"]
    ms2 = list(sub_feature_table["MS2"])
    matched_ms2 = list(sub_feature_table["matched_MS2"])
    score = list(sub_feature_table["similarity"])
    mz = list(sub_feature_table["m/z"])
    annotations = list(sub_feature_table["annotation"])
    id = list(sub_feature_table["ID"])

    if output_dir is None and params is not None:
        output_dir = params.ms2_matching_dir
    elif output_dir is None and params is None:
        print("Please provide the output directory for MS2 matching plots.")
        return None

    for i in range(len(ms2)):
        peaks1 = _extract_peaks_from_string(ms2[i])
        peaks2 = _extract_peaks_from_string(matched_ms2[i])

        # replace all the special characters to "_"
        a = re.sub(r"[^a-zA-Z0-9]", "_", annotations[i])
        a = re.sub(r"[^a-zA-Z0-9]+", "_", a)
        a = "ID" + "_" + str(id[i]) + "_" + a
        output = os.path.join(output_dir, "{}.png".format(a))
        mirror_ms2(mz[i], mz[i], peaks1, peaks2, annotations[i], score[i], output)


def plot_pca(vecPC1, vecPC2, var_PC1, var_PC2, group_names, output_dir=None):
    
    fig, ax = plt.subplots(figsize=(10,10))
    groups = np.unique(np.array(group_names))
    colors = COLORS[:len(groups)]
    for group, color in zip(groups, colors):
        idxs = np.where(np.array(group_names) == group)
        # No legend will be generated if we don't pass label=species
        confidence_ellipse(vecPC1[idxs], vecPC2[idxs], ax, edgecolor='black', facecolor=color, alpha=0.1, zorder=1, linewidth=1)
        ax.scatter(
            vecPC1[idxs], vecPC2[idxs], label=group,
            s=300, color=color, alpha=0.3
        )
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(width=2, length=8) 

    plt.xlabel("PC 1 ({:.1f} %)".format(var_PC1*100), fontname='Arial', fontsize=30, labelpad=25)
    plt.ylabel("PC 2 ({:.1f} %)".format(var_PC2*100), fontname='Arial', fontsize=30, labelpad=25)
    plt.xticks(fontname='Arial', fontsize=24)
    plt.yticks(fontname='Arial', fontsize=24)
    plt.grid(linestyle=':')
    plt.legend(fontsize=20, loc='upper right', frameon=False)
    plt.rcParams.update({'font.size': 24})

    if output_dir is not None:
        plt.savefig(output_dir, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



COLORS = ["#FF5050", "#0078F0", "#00B050", "#FFC000", "#7030A0", "#FF00FF", "#00B0F0", "#FF0000", "#00FF00", "#0000FF"]



