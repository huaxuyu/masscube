# Author: Huaxu Yu

# A module for data visualization.


# imports
import random
import numpy as np
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import matplotlib.font_manager as fm

from .annotation import extract_signals_from_string


def plot_bpcs(data_list=None, autocolor=False, show_legend=True, output_path=None):
    """
    A function to plot the base peak chromatograms (overlapped) of a list of data.
    
    Parameters
    ----------
    data_list : list of MSData objects
        A list of data to be plotted.
    autocolor : bool
        Whether to automatically assign colors to the data.
    show_legend : bool
        Whether to show the legend.
    output : str
        The output file name.
    """

    if data_list is not None:
        if autocolor:
            color_list = _color_list
        else:
            color_list = ["black"] * len(data_list)

        plt.figure(figsize=(10, 4))
        plt.rcParams['font.size'] = 14
        # check if arial font is available
        if 'Arial' in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = 'Arial'

        for i, d in enumerate(data_list):
            plt.plot(d.ms1_rt_seq, d.bpc_int, color=color_list[i], linewidth=0.5)
            plt.xlabel("Retention Time (min)", fontsize=18)
            plt.ylabel("Intensity", fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        
        if show_legend:
            plt.legend([d.params.file_name for d in data_list], fontsize=10)
        
        if output_path is not None:
            plt.savefig(output_path, dpi=600, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_feature(d, feature, mz_tol=0.005, rt_tol=0.3, rt_range=None, output=False, break_scan=None):
    """
    Function to plot the chromatogram of a feature.

    Parameters
    ----------
    d : MSData object
        The MSData object.
    feature : Feature object
        The feature to be plotted.
    mz_tol : float
        The m/z tolerance.
    rt_tol : float
        The retention time tolerance.
    rt_range : list
        The retention time range.
    output : str
        The output file name.
    break_scan : int
        The scan index to break the chromatogram.
    """

    if rt_range is None:
        rt_range = [feature.rt_seq[0]-rt_tol, feature.rt_seq[-1]+rt_tol]

    eic_rt, eic_signals, eic_scan_idx_arr = d.get_eic_data(target_mz=feature.mz, target_rt=feature.rt, 
                                                                 mz_tol=mz_tol, rt_range=rt_range)
    
    idx_start = np.where(eic_scan_idx_arr == feature.scan_idx_seq[0])[0][0]
    idx_end = np.where(eic_scan_idx_arr == feature.scan_idx_seq[-1])[0][0] + 1
    if break_scan is not None:
        idx_middle = np.where(eic_scan_idx_arr == break_scan)[0][0]
    
    eic_int = eic_signals[:,1]
    eic_int[idx_start:idx_end] = feature.signals[:,1]

    max_int = np.max(eic_int[idx_start:idx_end])

    plt.figure(figsize=(9, 3))
    plt.rcParams['font.size'] = 14
    if 'Arial' in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams['font.family'] = 'Arial'
    plt.plot(eic_rt, eic_int, linewidth=0.5, color="black")
    plt.ylim(0, max_int*1.2)

    if break_scan is not None:
        plt.fill_between(eic_rt[idx_start:(idx_middle+1)], eic_int[idx_start:(idx_middle+1)], color="blue", alpha=0.2)
        plt.fill_between(eic_rt[idx_middle:idx_end], eic_int[idx_middle:idx_end], color="red", alpha=0.2)
    else:
        plt.fill_between(eic_rt[idx_start:idx_end], eic_int[idx_start:idx_end], color="black", alpha=0.2)
    plt.axvline(x = feature.rt, color = 'b', linestyle = '--', linewidth=1, ymax=0.8)
    # label the left and right of the feature
    plt.axvline(x = feature.rt_seq[0], color = 'black', linestyle = '--', linewidth=0.5, ymax=0.8)
    plt.axvline(x = feature.rt_seq[-1], color = 'black', linestyle = '--', linewidth=0.5, ymax=0.8)
    plt.xlabel("Retention Time (min)", fontsize=18)
    plt.ylabel("Intensity", fontsize=18)
    plt.text(eic_rt[0], max_int*1.1, "m/z = {:.4f}".format(feature.mz), fontsize=11)
    plt.text(eic_rt[0]+(eic_rt[-1]-eic_rt[0])*0.2, max_int*1.1, "G-score = {:.2f}".format(feature.gaussian_similarity), fontsize=11, color="blue")
    plt.text(eic_rt[0]+(eic_rt[-1]-eic_rt[0])*0.4, max_int*1.1, "N-score = {:.2f}".format(feature.noise_score), fontsize=11, color="red")
    plt.text(eic_rt[0]+(eic_rt[-1]-eic_rt[0])*0.6, max_int*1.1, "A-score = {:.2f}".format(feature.asymmetry_factor), fontsize=11, color="darkgreen")
    plt.text(eic_rt[0]+(eic_rt[-1]-eic_rt[0])*0.8,max_int*1.1, d.params.file_name, fontsize=7, color="gray")

    if output:
        plt.savefig(output, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def mirror_ms2_signals(s1, s2, output_path=None):
    """
    Plot a mirror image of two MS2 spectra for comparison.

    Parameters
    ----------
    s1 : Scan object
        The first MS2 spectrum.
    s2 : Scan object
        The second MS2 spectrum.
    output_path : str
        The output file name.
    """

    if s1.level == 2 and s2.level == 2:
        mirror_ms2(precursor_mz1=s1.precursor_mz, precursor_mz2=s2.precursor_mz, signals1=s1.signals, signals2=s2.signals, output=output_path)


def mirror_ms2(precursor_mz1, precursor_mz2, signals1, signals2, annotation=None, score=None, output_path=None):
    """
    Plot a mirror image of two MS2 spectra for comparison.

    Parameters
    ----------
    precursor_mz1 : float
        The precursor m/z of the first MS2 spectrum.
    precursor_mz2 : float
        The precursor m/z of the second MS2 spectrum.
    signals1 : numpy array
        The signals of the first MS2 spectrum: [[mz1, int1], [mz2, int2], ...].
    signals2 : numpy array
        The signals of the second MS2 spectrum: [[mz1, int1], [mz2, int2], ...].
    annotation : str
        The annotation of the MS2 spectra.
    score : float
        The similarity score between the two MS2 spectra.
    output_path : str
        The output file name.
    """

    plt.figure(figsize=(10, 3))
    plt.rcParams['font.size'] = 14
    if 'Arial' in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams['font.family'] = 'Arial'
    # plot precursor
    plt.vlines(x = precursor_mz1, ymin = 0, ymax = 1, color="cornflowerblue", linewidth=1.5, linestyles='dashed')
    plt.vlines(x = precursor_mz2, ymin = 0, ymax = -1, color="lightcoral", linewidth=1.5, linestyles='dashed')

    # plot fragment ions
    plt.vlines(x = signals1[:, 0], ymin = 0, ymax = signals1[:, 1] / np.max(signals1[:, 1]), color="blue", linewidth=1.5)
    plt.vlines(x = signals2[:, 0], ymin = 0, ymax = -signals2[:, 1] / np.max(signals2[:, 1]), color="red", linewidth=1.5)

    xmax = max([precursor_mz1, precursor_mz2])*1.2
    # plot zero line
    plt.hlines(y = 0, xmin = 0, xmax = xmax, color="black", linewidth=1.5)
    plt.xlabel("m/z, Dalton", fontsize=18)
    plt.ylabel("Intensity", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # note name and similarity score
    plt.text(xmax*0.9, 0.9, "Experiment", fontsize=12, color="grey")
    plt.text(xmax*0.9, -0.9, "Database", fontsize=12, color="grey")
    if score is not None:
        plt.text(0, 0.9, "similarity = {:.3f}".format(score), fontsize=12, color="blue")
    if annotation is not None:
        plt.text(0, -0.95, annotation, fontsize=12, color="black")

    if output_path is not None:
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
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
    id = list(sub_feature_table["feature_ID"])

    if output_dir is None and params is not None:
        output_dir = params.ms2_matching_dir
    elif output_dir is None and params is None:
        print("Please provide the output directory for MS2 matching plots.")
        return None

    for i in tqdm(range(len(ms2))):
        peaks1 = extract_signals_from_string(ms2[i])
        peaks2 = extract_signals_from_string(matched_ms2[i])

        # replace all the special characters to "_"
        a = re.sub(r"[^a-zA-Z0-9]", "_", annotations[i])
        a = re.sub(r"[^a-zA-Z0-9]+", "_", a)
        a = "ID" + "_" + str(id[i]) + "_" + a
        # only keep the first 20 characters
        a = a[:20]
        output = os.path.join(output_dir, "{}.png".format(a))
        mirror_ms2(mz[i], mz[i], peaks1, peaks2, annotations[i], score[i], output)


def plot_lowess_normalization(arr, fit_curve, arr_new, sample_idx, qc_idx, n, id=None, dpi=100, output_dir=None):
    """
    Plot the lowess normalization results by highlighting the QC samples, smoothing curve, and 95% confidence interval.

    Parameters
    ----------
    arr : numpy array
        The original intensity array.
    fit_curve : numpy array
        The smoothed curve by LOWESS.
    arr_new : numpy array
        The normalized intensity array.
    sample_idx : numpy array of bool
        Blank samples are false, while qc and real samples are true. It's length is n.
    qc_idx : numpy array
        QC samples are true, while blank and real samples are false. It's length is n.
    n : int
        The number of samples.
    id : int
        The feature ID.
    dpi : int
        The dpi of the output image.
    output_dir : str
        The output directory.
    """
    
    v = np.arange(n)
    
    plt.rcParams['font.size'] = 20
    if 'Arial' in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(20,8))

    plt.subplot(2, 1, 1)
    plt.title("Before normalization")
    plt.ylabel("Intensity")
    plt.plot(v[sample_idx], arr[sample_idx], 'o', markersize=4, color='grey')
    plt.plot(v[qc_idx], arr[qc_idx], 'o', markersize=6, color='red')
    plt.plot(v, fit_curve, '--', color='blue', linewidth=2)
    plt.legend(["Sample", "QC", "LOWESS"])
    plt.ylim(-300, np.max(arr[qc_idx]) * 1.1)
    plt.xlim(-n*0.01, n*1.2)
    plt.text(n*0.1, np.max(arr[qc_idx]) * 1.15, "Feature ID: %d" % id, color='grey')
    rsd = np.std(arr[qc_idx]) / np.mean(arr[qc_idx]) * 100
    plt.text(n, np.max(arr[qc_idx]) * 1.15, "QC RSD: %.2f%%" % rsd, color='grey')

    plt.subplot(2, 1, 2)
    plt.title("After normalization")
    plt.xlabel("Analytical order")
    plt.ylabel("Intensity")
    plt.plot(v[sample_idx], arr_new[sample_idx], 'o', markersize=4, color='grey')
    plt.plot(v[qc_idx], arr_new[qc_idx], 'o', markersize=6, color='red')
    plt.ylim(-300, np.max(arr_new[qc_idx]) * 1.1)
    # use color band to show the 95% confidence interval
    y_up = np.median(arr_new[qc_idx]) + 1.96 * np.std(arr_new[qc_idx])
    y_down = np.median(arr_new[qc_idx]) - 1.96 * np.std(arr_new[qc_idx])
    plt.fill_between(v, y_down, y_up, color='lightblue', alpha=0.5)
    rsd = np.std(arr_new[qc_idx]) / np.mean(arr_new[qc_idx]) * 100
    plt.text(n, np.max(arr_new[qc_idx]) * 1.15, "QC RSD: %.2f%%" % rsd, color='grey')
    plt.legend(["Sample", "QC", "95% CI"])
    plt.xlim(-n*0.01, n*1.2)

    plt.subplots_adjust(hspace=0.5)
    
    if output_dir is not None:
        file_name = os.path.join(output_dir, "feature_{}_normalization.png".format(id))
        plt.savefig(file_name, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_pca(vecPC1, vecPC2, var_PC1, var_PC2, group_names, colors=None, plot_order=None, output_dir=None):
    
    fig, ax = plt.subplots(figsize=(10,10))
    # set font to arial
    if 'Arial' in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams['font.family'] = 'Arial'
    groups = np.unique(np.array(group_names))
    if colors is None:
        colors = COLORS[:len(groups)]
    
    if plot_order is not None:
        groups = [groups[i] for i in plot_order]
        colors = [colors[i] for i in plot_order]

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

    plt.xlabel("PC 1 ({:.1f} %)".format(var_PC1*100), fontsize=30, labelpad=25)
    plt.ylabel("PC 2 ({:.1f} %)".format(var_PC2*100), fontsize=30, labelpad=25)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(linestyle=':')
    plt.legend(fontsize=20, loc='upper right', frameon=True)
    plt.rcParams.update({'font.size': 24})

    if output_dir is not None:
        plt.savefig(output_dir, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


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
_color_list = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

def random_color_generator():
    # set seed
    color = random.choice(list(mcolors.CSS4_COLORS.keys()))
    return color