# select a group of metabolic features for correcting m/z and retention time error
# the selected features should be:
# 1. present in all (most) biological samples
# 2. confirmed by unique m/z, good peak shape, and MS/MS spectra

# input: a list of files

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..raw_data_utils import MSData

import numpy as np

from .._resources import load_core_db

def select_anchors(d: MSData, num: int = 50, anchor_table_path: str = None, 
                   core_db = None, mz_tol: float = 0.01, sim_tol: float = 0.9,
                   return_anchors: bool = False):
    """
    From a single file, select a group of metabolic features that have confirmed identity.
    Identity should be  confirmed by m/z, MS/MS spectrum, and retention time (optional).

    This function uses an internal, small MS/MS spectral database that contains the well-known metabolites that
    are suitable for anchor annotation and matching.

    Parameters
    ----------
    d : MSData
        The MSData object containing the metabolic features.
    num : int
        The largest number of anchors to be selected. Available anchors can be fewer than this number.
    anchor_table_path : str
        Optional path to a user defined table of anchors. This table should contain the m/z and retention time
        values for the anchors.
    core_db : dict
        The internal MS/MS spectral database. If None, the database will be loaded from the default location.
    mz_tol : float
        The m/z tolerance for selecting anchors.
    return_anchors : bool
        If True, return the list of selected anchors. Otherwise, store the selected anchors in the MSData object.
    
    Returns
    -------
    anchors: list
        A list of Feature objects that are selected as anchors.
    """

    # step 1. rank the features by peak height from high to low
    d.features.sort(key=lambda x: x.peak_height, reverse=True)

    # step 2. load the internal MS/MS spectral database if not loaded
    if core_db is None:
        core_db = load_core_db()

    # step 3. anchor selection
    mz_arr = np.array([f.mz for f in d.features])
    for cpd in core_db:
        mz = cpd["isotopes"][0,0]
        matched_idx = np.where(np.abs(mz_arr - mz) < mz_tol)[0]
        if len(matched_idx) > 0:
            for idx in matched_idx:
                d.features[idx].is_anchor = True




def rt_anchor_selection(data_path, num=50, noise_score_tol=0.1, mz_tol=0.01):
    """
    Retention time anchors have unique m/z values and low noise scores. From all candidate features, 
    the top *num* features with the highest peak heights are selected as anchors.

    Parameters
    ----------
    data_path : str
        Absolute directory to the feature tables.
    num : int
        The number of anchors to be selected.
    noise_tol : float
        The noise level for the anchors. Suggestions: 0.3 or lower.
    mz_tol : float
        The m/z tolerance for selecting anchors.

    Returns
    -------
    anchors: list
        A list of anchors (dict) for retention time correction.
    """

    df = pd.read_csv(data_path, sep="\t", low_memory=False)
    # sort by m/z
    df = df.sort_values(by="m/z")
    df.index = range(len(df))
    mzs = df["m/z"].values
    candidates = []
    diff = np.diff(mzs)
    for i in range(1, len(mzs)-1):
        if diff[i-1] > mz_tol and diff[i] > mz_tol and df["noise_score"][i] < noise_score_tol:
            candidates.append(i)
    candidates = np.array(candidates)
    candidates = candidates[np.argsort(df["peak_height"].values[candidates])[-num:]]
    # reverse the order
    candidates = candidates[::-1]
    valid_mzs = mzs[candidates]
    valid_rts = df["RT"].values[candidates]
    
    return valid_mzs, valid_rts


