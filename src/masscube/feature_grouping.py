# Author: Huaxu Yu

# Group features from the same compound including isotopes, in-source fragments and adducts.

# important note:
# don't use 1.003355 * charge_state here, because only carbon isotopes mass diff is larger than 1 Da

# imports
import numpy as np
import os
from tqdm import tqdm
import pickle

from .params import Params
from .raw_data_utils import read_raw_file_to_obj, MSData
from .utils_functions import extract_signals_from_string, POS_ADDUCTS, NEG_ADDUCTS


def group_features_after_alignment(features: list, params: Params):
    """
    For the untargeted metabolomics workflow only. This function requires
    to reload the raw data to examine the scan-to-scan correlation between features. 
    
    Therefore, a project directory is required to reload the raw MS data.
    
    The annotated feature group is stored in the `feature_group_id` attribute of the AlignedFeature objects.
    Feautures that are grouped together will have the same feature_group_id.

    Parameters
    ----------
    features: list
        A list of AlignedFeature objects.
    params: Params object
        A Params object that contains the parameters for feature grouping.
    """

    # check if features are provided
    if features is None or len(features) == 0:
        print("No features found for aligned feature grouping.")
        return None

    # sort features by the highest peak height from high to low
    features.sort(key=lambda x: x.highest_intensity, reverse=True)
    
    # initialization
    mz_arr = np.array([f.mz for f in features])
    rt_arr = np.array([f.rt for f in features])
    is_grouped = np.zeros(len(features), dtype=bool)
    feature_group_id = 1
    default_adduct = "[M+H]+" if params.ion_mode.lower() == "positive" else "[M-H]-"

    # load RT correction models if available
    if params.correct_rt and os.path.exists(os.path.join(params.project_dir, "rt_correction_models.pkl")):
        with open(os.path.join(params.project_dir, "rt_correction_models.pkl"), "rb") as f:
            rt_cor_functions = pickle.load(f)
    else:
        rt_cor_functions = {}

    # cache for raw data objects
    from collections import OrderedDict
    RAW_CACHE_MAXSIZE = 20
    raw_data_cache = OrderedDict()

    def get_raw(ref_file: str):
        # hit
        if ref_file in raw_data_cache:
            d = raw_data_cache.pop(ref_file)
            raw_data_cache[ref_file] = d  # mark as most recently used
            return d

        # miss → load
        fn = os.path.join(params.tmp_file_dir, ref_file + ".mzpkl")
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Reference file {fn} not found.")

        d = read_raw_file_to_obj(fn)

        func = rt_cor_functions.get(ref_file)
        if func is not None:
            d.correct_retention_time(func)

        # evict LRU if full
        if len(raw_data_cache) >= RAW_CACHE_MAXSIZE:
            _, old_d = raw_data_cache.popitem(last=False)
            del old_d

        raw_data_cache[ref_file] = d
        
        return d

    # find isotopes, adducts and in-source fragments for each feature
    for i, f in enumerate(tqdm(features)):

        # skip if already grouped
        if is_grouped[i]:
            continue
        
        # mark as grouped
        is_grouped[i] = True
        f.feature_group_id = feature_group_id
        f.isotope_state = "M+0"
        # check if adduct type is assigned by MS/MS library search
        # if not, assign default adduct type based on ionization mode
        if f.adduct_type is None:
            f.adduct_type = default_adduct

        d = get_raw(f.reference_file)
        
        # feature EIC needs to be extracted again
        _, eic_a, _ = d.get_eic_data(f.mz, f.rt, mz_tol=0.01, rt_tol=0.2)

        # in rare cases, the EIC may not be found owing to large RT shift
        if len(eic_a) == 0:
            feature_group_id += 1
            continue

        k = np.argmin(np.abs(d.ms1_time_arr-f.rt))

        f.isotope_signals = find_isotope_signals(mz=f.mz, signals=d.scans[d.ms1_idx[k]].signals, 
                                                 mz_tol=params.mz_tol_feature_grouping, 
                                                 rel_int_limit=params.isotope_rel_int_limit)
        
        # masks
        rt_mask = np.abs(rt_arr - f.rt) < params.rt_tol_feature_grouping
        base_mask = rt_mask & (~is_grouped)
        # generate all possible ions for grouping
        if f.ms2 is not None:
            ms2_signals = extract_signals_from_string(f.ms2)
        ion_dict = enumerate_source_ions(mz=f.mz, ms2=ms2_signals if f.ms2 is not None else None, 
                                         adduct=f.adduct_type, ion_mode=params.ion_mode)
        
        for key, val in ion_dict.items():
            mz_mask = np.abs(mz_arr - val) < params.mz_tol_feature_grouping
            mask = base_mask & mz_mask
            matched_num = np.sum(mask)

            # one isotope mz can be matched to multiple features because of high-resolution MS
            if key.startswith("M+"):
                is_grouped[mask] = True
                for j in np.where(mask)[0]:
                    features[j].feature_group_id = f.feature_group_id
                    features[j].is_isotope = True
                    features[j].adduct_type = f.adduct_type
                    features[j].isotope_state = key
                    features[j].isotope_signals = f.isotope_signals
                continue

            if matched_num > 0:
                # prioritize the one with the lowest RT difference
                v = np.where(mask)[0]
                vi = v[np.argmin(np.abs(rt_arr[v] - f.rt))]
                _, eic_b, _ = d.get_eic_data(mz_arr[vi], f.rt, mz_tol=0.01, rt_tol=0.2)
                scan_scan_cor = scan_to_scan_cor_intensity(eic_a[:, 1], eic_b[:, 1])
                
                if scan_scan_cor > params.scan_scan_cor_tol:
                    is_grouped[vi] = True
                    features[vi].feature_group_id = f.feature_group_id
                    features[vi].scan_scan_cor = scan_scan_cor
                    # for adducts:
                    if not key.startswith("ISF"):
                        features[vi].adduct_type = key
                    # for in-source fragments:
                    else:
                        features[vi].is_in_source_fragment = True
                        features[vi].adduct_type = f.adduct_type
                    # find isotopes for this ion
                    features[vi].isotope_signals = find_isotope_signals(features[vi].mz,
                                                                        signals=d.scans[d.ms1_idx[k]].signals,
                                                                        mz_tol=params.mz_tol_feature_grouping,
                                                                        rel_int_limit=params.isotope_rel_int_limit)
                    # retrieve the isotope signals from the feature list
                    iso_deep_dic = enumerate_source_ions(features[vi].mz, ms2=None, adduct=None, ion_mode=None, isotope_only=True)
                    for key_deep, mz_val_deep in iso_deep_dic.items():
                        mz_mask_deep = np.abs(mz_arr - mz_val_deep) < params.mz_tol_feature_grouping
                        # use the latest grouping status
                        mask_deep = rt_mask & (~is_grouped) & mz_mask_deep
                        for j in np.where(mask_deep)[0]:
                            # need to refine by relative intensity
                            _, eic_c, _ = d.get_eic_data(mz_arr[j], f.rt, mz_tol=0.01, rt_tol=0.2)
                            if np.max(eic_c[:,1]) < np.max(eic_b[:,1]) * params.isotope_rel_int_limit:
                                is_grouped[j] = True
                                features[j].feature_group_id = f.feature_group_id
                                features[j].is_isotope = True
                                features[j].adduct_type = features[vi].adduct_type
                                features[j].isotope_state = key_deep
                                features[j].isotope_signals = features[vi].isotope_signals
                                features[j].is_in_source_fragment = features[vi].is_in_source_fragment
            
        feature_group_id += 1


def group_features_single_file(d: MSData, rt_tol: float = 0.05) -> None:
    """
    Group isotopes, adducts, and in-source fragments from a single file 
    based on the m/z, retention time, MS/MS, scan-to-scan correlation and relative intensity.
    The feature_group_id attribute of the Feature objects in the MSData object will be updated.

    Parameters
    ----------
    d: MSData object
        Contains the raw data and the features to be grouped.
    rt_tol: float
        The retention time tolerance for grouping features (0.05 min by default).
    """

    # check if features have been detected
    if d.features is None or len(d.features) == 0:
        raise ValueError("No features found in the MSData object for feature grouping.")

    # sort features by peak height from high to low
    d.features.sort(key=lambda x: x.peak_height, reverse=True)
    
    # initialization
    mz_arr = np.array([f.mz for f in d.features])
    rt_arr = np.array([f.rt for f in d.features])
    is_grouped = np.zeros(len(d.features), dtype=bool)
    feature_group_id = 1
    default_adduct = "[M+H]+" if d.params.ion_mode.lower() == "positive" else "[M-H]-"

    # find isotopes, adducts and in-source fragments for each feature
    for i, f in enumerate(tqdm(d.features)):

        # skip if already grouped
        if is_grouped[i]:
            continue
        
        # mark as grouped
        is_grouped[i] = True
        f.feature_group_id = feature_group_id
        f.isotope_state = "M+0"
        # check if adduct type is assigned by MS/MS library search
        # if not, assign default adduct type based on ionization mode
        if f.adduct_type is None:
            f.adduct_type = default_adduct

        peak_main = f.signals[:,1]
        rt_range  = [f.rt_seq[0] - 1e-6, f.rt_seq[-1] + 1e-6]

        # extract isotope signals from the apex scan
        f.isotope_signals = find_isotope_signals(mz=f.mz, signals=d.scans[f.scan_idx].signals, 
                                                 mz_tol=d.params.mz_tol_feature_grouping, 
                                                 rel_int_limit=d.params.isotope_rel_int_limit)
        
        # masks
        # intensity mask is not needed since features have been sorted by peak height
        rt_mask = np.abs(rt_arr - f.rt) < rt_tol
        base_mask = rt_mask & (~is_grouped)

        # generate all possible ions for grouping
        ion_dict = enumerate_source_ions(mz=f.mz, ms2=f.ms2.signals if f.ms2 is not None else None, 
                                         adduct=f.adduct_type, ion_mode=d.params.ion_mode)

        for key, mz_val in ion_dict.items():
            mz_mask = np.abs(mz_arr - mz_val) < d.params.mz_tol_feature_grouping
            mask = base_mask & mz_mask
            matched_num = np.sum(mask)

            # one isotope mz can be matched to multiple features because of high-resolution MS
            if key.startswith("M+"):
                is_grouped[mask] = True
                for j in np.where(mask)[0]:
                    d.features[j].feature_group_id = f.feature_group_id
                    d.features[j].is_isotope = True
                    d.features[j].adduct_type = f.adduct_type
                    d.features[j].isotope_state = key
                    d.features[j].isotope_signals = f.isotope_signals
                continue
            
            # only one matched is allowed for adducts and in-source fragments
            if matched_num > 0:
                # prioritize the one with the lowest RT difference
                v = np.where(mask)[0]
                vi = v[np.argmin(np.abs(rt_arr[v] - f.rt))]
                _, eic, _ = d.get_eic_data(mz_arr[vi], f.rt, rt_range=rt_range)
                scan_scan_cor = scan_to_scan_cor_intensity(peak_main, eic[:, 1])
                
                if scan_scan_cor > d.params.scan_scan_cor_tol:
                    is_grouped[vi] = True
                    d.features[vi].feature_group_id = f.feature_group_id
                    d.features[vi].scan_scan_cor = scan_scan_cor
                    # for adducts:
                    if not key.startswith("ISF"):
                        d.features[vi].adduct_type = key
                    # for in-source fragments:
                    else:
                        d.features[vi].is_in_source_fragment = True
                        d.features[vi].adduct_type = f.adduct_type
                    # find isotopes for this ion
                    d.features[vi].isotope_signals = find_isotope_signals(d.features[vi].mz,
                                                                          signals=d.scans[d.features[vi].scan_idx].signals,
                                                                          mz_tol=d.params.mz_tol_feature_grouping,
                                                                          rel_int_limit=d.params.isotope_rel_int_limit)
                    # retrieve the isotope signals from the feature list
                    iso_deep_dic = enumerate_source_ions(d.features[vi].mz, ms2=None, adduct=None, ion_mode=None, isotope_only=True)
                    for key_deep, mz_val_deep in iso_deep_dic.items():
                        mz_mask_deep = np.abs(mz_arr - mz_val_deep) < d.params.mz_tol_feature_grouping
                        # use the latest grouping status
                        mask_deep = rt_mask & (~is_grouped) & mz_mask_deep
                        for j in np.where(mask_deep)[0]:
                            # need to refine by relative intensity
                            if d.features[j].peak_height < d.features[vi].peak_height * d.params.isotope_rel_int_limit:
                                is_grouped[j] = True
                                d.features[j].feature_group_id = feature_group_id
                                d.features[j].is_isotope = True
                                d.features[j].adduct_type = d.features[vi].adduct_type
                                d.features[j].isotope_state = key_deep
                                d.features[j].isotope_signals = d.features[vi].isotope_signals

        feature_group_id += 1


def enumerate_source_ions(mz: float, ms2: np.array, adduct: str, ion_mode: str, iso_num: int = 4, isotope_only: bool = False) -> dict:
    """
    Calculate all possible isotopes, adducts, and in-source fragments m/z values for a given m/z.
    The calculation does not compute the isotopes of the adducts and in-source fragments.

    Parameters
    ----------
    mz: float
        The m/z value of the feature.
    ms2: numpy.array
        The MS2 signals as [[m/z, intensity], ...]
    adduct: str
        The adduct form of the feature.
    ion_mode: str
        The ionization mode. "positive" or "negative".
    iso_num: int
        The number of isotopes to consider after the monoisotopic peak.
    isotope_only: bool
        Whether to only calculate isotopes. Default is False.

    Returns
    -------
    dict
        A dictionary with keys as adduct/in-source fragment forms and values as the corresponding m/z values.
    """

    ion_dict = {}
    
    # isotopes
    for i in range(1, iso_num):
        ion_dict[f'M+{i}'] = mz + 1 + (i-1) * 1.003355
    
    if isotope_only:
        return ion_dict

    # adducts
    if ion_mode.lower() == "positive":
        adduct_dic = POS_ADDUCTS
    elif ion_mode.lower() == "negative":
        adduct_dic = NEG_ADDUCTS
    
    if adduct in adduct_dic.keys():
        a = adduct_dic[adduct]
        base_mz = (mz - a.mass_shift) * a.charge / a.mol_multiplier
        for key, val in adduct_dic.items():     
            if key != adduct and val.considered:
                ion_dict[key] = (base_mz * val.mol_multiplier + val.mass_shift) / val.charge

    # in-source fragments
    if ms2 is not None:
        if type(ms2) == str:
            ms2 = extract_signals_from_string(ms2)
        for i, p in enumerate(ms2):
            ion_dict[f'ISF_{i+1}'] = p[0]
    
    return ion_dict


def find_isotope_signals(mz: float, signals: np.array, mz_tol: float = 0.01, 
                         num: int = 5, rel_int_limit: float = 0.7) -> np.array:
    """
    Find isotopes from MS1 signals. Only single charged isotopes are considered.

    Parameters
    ----------
    mz: float
        The m/z value of the feature.
    signals: numpy.array
        The MS1 signals as [[m/z, intensity], ...]
    mz_tol: float
        The m/z tolerance to find isotopes.
    num: int
        The maximum number of isotopes to be found.
    rel_int_limit: float
        Isotope's intensity cannot be higher than rel_int_limit * intensity of the monoisotopic peak.

    Returns
    -------
    numpy.array
        The m/z and intensity of the isotopes.
    """

    mzs = signals[:, 0]
    intens = signals[:, 1]

    idx = np.abs(mzs - mz).argmin()
    limit = intens[idx] * rel_int_limit

    targets = mz + np.arange(num, dtype=mzs.dtype)
    within = np.any(np.abs(mzs[None, :] - targets[:, None]) < mz_tol, axis=0)
    good_int = intens < limit

    keep = within & good_int
    
    return signals[keep]


"""
Helper functions and constants
==============================
"""

def scan_to_scan_cor_intensity(a: np.array, b: np.array) -> float:
    """
    Calculate the scan-to-scan correlation (Pearson correlation) between two intensity arrays.

    Parameters
    ----------
    a: np.array
        Intensity array of the first m/z
    b: np.array
        Intensity array of the second m/z
    
    Returns
    -------
    float
        The scan-to-scan correlation (Pearson correlation) between the two intensity arrays.
    """

    v = (a>0) & (b>0)

    # if the commonly detected points are less than 3, ignore the correlation calculation and return 1.0
    if np.sum(v) < 3:
        return 1.0

    return np.corrcoef(a[v], b[v])[0, 1]