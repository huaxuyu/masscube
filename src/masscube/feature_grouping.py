# Author: Hauxu Yu

# A module to group features (unique m/z and reteniton time) originated from the 
# same compound. It will annotate isotopes, in-source fragments and adducts.

# Development plan: annotation of multimers are to be performed in the future.

# imports
import numpy as np

from .params import Params


def group_features_after_alignment(features, params: Params):
    """
    Group features after alignment based on the reference file. This function requires
    to reload the raw data to examine the scan-to-scan correlation between features.

    Parameters
    ----------
    features: list
        A list of AlignedFeature objects.
    params: Params object
        A Params object that contains the parameters for feature grouping.
    """

    pass

def annotate_feature_groups(d, adduct_dict=None, primary_ions=None, rt_tol=0.1):
    """
    Assign feature grouping to features by annotating isotopes, in-source fragments, 
    and adducts.

    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object that contains the detected rois to be grouped.
    adduct_dict: dict
        A dictionary that contains the adducts to be annotated. If None, the default
        adducts will be used.
    primary_ions: list
        A list of possible primary ions to be considered. The primary ion has the highest 
        intensity in the group. By default, the primary ions for
        1. positve mode: [M+H]+, [M+Na]+, [M+NH4]+ and [M+H-H2O]+
        2. negative mode: [M-H]-, [M-H-H2O]-
    rt_tol: float
        The retention time tolerance to find isotopes and in-source fragments.
    """

    # prioritize the features with higher intensity
    d.features.sort(key=lambda x: x.peak_height, reverse=True)
    feature_mz = np.array([f.mz for f in d.features])
    feature_rt = np.array([f.rt for f in d.features])
    group_to_assign = np.ones(len(d.features), dtype=bool)

    feature_group_id = 0

    for i, feature in enumerate(d.features):

        if group_to_assign[i]:
            continue
        
        # set feature group id
        feature.feature_group_id = feature_group_id

        # find isotope first
        

        # list all possible adducts and in-source fragments. if any of them is found, further find their isotopes
        adduct_form = find_adduct_form(feature.mz, d.scans[feature.scan_idx].signals, 
                                       d.params.ion_mode, d.params.mz_tol_ms1)
        search_dict = generate_search_dict(adduct_form, adduct_dict, primary_ions)

        # find the features that are in the same group
        for key in search_dict.keys():
            v1 = np.abs(feature_mz - search_dict[key]) < d.params.mz_tol_ms1
            v2 = np.abs(feature_rt - feature.rt) < rt_tol
            v = np.where(np.logical_and(v1, v2, group_to_assign))[0]

            if len(v) == 0:
                continue
            else:
                # check which has the largest scan-to-scan correlation
                for vi in v:
                    if scan_to_scan_correlation(feature, d.features[vi]) > d.params.ppr:
                        group_to_assign[vi] = False
                        d.features[vi].feature_group_id = feature_group_id
                        # find isotopes for this ion

        feature_group_id += 1
    pass


def annotate_feature_groups_aligned(d, adduct_dict=None, primary_ions=None, rt_tol=0.1):
    pass



def generate_search_dict(feature, adduct_form, ion_mode):
    """
    Generate a search dictionary for the feature grouping.

    Parameters
    ----------
    feature: Feature object
        The feature object to be grouped.
    adduct_form: str
        The adduct form of the feature.
    ion_mode: str
        The ionization mode. "positive" or "negative".

    Returns
    -------
    dict
        A dictionary that contains the possible adducts and in-source fragments.
    """

    search_dict = {}
    if ion_mode.lower() == "positive":
        dic = ADDUCT_POS
    elif ion_mode.lower() == "negative":
        dic = ADDUCT_NEG
    
    base_mz = feature.mz - dic['adduct_form'][0]

    # possible adducts
    for key in dic.keys():
        if key != adduct_form:
            search_dict[key] = base_mz*dic[key][1] + dic[key][0]
    
    # possible in-source fragments
    if feature.ms2 is not None:
        for i, p in enumerate(feature.ms2.signals):
            search_dict[f'ISF_{i}'] = p[0]
    
    return search_dict


def find_adduct_form(mz, signals, mode="positive", mz_tol=0.01):
    """
    Find the most likely adduct form of a feature based on its m/z and MS1 signals.
    By default, the postive mode adducts considered are [M+H]+, [M+Na]+, [M+NH4]+, and [M+H-H2O]+, 
    and the negative mode adducts considered are [M-H]-, [M-H-H2O]-, [M+FA]-, and [M+Ac]-.

    Parameters
    ----------
    mz: float
        The m/z value of the feature.
    signals: np.array
        The MS1 signals as [[m/z, intensity], ...]
    mode: str
        The ionization mode. "positive" or "negative".
    mz_tol: float
        The m/z tolerance to find the adduct.
    
    Returns
    -------
    str
        The most likely adduct.
    """

    if mode.lower() == "positive":
        adduct_dict = _adduct_pos_primary
    elif mode.lower() == "negative":
        adduct_dict = _adduct_neg_primary
    else:
        return None

    diff = mz - signals[:, 0]
    scores = np.zeros(len(adduct_dict.keys()))
    for i, values in enumerate(adduct_dict.values()):
        for v in values:
            scores[i] += np.sign(np.sum(np.abs(diff - v) < mz_tol)) * signals[:,1][np.argmin(np.abs(diff - v))]
    
    return list(adduct_dict.keys())[np.argmax(scores)]
    

def annotate_isotope(d, mz_tol=0.015, rt_tol=0.1, valid_intensity_ratio_range=[0.001, 1.2], charge_state_range=[1,2]):
    """
    A function to annotate isotopes in the MS data.
    
    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object.
    mz_tol: float
        The m/z tolerance to find isotopes.
    rt_tol: float
        The retention time tolerance to find isotopes.
    valid_intensity_ratio_range: list
        The valid intensity ratio range between isotopes.
    charge_state_range: list (not used)
        The charge state range of the isotopes. [lower, upper]
    """

    # rank the rois (d.rois) in each file by m/z
    d.rois.sort(key=lambda x: x.mz)

    for r in d.rois:

        if r.is_isotope:
            continue

        r.isotope_int_seq = [r.peak_height]
        r.isotope_mz_seq = [r.mz]

        last_mz = r.mz

        # check if the currect ion is double charged
        r.charge_state = 1
        target_mz = r.mz + 1.003355/2
        v = np.where(np.logical_and(np.abs(d.roi_mz_seq - target_mz) < mz_tol, np.abs(d.roi_rt_seq - r.rt) < rt_tol))[0]
        if len(v) > 0:
            # an isotope can't have intensity 1.2 fold or higher than M0 or 1% lower than the M0
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height < valid_intensity_ratio_range[1]*r.peak_height]
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height > 0.01*r.peak_height]
            if len(v) > 0:
                r.charge_state = 2

        isotope_id_seq = []
        target_mz = r.mz + 1.003355/r.charge_state
        total_int = r.peak_height

        # find roi using isotope list
        i = 0
        while i < 5:   # maximum 5 isotopes
            # if isotpoe is not found in two daltons, stop searching
            if target_mz - last_mz > 2.2:
                break

            v = np.where(np.logical_and(np.abs(d.roi_mz_seq - target_mz) < mz_tol, np.abs(d.roi_rt_seq - r.rt) < rt_tol))[0]

            if len(v) == 0:
                i += 1
                continue

            # an isotope can't have intensity 1.2 fold or higher than M0 or 0.1% lower than the M0
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height < valid_intensity_ratio_range[1]*r.peak_height]
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height > valid_intensity_ratio_range[0]*total_int]

            if len(v) == 0:
                i += 1
                continue
            
            # for high-resolution data, C and N isotopes can be separated and need to be summed
            total_int = np.sum([d.rois[vi].peak_height for vi in v])
            last_mz = np.mean([d.rois[vi].mz for vi in v])

            r.isotope_mz_seq.append(last_mz)
            r.isotope_int_seq.append(total_int)
            
            for vi in v:
                d.rois[vi].is_isotope = True
                isotope_id_seq.append(d.rois[vi].id)

            target_mz = last_mz + 1.003355/r.charge_state
            i += 1

        r.charge_state = get_charge_state(r.isotope_mz_seq)
        r.isotope_id_seq = isotope_id_seq


def annotate_in_source_fragment(d, mz_tol=0.01, rt_tol=0.05):
    """
    Function to annotate in-source fragments in the MS data.
    Only [M+O] (roi.is_isotope==True) will be considered in this function.
    Two criteria are used to annotate in-source fragments:
    1. The precursor m/z of the child is in the MS2 spectrum of the parent.
    2. Peak-peak correlation > params.ppr.
    
    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object that contains the detected rois to be grouped.
    """

    # sort ROI by m/z from high to low
    d.rois.sort(key=lambda x: x.mz)

    roi_to_label = np.ones(len(d.rois), dtype=bool)

    # isotopes or rois with length < 4 can't be parent or child
    for idx, r in enumerate(d.rois):
        if r.is_isotope or len(r.scan_idx_seq) < 4:
            roi_to_label[idx] = False
    
    # find in-source fragments
    for idx in range(len(d.rois)-1, -1, -1):

        r = d.rois[idx]

        # isotpes and in-source fragments cannot be the parent
        if not roi_to_label[idx] or r.best_ms2 is None:
            continue

        for m in r.best_ms2.peaks[:, 0]:

            v = np.logical_and(np.abs(d.roi_mz_seq - m) < mz_tol, np.abs(d.roi_rt_seq - r.rt) < rt_tol)
            v = np.where(np.logical_and(v, roi_to_label))[0]

            # an in-source fragment can't have intensity 3 fold or higher than the parent
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height < 3*r.peak_height]

            if len(v) == 0:
                continue

            if len(v) > 1:
                # select the one with the lowest RT difference
                v = v[np.argmin(np.abs(d.roi_rt_seq[v] - r.rt))]
            else:
                v = v[0]

            if scan_to_scan_correlation(r, d.rois[v]) > d.params.ppr:
                roi_to_label[v] = False
                d.rois[v].is_in_source_fragment = True
                d.rois[v].isf_parent_roi_id = r.id
                r.isf_child_roi_id.append(d.rois[v].id)


def annotate_adduct(d, mz_tol=0.01, rt_tol=0.05):
    """
    A function to annotate adducts from the same compound.

    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object that contains the detected rois to be grouped.
    """

    d.rois.sort(key=lambda x: x.mz)
    roi_to_label = np.ones(len(d.rois), dtype=bool)

    # isotopes, in-source fragments, and rois with length < 4 can't be parent or child
    for idx, r in enumerate(d.rois):
        if r.is_isotope or r.is_in_source_fragment or len(r.scan_idx_seq) < 4:
            roi_to_label[idx] = False

    if d.params.ion_mode.lower() == "positive":
        default_adduct = "[M+H]+"
        adduct_mass_diffence = _ADDUCT_MASS_DIFFERENCE_POS_AGAINST_H
        adduct_mass_diffence['[2M+H]+'] = r.mz - 1.007276
        adduct_mass_diffence['[3M+H]+'] = 2*(r.mz - 1.007276)

    elif d.params.ion_mode.lower() == "negative":
        default_adduct = "[M-H]-"
        adduct_mass_diffence = _ADDUCT_MASS_DIFFERENCE_NEG_AGAINST_H
        adduct_mass_diffence['[2M-H]-'] = r.mz + 1.007276
        adduct_mass_diffence['[3M-H]-'] = 2*(r.mz + 1.007276)


    # find adducts by assuming the current roi is the [M+H]+ ion in positive mode and [M-H]- ion in negative mode
    for idx, r in enumerate(d.rois):
        
        if not roi_to_label[idx]:
            continue

        if r.charge_state == 2:
            if d.params.ion_mode.lower() == "positive":
                r.adduct_type = "[M+2H]2+"
                roi_to_label[idx] = False
            elif d.params.ion_mode.lower() == "negative":
                r.adduct_type = "[M-2H]2-"
                roi_to_label[idx] = False
            continue
        
        for adduct in adduct_mass_diffence.keys():
            m = r.mz + adduct_mass_diffence[adduct]
            v = np.logical_and(np.abs(d.roi_mz_seq - m) < mz_tol, np.abs(d.roi_rt_seq - r.rt) < rt_tol)
            v = np.where(np.logical_and(v, roi_to_label))[0]

            if len(v) == 0:
                continue

            if len(v) > 1:
                # select the one with the lowest RT difference
                v = v[np.argmin(np.abs(d.roi_rt_seq[v] - r.rt))]
            else:
                v = v[0]

            if scan_to_scan_correlation(r, d.rois[v]) > d.params.ppr:
                roi_to_label[v] = False
                d.rois[v].adduct_type = adduct
                d.rois[v].adduct_parent_roi_id = r.id
                r.adduct_child_roi_id.append(d.rois[v].id)

        if len(r.adduct_child_roi_id) > 0:
            r.adduct_type = default_adduct
        
    for r in d.rois:
        if r.adduct_type is None:
            r.adduct_type = default_adduct


"""
Helper functions and constants
==============================
"""

def scan_to_scan_correlation(feature_a, feature_b):
    """
    Calculate the scan-to-scan correlation between two features using Pearson correlation.

    Parameters
    ----------
    feature_a: Feature object
        The first feature object.
    feature_b: Feature object
        The second feature object.
    
    Returns
    -------
    float
        The scan-to-scan correlation between the two features.
    """

    # find the common scans in the two rois
    common_idx_a = np.nonzero(np.in1d(feature_a.scan_idx_seq, feature_b.scan_idx_seq))[0]
    common_idx_b = np.nonzero(np.in1d(feature_b.scan_idx_seq, feature_a.scan_idx_seq))[0]

    # if the number of common scans is less than 5, return 1
    if len(common_idx_a) < 5:
        return 1.0

    # find the intensities of the common scans in the two rois
    int1 = feature_a.signals[common_idx_a,1]
    int2 = feature_b.signals[common_idx_b,1]

    # if all values are same in either feature, return 1
    # this is to avoid the case where the common scans are all zeros
    if np.all(int1 == int1[0]) or np.all(int2 == int2[0]):
        return 1.0
    
    return np.corrcoef(int1, int2)[0, 1]


def get_charge_state(mz_seq, valid_charge_states=[1,2]):
    """
    A function to determine the charge state using the m/z sequence of isotopes.

    Parameters
    ----------
    mz_seq: list
        A list of m/z values of isotopes.
    valid_charge_states: list
        A list of valid charge states.

    Returns
    -------
    int
        The charge state of the isotopes.
    """

    # if there is only one isotope, return 1
    if len(mz_seq) < 2:
        return 1
    else:
        for charge in valid_charge_states:
            if abs(mz_seq[1] - mz_seq[0] - 1.003355/charge) < 0.01:
                return charge
    return 1


# could include more adducts
_ADDUCT_MASS_DIFFERENCE_POS_AGAINST_H = {
    '[M+H-H2O]+': -18.010565,
    '[M+Na]+': 21.981945,
    '[M+K]+': 37.955882,
    '[M+NH4]+': 17.02655,
}

_ADDUCT_MASS_DIFFERENCE_NEG_AGAINST_H = {
    '[M-H-H2O]-': -18.010564,
    '[M+Cl]-': 35.976677,
    '[M+CH3COO]-': 60.021129,
    '[M+HCOO]-': 46.005479,
}


ADDUCT_POS = {
    '[M+H]+': (1.007276, 1),
    '[M+NH4]+': (18.033826, 1),
    '[M+H+CH3OH]+': (33.03349, 1),
    '[M+Na]+': (22.989221, 1),
    '[M+K]+': (38.963158, 1),
    '[M+Li]+': (6.014574, 1),
    '[M+Ag]+': (106.904548, 1),
    '[M+H+CH3CN]+': (42.033826, 1),
    '[M-H+2Na]+': (44.971165, 1),
    '[M+H-H2O]+': (-17.003288, 1),
    '[M+H-2H2O]+': (-35.01385291583, 1),
    '[M+H-3H2O]+': (-53.02441759986, 1),
    '[M+H+HAc]+': (61.02841, 1),
    '[M+H+HFA]+': (47.01276, 1),
    '[M+Ca]2+': (39.961493, 1),
    '[M+Fe]2+': (55.93384, 1),
    '[2M+H]+': (1.007276, 2),
    '[2M+NH4]+': (18.033826, 2),
    '[2M+Na]+': (22.989221, 2),
    '[2M+H-H2O]+': (-17.003288, 2),
    '[3M+H]+': (1.007276, 3),
    '[3M+NH4]+': (18.033826, 3),
    '[3M+Na]+': (22.989221, 3),
    '[3M+H-H2O]+': (-17.003288, 3),
    '[M+2H]2+': (0.503638, 1),
    '[M+3H]3+': (0.335759, 1),
}

ADDUCT_NEG = {
    '[M-H]-': (-1.007276, 1),
    '[M+Cl]-': (34.969401, 1),
    '[M+Br]-': (78.918886, 1),
    '[M+FA]-': (44.998203, 1),
    '[M+Ac]-': (59.013853, 1),
    '[M-H-H2O]-': (-19.017841, 1),
    '[M-H+Cl]2-': (33.962124, 1),
    '[2M-H]-': (-1.007276, 2),
    '[2M+Cl]-': (34.969401, 2),
    '[2M+Br]-': (78.918886, 2),
    '[2M+FA]-': (44.998203, 2),
    '[2M+Ac]-': (59.013853, 2),
    '[2M-H-H2O]-': (-19.017841, 2),
    '[3M-H]-': (-1.007276, 3),
    '[3M+Cl]-': (34.969401, 3),
    '[3M+Br]-': (78.918886, 3),
    '[3M+FA]-': (44.998203, 3),
    '[3M+Ac]-': (59.013853, 3),
    '[3M-H-H2O]-': (-19.017841, 3),
    '[M-2H]2-': (-0.503638, 1),
    '[M-3H]3-': (-0.335759, 1),
}


_adduct_pos_primary = {
    '[M+H]+': [-21.981945, -17.02655, 18.010564],
    '[M+Na]+': [21.981945, 4.955395, 39.992509],
    '[M+NH4]+': [17.02655, -4.955395, 35.037114],
    '[M+H-H2O]+': [-18.010564, -39.992509, -35.037114]
}

_adduct_neg_primary = {
    '[M-H]-': [18.010565, -46.005479, -60.021129],
    '[M-H-H2O]-': [-18.010565, -64.016044, -78.031694],
    '[M+FA]-': [46.005479, 64.016044, -14.01565],
    '[M+Ac]-': [60.021129, 78.031694, 14.01565]
}