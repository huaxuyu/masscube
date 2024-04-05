# Author: Hauxu Yu

# A module to group metabolic features from unique compounds
# 1. annotate isotopes
# 2. annotate adducts
# 3. annotate in-source fragments

# Import modules
import numpy as np


def annotate_isotope(d):
    """
    Function to annotate isotopes in a MS data.
    
    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object.
    """

    # rank the rois (d.rois) in each file by m/z
    d.rois.sort(key=lambda x: x.mz)

    for r in d.rois:
        
        if r.is_isotope:
            continue

        r.isotope_int_seq = [r.peak_height]
        r.isotope_mz_seq = [r.mz]

        # go to that scan and determine the isotopes
        isotopes = r.mz + _ISOTOPE_MASS_ARRAY

        last_mz = r.mz
        isotope_id_seq = []

        # find roi using isotope list
        for iso in isotopes:
            
            # if isotpoe is not found in two daltons, stop searching
            if iso - last_mz > 2.2:
                break

            v = np.where(np.logical_and(np.abs(d.roi_mz_seq - iso) < 0.01, np.abs(d.roi_rt_seq - r.rt) < 0.05))[0]

            if len(v) == 0:
                continue

            # an isotope can't have intensity 3 fold or higher than M0 or 1% lower than the M0
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height < 1.2*r.peak_height]
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height > 0.01*r.peak_height]

            if len(v) == 0:
                continue
            
            # for high-resolution data, C and N isotopes can be separated and need to be summed
            total_int = np.sum([d.rois[vi].peak_height for vi in v])

            r.isotope_mz_seq.append(iso)
            r.isotope_int_seq.append(total_int)
            
            for vi in v:
                d.rois[vi].is_isotope = True
                isotope_id_seq.append(d.rois[vi].id)

            last_mz = iso

        r.charge_state = get_charge_state(r.isotope_mz_seq)
        r.isotope_id_seq = isotope_id_seq


def annotate_in_source_fragment(d):
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

    # isotopes can't be parent or child
    for idx, r in enumerate(d.rois):
        if r.is_isotope:
            roi_to_label[idx] = False
    
    # find in-source fragments
    for idx in range(len(d.rois)-1, -1, -1):

        r = d.rois[idx]

        # isotpes and in-source fragments cannot be the parent
        if not roi_to_label[idx] or r.best_ms2 is None:
            continue

        for m in r.best_ms2.peaks[:, 0]:

            v = np.logical_and(np.abs(d.roi_mz_seq - m) < 0.01, np.abs(d.roi_rt_seq - r.rt) < 0.05)
            v = np.where(np.logical_and(v, roi_to_label))[0]

            if len(v) == 0:
                continue

            if len(v) > 1:
                # select the one with the lowest RT difference
                v = v[np.argmin(np.abs(d.roi_rt_seq[v] - r.rt))]
            else:
                v = v[0]

            if peak_peak_correlation(r, d.rois[v]) > d.params.ppr:
                roi_to_label[v] = False
                d.rois[v].is_in_source_fragment = True
                d.rois[v].isf_parent_roi_id = r.id
                r.isf_child_roi_id.append(d.rois[v].id)


def annotate_adduct(d):
    """
    A function to annotate adducts from the same compound.

    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object that contains the detected rois to be grouped.
    """

    d.rois.sort(key=lambda x: x.mz)
    roi_to_label = np.ones(len(d.rois), dtype=bool)

    # isotopes and in-source fragments cannot be the parent
    for idx, r in enumerate(d.rois):
        if r.is_isotope or r.is_in_source_fragment:
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
            v = np.logical_and(np.abs(d.roi_mz_seq - m) < 0.01, np.abs(d.roi_rt_seq - r.rt) < 0.05)
            v = np.where(np.logical_and(v, roi_to_label))[0]

            if len(v) == 0:
                continue

            if len(v) > 1:
                # select the one with the lowest RT difference
                v = v[np.argmin(np.abs(d.roi_rt_seq[v] - r.rt))]
            else:
                v = v[0]

            if peak_peak_correlation(r, d.rois[v]) > d.params.ppr:
                roi_to_label[v] = False
                d.rois[v].adduct_type = adduct
                d.rois[v].adduct_parent_roi_id = r.id
                r.adduct_child_roi_id.append(d.rois[v].id)

        if len(r.adduct_child_roi_id) > 0:
            r.adduct_type = default_adduct
        
    for r in d.rois:
        if r.adduct_type is None:
            r.adduct_type = default_adduct


def peak_peak_correlation(roi1, roi2):
    """
    A function to find the peak-peak correlation between two rois.

    Parameters
    ----------------------------------------------------------
    roi1: ROI object
        An ROI object.
    roi2: ROI object
        An ROI object.
    
    Returns
    ----------------------------------------------------------
    pp_cor: float
        The peak-peak correlation between the two rois.
    """

    # find the common scans in the two rois
    common_scans = np.intersect1d(roi1.scan_idx_seq, roi2.scan_idx_seq)

    if len(common_scans) < 5:
        return 1.0

    # find the intensities of the common scans in the two rois
    int1 = roi1.int_seq[np.isin(roi1.scan_idx_seq, common_scans)]
    int2 = roi2.int_seq[np.isin(roi2.scan_idx_seq, common_scans)]

    # calculate the correlation
    # if all values are same, return 1
    if np.all(int1 == int1[0]) or np.all(int2 == int2[0]):
        return 1.0
    
    pp_cor = np.corrcoef(int1, int2)[0, 1]
    return pp_cor


def get_charge_state(mz_seq):
    
    if len(mz_seq) < 2:
        return 1
    else:
        mass_diff = mz_seq[1] - mz_seq[0]

        # check mass diff is closer to 1 or 0.5 | note, mass_diff can be larger than 1
        if abs(mass_diff - 1) < abs(mass_diff - 0.5):
            return 1
        else:
            return 2


_ISOTOPIC_MASS_DIFFERENCE = {
    'H': 1.006277,
    'C': 1.003355,
    'N': 0.997035,
    'O': 2.004246,
    'S': 1.995796,
    'Cl': 1.99705
}


# adduct mass difference is calculated against the [M+H]+ ion in positive mode, and [M-H]- ion in negative mode
_ADDUCT_MASS_DIFFERENCE_NEG = {
    '-H': -1.007276,
    '-H-H2O': -19.01784,
    '+Cl': 34.969401,
    '+CH3COO': 59.013853,
    '+HCOO': 44.998203,
}

_ADDUCT_MASS_DIFFERENCE_POS = {
    '+H': 1.007276,
    '+H-H2O': -17.003289,
    '+Na': 22.989221,
    '+K': 38.963158,
    '+NH4': 18.033826,
}


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


_ISOTOPE_MASS_ARRAY = np.arange(0, 10.1, 1.003355/2)[1:]