# Author: Hauxu Yu

# A module to group metabolic features from unique compounds
# 1. annotate isotopes
# 2. annotate adducts
# 3. annotate in-source fragments

# Import modules
import numpy as np


def annotate_isotope(d, mz_tol=0.015, rt_tol=0.1, valid_intensity_ratio_range=[0.001, 1.2], charge_state_range=[1,2]):
    """
    Function to annotate isotopes in a MS data.
    
    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object.
    mz_tol: float
        The m/z tolerance to find isotopes.
    rt_tol: float
        The RT tolerance to find isotopes.
    valid_intensity_ratio_range: list
        The valid intensity ratio range between isotopes.
    charge_state_range: list
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

            if peak_peak_correlation(r, d.rois[v]) > d.params.ppr:
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

"""
_adduct_pos = [
    {'name': '[M+H]+', 'm': 1, 'charge': 1, 'mass': 1.00727645223},
    {'name': '[M+Na]+', 'm': 1, 'charge': 1, 'mass': 22.989220702},
    {'name': '[M+K]+', 'm': 1, 'charge': 1, 'mass': 38.9631579064},
    {'name': '[M+NH4]+', 'm': 1, 'charge': 1, 'mass': 18.03382555335},
    {'name': '[M-H+2Na]+', 'm': 1, 'charge': 1, 'mass': 44.97116495177},
    {'name': '[M+H-H2O]+', 'm': 1, 'charge': 1, 'mass': -17.0032882318},
    {'name': '[M+H-2H2O]+', 'm': 1, 'charge': 1, 'mass': -35.01385291583},
    {'name': '[M+H-3H2O]+', 'm': 1, 'charge': 1, 'mass': -53.02441759986},

    {'name': '[2M+H]+', 'm': 2, 'charge': 1, 'mass': 1.00727645223},
    {'name': '[2M+Na]+', 'm': 2, 'charge': 1, 'mass': 22.989220702},
    {'name': '[2M+K]+', 'm': 2, 'charge': 1, 'mass': 38.9631579064},
    {'name': '[2M+NH4]+', 'm': 2, 'charge': 1, 'mass': 18.03382555335},
    {'name': '[2M-H+2Na]+', 'm': 2, 'charge': 1, 'mass': 44.97116495177},
    {'name': '[2M+H-H2O]+', 'm': 2, 'charge': 1, 'mass': -17.0032882318},
    {'name': '[2M+H-2H2O]+', 'm': 2, 'charge': 1, 'mass': -35.01385291583},
    {'name': '[2M+H-3H2O]+', 'm': 2, 'charge': 1, 'mass': -53.02441759986},

    {'name': '[M+2H]2+', 'm': 1, 'charge': 2, 'mass': 2.01455290446},
    {'name': '[M+H+Na]2+', 'm': 1, 'charge': 2, 'mass': 23.99649715423},
    {'name': '[M+H+NH4]2+', 'm': 1, 'charge': 2, 'mass': 19.04110200558},
    {'name': '[M+Ca]2+', 'm': 1, 'charge': 2, 'mass': 39.961493703},
    {'name': '[M+Fe]2+', 'm': 1, 'charge': 2, 'mass': 55.93383917}
]

_adduct_neg = {
    {'name': '[M-H]-', 'm': 1, 'charge': 1, 'mass': -1.00727645223},
    {'name': '[M+Cl]-', 'm': 1, 'charge': 1, 'mass': 34.968304102},
    {'name': '[M+Br]-', 'm': 1, 'charge': 1, 'mass': 78.91778902},
    {'name': '[M+FA]-', 'm': 1, 'charge': 1, 'mass': 44.99710569137},
    {'name': '[M+Ac]-', 'm': 1, 'charge': 1, 'mass': 59.01275575583},
    {'name': '[M-H-H2O]-', 'm': 1, 'charge': 1, 'mass': -19.01784113626},

    {'name': '[2M-H]-', 'm': 2, 'charge': 1, 'mass': -1.00727645223},
    {'name': '[2M+Cl]-', 'm': 2, 'charge': 1, 'mass': 34.968304102},
    {'name': '[2M+Br]-', 'm': 2, 'charge': 1, 'mass': 78.91778902},
    {'name': '[2M+FA]-', 'm': 2, 'charge': 1, 'mass': 44.99710569137},
    {'name': '[2M+Ac]-', 'm': 2, 'charge': 1, 'mass': 59.01275575583},
    {'name': '[2M-H-H2O]-', 'm': 2, 'charge': 1, 'mass': -19.01784113626},

    {'name': '[M-2H]2-', 'm': 1, 'charge': 2, 'mass': -2.01455290446},
    {'name': '[M-H+Cl]2-', 'm': 1, 'charge': 2, 'mass': 33.96157622977},
    {'name': '[M-H+Br]2-', 'm': 1, 'charge': 2, 'mass': 77.91106114777},
}
"""