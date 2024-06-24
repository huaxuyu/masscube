# Utility functions for feature table manipulation
import pandas as pd

def calculate_fill_percentage(feature_table, individual_sample_groups):
    """
    calculate fill percentage for each feature

    Parameters
    ----------
    feature_table : pd.DataFrame
        feature table
    individual_sample_groups : list
        list of individual sample groups

    Returns
    -------
    feature_table : pd.DataFrame
        feature table with fill percentage
    """

    # blank samples are not included in fill percentage calculation
    blank_number = len([x for x in individual_sample_groups if 'blank' in x])
    total_number = len(individual_sample_groups)
    if blank_number == 0:
        feature_table['fill_percentage'] = (feature_table.iloc[:, -total_number:] > 0).sum(axis=1)/total_number * 100
    else:
        feature_table['fill_percentage'] = (feature_table.iloc[:, -total_number:-blank_number] > 0).sum(axis=1)/(total_number - blank_number) * 100
    return feature_table

def convert_features_to_df(features, sample_names):
    """
    convert feature list to DataFrame

    Parameters
    ----------
    features : list
        list of features

    Returns
    -------
    feature_table : pd.DataFrame
        feature DataFrame
    """

    results = []
    sample_names = list(sample_names)
    columns=["ID", "m/z", "RT", "adduct", "is_isotope", "is_in_source_fragment", "Gaussian_similarity", "noise_level", 
             "asymmetry_factor", "charge", "isotopes", "MS2", "matched_MS2", "search_mode", "annotation", "formula", "similarity", "matched_peak_number", "SMILES", "InChIKey", 
                "fill_percentage", "alignment_reference"] + sample_names
    for feature in features:
        results.append([feature.id, feature.mz, feature.rt, feature.adduct_type, feature.is_isotope, feature.is_in_source_fragment, feature.gaussian_similarity, feature.noise_level, 
                        feature.asymmetry_factor, feature.charge_state, feature.isotopes, feature.best_ms2, feature.matched_ms2, feature.search_mode, feature.annotation, feature.formula, feature.similarity, feature.matched_peak_number, feature.smiles, feature.inchikey, 
                        feature.fill_percentage, feature.reference_file] + list(feature.peak_height_seq))
    feature_table = pd.DataFrame(results, columns=columns)
    return feature_table