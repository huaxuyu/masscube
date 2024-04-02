# Author: Hauxu Yu

# A module to annotate metabolites based on their MS/MS spectra

# Import modules
import os
from ms_entropy import read_one_spectrum, FlashEntropySearch
import pickle
import numpy as np
import json
import pandas as pd
import re

def load_msms_db(path):
    """
    A function to load the MS/MS database in MSP format or pickle format.

    Parameters
    ----------
    path : str
        The path to the MS/MS database in MSP format.    
    """

    print("Loading MS/MS database...")
    # get extension of path
    ext = os.path.splitext(path)[1]

    if ext.lower() == '.msp':
        db =[]
        for a in read_one_spectrum(path):
            db.append(a)
        _correct_db(db)
        entropy_search = FlashEntropySearch()
        entropy_search.build_index(db)
        print("MS/MS database loaded.")
        return entropy_search
    
    elif ext.lower() == '.pkl':
        entropy_search = pickle.load(open(path, 'rb'))
        print("MS/MS database loaded.")
        return entropy_search
    
    elif ext.lower() == '.json':
        db = json.load(open(path, 'r'))
        entropy_search = FlashEntropySearch()
        entropy_search.build_index(db)
        print("MS/MS database loaded.")
        return entropy_search


def feature_annotation(feature_table: pd.DataFrame, parameters):
    """
    A function to annotate features based on their MS/MS spectra and a MS/MS database. 
    Input: feature_table, parameters
    
    """
    
    entropy_search = load_msms_db(parameters.msms_library)

    # set the value type in annotation, search_mode, InChIKey, SMILES, matched_MS2 to string
    feature_table['annotation'] = None
    feature_table['search_mode'] = None
    feature_table['InChIKey'] = None
    feature_table['matched_MS2'] = None
    feature_table['SMILES'] = None
    feature_table['formula'] = None

    for i in range(len(feature_table)):
        ms2 = feature_table.loc[i, 'MS2']
        # check if ms2 is nan
        if ms2 != ms2:
            continue
        
        precursor_mz = feature_table.loc[i, 'm/z']
        peaks = _extract_peaks_from_string(ms2)

        peaks = entropy_search.clean_spectrum_for_search(precursor_mz, peaks, precursor_ions_removal_da=-1)
        entropy_similarity, matched_peaks_number = entropy_search.identity_search(precursor_mz=precursor_mz, peaks=peaks, ms1_tolerance_in_da=parameters.mz_tol_ms1, 
                                                                                  ms2_tolerance_in_da=parameters.mz_tol_ms2, output_matched_peak_number=True)
        idx = np.argmax(entropy_similarity)
        if entropy_similarity[idx] > parameters.ms2_sim_tol:
            matched = entropy_search[np.argmax(entropy_similarity)]
            matched = {k.lower():v for k,v in matched.items()}
            feature_table.loc[i, 'annotation'] = matched['name']
            feature_table.loc[i, 'search_mode'] = 'identity_search'
            feature_table.loc[i, 'similarity'] = entropy_similarity[idx]
            feature_table.loc[i, 'matched_peak_number'] = matched_peaks_number[idx]
            feature_table.loc[i, 'SMILES'] = matched['smiles'] if 'smiles' in matched else None
            feature_table.loc[i, 'InChIKey'] = matched['inchikey'] if 'inchikey' in matched else None
            feature_table.loc[i, 'matched_MS2'] = _convert_peaks_to_string(matched['peaks'])
            feature_table.loc[i, 'formula'] = matched['formula'] if 'formula' in matched else None
            feature_table.loc[i, 'adduct'] = matched['precursor_type']
        else:
            entropy_similarity = entropy_search.hybrid_search(precursor_mz=precursor_mz, peaks=peaks, ms1_tolerance_in_da=parameters.mz_tol_ms1, 
                                                              ms2_tolerance_in_da=parameters.mz_tol_ms2)
            idx = np.argmax(entropy_similarity)
            if entropy_similarity[idx] > parameters.ms2_sim_tol:
                matched = entropy_search[np.argmax(entropy_similarity)]
                matched = {k.lower():v for k,v in matched.items()}
                feature_table.loc[i, 'annotation'] = matched['name']
                feature_table.loc[i, 'search_mode'] = 'hybrid_search'
                feature_table.loc[i, 'similarity'] = entropy_similarity[idx]
                feature_table.loc[i, 'SMILES'] = matched['smiles'] if 'smiles' in matched else None
                feature_table.loc[i, 'InChIKey'] = matched['inchikey'] if 'inchikey' in matched else None
                feature_table.loc[i, 'matched_MS2'] = _convert_peaks_to_string(matched['peaks'])
                feature_table.loc[i, 'formula'] = matched['formula'] if 'formula' in matched else None
    
    return feature_table


def annotate_feature_table(feature_list, params):
    """
    A function to annotate features based on their MS/MS spectra and a MS/MS database.

    Parameters
    ----------
    feature_list : list
        A list of features.
    params : Params object
        The parameters for the workflow.
    """

    # load the MS/MS database
    entropy_search = load_msms_db(params.msms_library)

    for f in feature_list:
        
        if f.best_ms2 is not None:
            peaks = entropy_search.clean_spectrum_for_search(f.mz, f.best_ms2.peaks, precursor_ions_removal_da=-1)
            entropy_similarity, matched_peaks_number = entropy_search.identity_search(precursor_mz=f.mz, peaks=peaks, ms1_tolerance_in_da=params.mz_tol_ms1, 
                                                                                      ms2_tolerance_in_da=params.mz_tol_ms2, output_matched_peak_number=True)
            
            idx = np.argmax(entropy_similarity)
            if entropy_similarity[idx] > params.ms2_sim_tol:
                matched = entropy_search[np.argmax(entropy_similarity)]
                matched = {k.lower():v for k,v in matched.items()}
                f.annotation = matched['name']
                f.similarity = entropy_similarity[idx]
                f.matched_peak_number = matched_peaks_number[idx]
                f.smiles = matched['smiles'] if 'smiles' in matched else None
                f.inchikey = matched['inchikey'] if 'inchikey' in matched else None
                f.matched_precursor_mz = matched['precursor_mz']
                f.matched_peaks = matched['peaks']
                f.formula = matched['formula'] if 'formula' in matched else None
                f.annotation_mode = 'identity_search'
            else:
                entropy_similarity = entropy_search.hybrid_search(precursor_mz=f.mz, peaks=peaks, ms1_tolerance_in_da=params.mz_tol_ms1, 
                                                                  ms2_tolerance_in_da=params.mz_tol_ms2)
                idx = np.argmax(entropy_similarity)
                if entropy_similarity[idx] > params.ms2_sim_tol:
                    matched = entropy_search[np.argmax(entropy_similarity)]
                    matched = {k.lower():v for k,v in matched.items()}
                    f.annotation = matched['name']
                    f.similarity = entropy_similarity[idx]
                    f.smiles = matched['smiles'] if 'smiles' in matched else None
                    f.inchikey = matched['inchikey'] if 'inchikey' in matched else None
                    f.matched_precursor_mz = matched['precursor_mz']
                    f.matched_peaks = matched['peaks']
                    f.formula = matched['formula'] if 'formula' in matched else None
                    f.annotation_mode = 'hybrid_search'


def annotate_features_all_mode_search(feature_list, params, mode='hybrid'):
    """
    A function to annotate features based on their MS/MS spectra and a MS/MS database.
    Four modes are supported: identity search, open search, neutral loss search, and hybrid search.
    See https://www.nature.com/articles/s41592-023-02012-9 Figure 1 for more details.

    Parameters
    ----------
    feature_list : list
        A list of features.
    params : Params object
        The parameters for the workflow.
    mode : str
        The mode for MS/MS search.
        'identity': identity search
        'hybrid': hybrid search
        'open': open search
        'neutral_loss': neutral loss search
    """

    # load the MS/MS database
    entropy_search = load_msms_db(params.msms_library)

    for f in feature_list:

        if f.annotation is not None:
            continue
        
        if f.best_ms2 is not None:
            peaks = entropy_search.clean_spectrum_for_search(f.mz, f.best_ms2.peaks, precursor_ions_removal_da=-1)
            if mode == 'hybrid':
                entropy_similarity = entropy_search.hybrid_search(precursor_mz=f.mz, peaks=peaks, ms1_tolerance_in_da=params.mz_tol_ms1,
                                                                                        ms2_tolerance_in_da=params.mz_tol_ms2, output_matched_peak_number=True)
            elif mode == 'open':
                entropy_similarity = entropy_search.open_search(precursor_mz=f.mz, peaks=peaks, ms1_tolerance_in_da=params.mz_tol_ms1,
                                                                                        ms2_tolerance_in_da=params.mz_tol_ms2, output_matched_peak_number=True)
            elif mode == 'neutral_loss':
                entropy_similarity = entropy_search.neutral_loss_search(precursor_mz=f.mz, peaks=peaks, ms1_tolerance_in_da=params.mz_tol_ms1,
                                                                                            ms2_tolerance_in_da=params.mz_tol_ms2, output_matched_peak_number=True)
            
            idx = np.argmax(entropy_similarity)
            if entropy_similarity[idx] > params.ms2_sim_tol:
                matched = entropy_search[np.argmax(entropy_similarity)]
                matched = {k.lower():v for k,v in matched.items()}
                f.annotation = matched['name']
                f.similarity = entropy_similarity[idx]
                f.smiles = matched['smiles'] if 'smiles' in matched else None
                f.inchikey = matched['inchikey'] if 'inchikey' in matched else None
                f.matched_precursor_mz = matched['precursor_mz']
                f.matched_peaks = matched['peaks']
                f.formula = matched['formula'] if 'formula' in matched else None
                f.annotation_mode = mode + '_search'


def annotate_rois(d):
    """
    A function to annotate rois based on their MS/MS spectra and a MS/MS database.

    Parameters
    ----------
    d : MSData object
        MS data.
    """

    # load the MS/MS database
    entropy_search = load_msms_db(d.params.msms_library)
    

    for f in d.rois:
        f.annotation = None
        f.similarity = None
        f.matched_peak_number = None
        f.smiles = None
        f.inchikey = None
        f.matched_precursor_mz = None
        f.matched_peaks = None
        f.formula = None

        if f.best_ms2 is not None:
            peaks = entropy_search.clean_spectrum_for_search(f.mz, f.best_ms2.peaks)
            entropy_similarity, matched_peaks_number = entropy_search.identity_search(precursor_mz=f.mz, peaks=peaks, ms1_tolerance_in_da=d.params.mz_tol_ms1, 
                                                                                      ms2_tolerance_in_da=d.params.mz_tol_ms2, output_matched_peak_number=True)
            
            idx = np.argmax(entropy_similarity)
            if entropy_similarity[idx] > d.params.ms2_sim_tol:
                matched = entropy_search[np.argmax(entropy_similarity)]
                matched = {k.lower():v for k,v in matched.items()}
                f.annotation = matched['name']
                f.similarity = entropy_similarity[idx]
                f.matched_peak_number = matched_peaks_number[idx]
                f.smiles = matched['smiles'] if 'smiles' in matched else None
                f.inchikey = matched['inchikey'] if 'inchikey' in matched else None
                f.matched_precursor_mz = matched['precursor_mz']
                f.matched_peaks = matched['peaks']
                f.formula = matched['formula'] if 'formula' in matched else None


def has_chlorine(iso):
    # to be constructed
    pass


def has_bromine(iso):
    # to be constructed
    pass


def feature_to_feature_search(feature_list, sim_tol=0.8):
    """
    A function to calculate the MS2 similarity between features using hybrid search.

    Parameters
    ----------
    feature_list : list
        A list of AlignedFeature objects.
    sim_tol : float
        The similarity threshold for feature-to-feature search.
    
    Returns
    ----------
    similarity_matrix : pandas.DataFrame
        similarity matrix between features.
    """

    results = []

    entropy_search = index_feature_list(feature_list)

    for f in feature_list:

        similarities = entropy_search.search(precursor_mz=f.mz, peaks=f.best_ms2.peaks)["hybrid_search"]
        for i, s in enumerate(similarities):
            if s > sim_tol and f.id != entropy_search[i]["id"]:
                results.append([f.network_name, entropy_search[i]["name"], s, f.id, entropy_search[i]["id"]])

    df = pd.DataFrame(results, columns=['feature_name_1', 'feature_name_2', 'similarity','feature_id_1', 'feature_id_2'])
    return df


def index_feature_list(feature_list, return_db=False):
    """
    A function to index a list of features for spectrum entropy search.

    Parameters
    ----------
    feature_list : list
        A list of AlignedFeature objects.
    """
    
    db = []
    for f in feature_list:
        if f.best_ms2 is not None:
            tmp = {
                "id": f.id,
                "name": f.network_name,
                "mode": f.annotation_mode,
                "precursor_mz": f.mz,
                "peaks": f.best_ms2.peaks
            }
            db.append(tmp)

    entropy_search = FlashEntropySearch()
    entropy_search.build_index(db)

    if return_db:
        return entropy_search, db
    else:
        return entropy_search


def _correct_db(db):
    """
    Correct the MS/MS database by changing the key names.
    """

    # make sure precursor_mz is in the db
    if 'precursor_mz' not in db[0].keys():
        for a in db:
            similar_key = [k for k in a.keys() if 'prec' in k and 'mz' in k]
            a['precursor_mz'] = float(a.pop(similar_key[0]))


def _extract_peaks_from_string(ms2):
    """
    Extract peaks from MS2 spectrum.

    Parameters
    ----------
    ms2 : str
        MS2 spectrum in string format.
    
    Example
    ----------
    
    """
    
    # Use findall function to extract all numbers matching the pattern
    numbers = re.findall(r'\d+\.\d+', ms2)
    
    # Convert the extracted numbers from strings to floats
    numbers = [float(num) for num in numbers]
    
    numbers = np.array(numbers).reshape(-1, 2)

    return numbers


def _convert_peaks_to_string(peaks):
    """
    Convert peaks to string format.

    Parameters
    ----------
    peaks : numpy.array
        Peaks in numpy array format.
    
    Example
    ----------
    
    """
    
    ms2 = ""
    for i in range(len(peaks)):
        ms2 += str(np.round(peaks[i, 0], decimals=4)) + ";" + str(np.round(peaks[i, 1], decimals=4)) + "|"
    ms2 = ms2[:-1]
    
    return ms2