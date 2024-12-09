# Author: Hauxu Yu

# A module to annotate metabolites based on their MS2 spectra

# imports
import os
import pickle
import numpy as np
import pandas as pd
import json
import re
from tqdm import tqdm
from ms_entropy import read_one_spectrum, FlashEntropySearch

from .utils_functions import extract_signals_from_string, convert_signals_to_string


def load_ms2_db(path):
    """
    A function to load the MS2 database in pickle, msp, or json format.

    Parameters
    ----------
    path : str
        The path to the MS2 database.
    """

    print("\tLoading MS2 database...")
    # get extension of path
    ext = os.path.splitext(path)[1]

    if ext.lower() == '.msp':
        db =[]
        for a in read_one_spectrum(path):
            db.append(a)
        _correct_db(db)
        entropy_search = FlashEntropySearch()
        entropy_search.build_index(db)
        print("\tMS2 database has been loaded.")
        return entropy_search
    
    elif ext.lower() == '.pkl':
        entropy_search = pickle.load(open(path, 'rb'))
        print("\tMS2 database has been loaded.")
        return entropy_search
    
    elif ext.lower() == '.json':
        db = json.load(open(path, 'r'))
        entropy_search = FlashEntropySearch()
        entropy_search.build_index(db)
        print("\tMS2 database has been loaded.")
        return entropy_search
    else:
        print("The MS2 database format {} is not supported.".format(ext))
        print("Please provide a MS2 database in pkl (best), msp, or json format.")


def annotate_aligned_features(features, params, num=5):
    """
    Annotate feature's MS2 using database.
    
    Parameters
    ----------
    features : list
        A list of AlignedFeature objects.
    params : Params object
        The parameters for the workflow.
    num : int
        The number of top MS2 spectra to search.
    """

    entropy_search = load_ms2_db(params.ms2_library_path)

    for f in tqdm(features):
        if len(f.ms2_seq) == 0:
            continue
        parsed_ms2 = []
        for file_name, ms2 in f.ms2_seq:
            signals = extract_signals_from_string(ms2)
            signals = entropy_search.clean_spectrum_for_search(f.mz, signals, precursor_ions_removal_da=params.precursor_mz_offset)
            parsed_ms2.append([file_name, signals])
        # sort parsed ms2 by summed intensity
        parsed_ms2.sort(key=lambda x: np.sum(x[1][:, 1]), reverse=True)
        parsed_ms2 = parsed_ms2[:num]
        matched = None

        f.similarity = 0
        f.ms2_reference_file = parsed_ms2[0][0]
        f.ms2 = parsed_ms2[0][1]
        f.matched_peak_number = 0

        for file_name, signals in parsed_ms2:
            similarity, matched_num = entropy_search.identity_search(precursor_mz=f.mz, peaks=signals, ms1_tolerance_in_da=params.mz_tol_ms1,
                                                                     ms2_tolerance_in_da=params.mz_tol_ms2, output_matched_peak_number=True)
            idx = np.argmax(similarity)
            if similarity[idx] > params.ms2_sim_tol and similarity[idx] > f.similarity:
                matched = entropy_search[idx]
                f.similarity = similarity[idx]
                f.ms2_reference_file = file_name
                f.ms2 = signals
                f.matched_peak_number = matched_num[idx]

        if matched is not None:
            matched = {k.lower():v for k,v in matched.items()}
            _assign_annotation_results_to_feature(f, score=f.similarity, matched=matched, matched_peak_num=f.matched_peak_number, 
                                                  search_mode='identity_search')

        else:
            similarity = entropy_search.hybrid_search(precursor_mz=f.mz, peaks=f.ms2, ms1_tolerance_in_da=params.mz_tol_ms1, 
                                                              ms2_tolerance_in_da=params.mz_tol_ms2)
            idx = np.argmax(similarity)
            if similarity[idx] > params.ms2_sim_tol:
                matched = entropy_search[idx]
                matched = {k.lower():v for k,v in matched.items()}
                _assign_annotation_results_to_feature(f, score=similarity[idx], matched=matched, 
                                                      matched_peak_num=None, search_mode='fuzzy_search')
        
        f.ms2 = convert_signals_to_string(f.ms2)

    return features


def annotate_features(d, sim_tol=None, fuzzy_search=True, ms2_library_path=None):
    """
    Annotate features from a single raw data file using MS2 database.
    
    Parameters
    ----------
    d : MSData object
        MS data file.
    sim_tol : float
        The similarity threshold for MS2 annotation. If not specified, the corresponding parameter from
        the MS data file will be used.
    fuzzy_search : bool
        Whether to further annotated the unmatched MS2 using fuzzy search.
    ms2_library_path : str
        The absolute path to the MS2 database. If not specified, the corresponding parameter from 
        the MS data file will be used.
    """

    if ms2_library_path is None:
        search_engine = load_ms2_db(d.params.ms2_library_path)
    else:
        search_engine = load_ms2_db(ms2_library_path)

    if sim_tol is None:
        sim_tol = d.params.ms2_sim_tol

    for f in tqdm(d.features):
    
        if f.ms2 is None:
            continue
        
        matched = None
        matched_peak_num = None
        signals = search_engine.clean_spectrum_for_search(precursor_mz=f.mz, peaks=f.ms2.signals, precursor_ions_removal_da=2.0)
        scores, peak_nums = search_engine.identity_search(precursor_mz=f.mz, peaks=signals, ms1_tolerance_in_da=d.params.mz_tol_ms1, 
                                                          ms2_tolerance_in_da=d.params.mz_tol_ms2, output_matched_peak_number=True)
        idx = np.argmax(scores)
        if scores[idx] > sim_tol:
            matched = search_engine[idx]
            matched_peak_num = peak_nums[idx]
            _assign_annotation_results_to_feature(f, score=scores[idx], matched=matched, matched_peak_num=matched_peak_num,
                                                  search_mode='identity_search')

        elif fuzzy_search:
            scores = search_engine.hybrid_search(precursor_mz=f.mz, peaks=signals, ms1_tolerance_in_da=d.params.mz_tol_ms1, 
                                                             ms2_tolerance_in_da=d.params.mz_tol_ms2)
            idx = np.argmax(scores)
            if scores[idx] > sim_tol:
                matched = search_engine[idx]
                matched_peak_num = None
                _assign_annotation_results_to_feature(f, score=scores[idx], matched=matched, matched_peak_num=matched_peak_num, 
                                                      search_mode='fuzzy_search')


def annotate_ms2(ms2, ms2_library_path, sim_tol=0.7, fuzzy_search=True):
    """
    Annotate MS2 spectra using MS2 database.

    Parameters
    ----------
    ms2 : Scan object
        MS2 spectrum.
    ms2_library_path : str
        The absolute path to the MS2 database. If not specified, the corresponding parameter from 
        the MS data file will be used.
    sim_tol : float
        The similarity threshold for MS2 annotation.
    fuzzy_search : bool
        Whether to further annotated the unmatched MS2 using fuzzy search.

    Returns
    -------
    score : float
        The similarity score.
    matched : dict
        The matched MS2 spectrum.
    matched_peak_num : int
        The number of matched peaks.
    search_mode : str
        The search mode, 'identity_search' or 'fuzzy_search'.
    """

    search_engine = load_ms2_db(ms2_library_path)

    signals = search_engine.clean_spectrum_for_search(precursor_mz=ms2.precursor_mz, peaks=ms2.signals, precursor_ions_removal_da=2.0)
    scores, peak_nums = search_engine.identity_search(precursor_mz=ms2.precursor_mz, peaks=signals, ms1_tolerance_in_da=0.01, 
                                                      ms2_tolerance_in_da=0.015, output_matched_peak_number=True)
    idx = np.argmax(scores)
    if scores[idx] > sim_tol:
        matched = search_engine[idx]
        matched_peak_num = peak_nums[idx]
        return scores[idx], matched, matched_peak_num, 'identity_search'

    elif fuzzy_search:
        scores = search_engine.hybrid_search(precursor_mz=ms2.precursor_mz, peaks=signals, ms1_tolerance_in_da=0.01, 
                                                         ms2_tolerance_in_da=0.015)
        idx = np.argmax(scores)
        if scores[idx] > sim_tol:
            matched = search_engine[idx]
            matched_peak_num = None
            return scores[idx], matched, None, 'fuzzy_search'
    
    return None, None, None, None


def feature_annotation_mzrt(features, path, mz_tol=0.01, rt_tol=0.3):
    """
    A function to annotate features based on a mzrt file (only .csv is supported now).

    parameters
    ----------
    features : list
        A list of features.
    path : str
        The path to the mzrt file in csv format.
    mz_tol : float
        The m/z tolerance for matching.
    rt_tol : float
        The RT tolerance for matching.

    returns
    ----------
    feature_table : pandas.DataFrame
        A DataFrame containing features with annotations.
    """

    # load the mzrt file
    df = pd.read_csv(path)
    features.sort(key=lambda x: x.highest_intensity, reverse=True)
    
    # match and annotate features
    feature_mz = np.array([f.mz for f in features])
    feature_rt = np.array([f.rt for f in features])
    to_anno = np.ones(len(features), dtype=bool)

    if 'adduct' not in df.columns:
        df['adduct'] = None

    for i in range(len(df)):
        mz = df.iloc[i,1]
        rt = df.iloc[i,2]
        v1 = np.abs(feature_mz - mz) < mz_tol
        v2 = np.abs(feature_rt - rt) < rt_tol
        matched_v = np.where(v1 & v2 & to_anno)[0]
        if len(matched_v) > 0:
            matched_idx = matched_v[0]
            features[matched_idx].annotation = df.iloc[i,0]
            features[matched_idx].search_mode = "mzrt_match"
            features[matched_idx].adduct_type = df['adduct'][i]
            to_anno[matched_idx] = False

    return features
          

def feature_to_feature_search(feature_list):
    """
    A function to calculate the MS2 similarity between features using hybrid search.

    Parameters
    ----------
    feature_list : list
        A list of AlignedFeature objects.
    
    Returns
    -------
    similarity_matrix : pandas.DataFrame
        similarity matrix between features.
    """

    entropy_search = index_feature_list(feature_list)
    dim = len(entropy_search.precursor_mz_array)
    ref_id = [item['id'] for item in entropy_search]
    results = np.zeros((dim, dim))

    for i, f in enumerate(feature_list):
        similarities = entropy_search.search(precursor_mz=f.mz, peaks=f.best_ms2.peaks)["hybrid_search"]
        matched = np.argmax(similarities)
        results[i, matched] = similarities[matched]

    df = pd.DataFrame(results, index=ref_id, columns=ref_id)
    
    return df


def index_feature_list(feature_list):
    """
    A function to index a list of features for spectrum entropy search.

    Parameters
    ----------
    feature_list : list
        A list of AlignedFeature objects.
    """
    
    db = []
    for f in feature_list:
        if f.ms2 is not None:
            tmp = {
                "id": f.id,
                "name": f.annotation,
                "precursor_mz": f.mz,
                "peaks": f.ms2.signals
            }
            db.append(tmp)

    entropy_search = FlashEntropySearch()
    entropy_search.build_index(db)

    return entropy_search


def output_ms2_to_msp(feature_table, output_path):
    """
    A function to output MS2 spectra to MSP format.

    Parameters
    ----------
    feature_table : pandas.DataFrame
        A DataFrame containing MS2 spectra.
    output_path : str
        The path to the output MSP file.
    """
    
    # check the output path to make sure it is a .msp file and it esists
    if not output_path.lower().endswith(".msp"):
        raise ValueError("The output path must be a .msp file.")

    with open(output_path, "w") as f:
        for i in range(len(feature_table)):
            if feature_table['MS2'][i] is None:
                continue

            if feature_table['annotation'][i] is None:
                name = "Unknown"
            else:
                name = str(feature_table['annotation'][i])

            peaks = re.findall(r"\d+\.\d+", feature_table['MS2'][i])
            f.write("NAME: " + name + "\n")
            f.write("PRECURSORMZ: " + str(feature_table['m/z'][i]) + "\n")
            f.write("PRECURSORTYPE: " + str(feature_table['adduct'][i]) + "\n")
            f.write("RETENTIONTIME: " + str(feature_table['RT'][i]) + "\n")
            f.write("SEARCHMODE: " + str(feature_table['search_mode'][i]) + "\n")
            f.write("FORMULA: " + str(feature_table['formula'][i]) + "\n")
            f.write("INCHIKEY: " + str(feature_table['InChIKey'][i]) + "\n")
            f.write("SMILES: " + str(feature_table['SMILES'][i]) + "\n")
            f.write("Num Peaks: " + str(int(len(peaks)/2)) + "\n")
            for j in range(len(peaks)//2):
                f.write(str(peaks[2*j]) + "\t" + str(peaks[2*j+1]) + "\n")
            f.write("\n")
            

def _correct_db(db):
    """
    Correct the MS2 database by changing the key names.
    """

    # make sure precursor_mz is in the db
    if 'precursor_mz' not in db[0].keys():
        for a in db:
            similar_key = [k for k in a.keys() if 'prec' in k and 'mz' in k]
            a['precursor_mz'] = float(a.pop(similar_key[0]))


def _assign_annotation_results_to_feature(f, score, matched, matched_peak_num, search_mode):
    """
    Assign annotation results to a feature.

    Parameters
    ----------
    f : Feature or AlignedFeature object
        Feature with MS2 spectrum to be annotated.
    score : float
        The similarity score.
    matched : dict
        The matched MS2 spectrum.
    matched_peak_num : int
        The number of matched peaks.
    search_mode : str
        The search mode, 'identity_search' or 'fuzzy_search'.
    """

    f.search_mode = search_mode
    f.similarity = score
    f.annotation = matched['name'] if 'name' in matched else None
    f.formula = matched['formula'] if 'formula' in matched else None
    f.matched_peak_number = matched_peak_num
    f.smiles = matched['smiles'] if 'smiles' in matched else None
    f.inchikey = matched['inchikey'] if 'inchikey' in matched else None
    f.matched_ms2 = convert_signals_to_string(matched['peaks'])
    f.matched_precursor_mz = matched['precursor_mz'] if 'precursor_mz' in matched else None
    f.matched_adduct_type = matched['precursor_type'] if 'precursor_type' in matched else None
    if search_mode == 'identity_search':
        f.adduct_type = matched['precursor_type'] if 'precursor_type' in matched else None
