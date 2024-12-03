# Author: Hauxu Yu

# A module to annotate metabolites based on their MS2 spectra

# Import modules
import os
import pickle
import numpy as np
import json
import pandas as pd
import re
from tqdm import tqdm

from ms_entropy import read_one_spectrum, FlashEntropySearch

def load_ms2_db(path):
    """
    A function to load the MS2 database in MSP format or pickle format.

    Parameters
    ----------
    path : str
        The path to the MS2 database in MSP format.    
    """

    print("Loading MS2 database...")
    # get extension of path
    ext = os.path.splitext(path)[1]

    if ext.lower() == '.msp':
        db =[]
        for a in read_one_spectrum(path):
            db.append(a)
        _correct_db(db)
        entropy_search = FlashEntropySearch()
        entropy_search.build_index(db)
        print("MS2 database loaded.")
        return entropy_search
    
    elif ext.lower() == '.pkl':
        entropy_search = pickle.load(open(path, 'rb'))
        print("MS2 database loaded.")
        return entropy_search
    
    elif ext.lower() == '.json':
        db = json.load(open(path, 'r'))
        entropy_search = FlashEntropySearch()
        entropy_search.build_index(db)
        print("MS2 database loaded.")
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
            f.annotation = matched['name']
            f.search_mode = 'identity_search'
            f.smiles = matched['smiles'] if 'smiles' in matched else None
            f.inchikey = matched['inchikey'] if 'inchikey' in matched else None
            f.matched_ms2 = _convert_ms2_signals_to_string(matched['peaks'])
            f.formula = matched['formula'] if 'formula' in matched else None
            f.adduct_type = matched['precursor_type']
            f.matched_mz = matched['precursor_mz']

        else:
            similarity = entropy_search.hybrid_search(precursor_mz=f.mz, peaks=f.ms2, ms1_tolerance_in_da=params.mz_tol_ms1, 
                                                              ms2_tolerance_in_da=params.mz_tol_ms2)
            idx = np.argmax(similarity)
            if similarity[idx] > params.ms2_sim_tol:
                matched = entropy_search[idx]
                matched = {k.lower():v for k,v in matched.items()}
                f.annotation = matched['name']
                f.search_mode = 'hybrid_search'
                f.similarity = similarity[idx]
                f.smiles = matched['smiles'] if 'smiles' in matched else None
                f.inchikey = matched['inchikey'] if 'inchikey' in matched else None
                f.matched_ms2 = _convert_ms2_signals_to_string(matched['peaks'])
                f.formula = matched['formula'] if 'formula' in matched else None
                f.matched_mz = matched['precursor_mz']
        
        f.ms2 = _convert_ms2_signals_to_string(f.ms2)

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

def annotate_ms2(ms2, seach_engine):
    pass




def feature_annotation_mzrt(features, path, default_adduct="[M+H]+", mz_tol=0.01, rt_tol=0.3):
    """
    A function to annotate features based on a mzrt file (only .csv is supported now).

    parameters
    ----------
    features : list
        A list of features.
    path : str
        The path to the mzrt file in csv format.
    default_adduct : str
        The default adduct for annotation.
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
    istd_df = pd.read_csv(path)

    # match and annotate features
    feature_mz = np.array([f.mz for f in features])
    feature_rt = np.array([f.rt for f in features])

    for i in range(len(istd_df)):
        mz = istd_df.iloc[i,1]
        rt = istd_df.iloc[i,2]
        matched_v = np.where(np.logical_and(np.abs(feature_mz - mz) < mz_tol, np.abs(feature_rt - rt) < rt_tol))[0]
        if len(matched_v) > 0:
            matched_idx = matched_v[0]
            features[matched_idx].adduct = default_adduct
            features[matched_idx].annotation = istd_df.iloc[i,0]
            features[matched_idx].search_mode = "mzrt_match"

    return features


def annoatate_peaks(peak_list, precursor_mz_list, ms2_library=None, ms2_library_path=None,
                    search_mode='identity_search', ms1_tol=0.01, ms2_tol=0.015, top_n=1, clean_spec=True):
    """
    A function to annotate peaks based on a MS2 database.

    Parameters
    ----------
    peak_list : list
        A list of peaks. Each peak is a list or numpy array with shape (N, 2), 
        N is the number of peaks. The format of the peaks is [[mz1, intensity1], [mz2, intensity2], ...].
    precursor_mz_list : list
        A list of precursor m/z values.
    ms2_library : FlashEntropySearch object
        A FlashEntropySearch object.
    ms2_library_path : str
        The path to the MS2 database (.pkl, .msp, or .json).
    search_mode : str
        The search mode. Options: 'identity_search', 'hybrid_search', or 'all' for both.
    ms1_tol : float
        The m/z tolerance for MS1 search.
    ms2_tol : float
        The m/z tolerance for MS2 search.
    top_n : int
        The number of top MS2 spectra with the highest spectral similarities.
    clean_spec : bool
        Whether to automatically clean the spectrum before searching. If you prefer to clean the spectrum using 
        other methods or parameters, set it to False and clean the spectrum before calling this function.
        By default, spectrum cleanning will 
        1. Remove ions > precursor_mz-2.0 Da.
        2. Remove peaks with intensity less than 1% of the base peak.
        3. Centroid the spectrum by grouping fragment ions within 0.05 Da.
    
    Returns
    ----------
    results : list
        A list of annotations. Each annotation is a dictionary with keys 'identity_search' and 'hybrid_search', 
        where the values are lists of top_n annotations as dictionaries with keys 'matched_ms2', 'similarity', and 'matched_peak_number'.
    """

    # check if the length of peak_list and precursor_mz_list are the same
    if len(peak_list) != len(precursor_mz_list):
        raise ValueError("The length of peak_list and precursor_mz_list must be the same.")
    
    # load the MS2 database
    if ms2_library is not None:
        entropy_search = ms2_library
    elif ms2_library_path is not None:
        entropy_search = load_ms2_db(ms2_library_path)
    else:
        raise ValueError("Please provide the MS2 database.")
    
    results = [{'identity_search': [], 'hybrid_search': []} for i in range(len(peak_list))]
    
    if search_mode == 'all' or search_mode == 'identity_search':
        for i in range(len(peak_list)):
            peaks = peak_list[i]
            if peaks is not None and len(peaks) > 0:
                if clean_spec:
                    peaks = entropy_search.clean_spectrum_for_search(precursor_mz_list[i], peaks, precursor_ions_removal_da=2.0)
                entropy_similarity, matched_peaks_number = entropy_search.identity_search(precursor_mz=precursor_mz_list[i], peaks=peaks, 
                                                                                        ms1_tolerance_in_da=ms1_tol, ms2_tolerance_in_da=ms2_tol, output_matched_peak_number=True)
                matched_idx = np.argsort(entropy_similarity)[::-1][:top_n]
                for idx in matched_idx:
                    results[i]['identity_search'].append({
                        'matched_ms2': entropy_search[idx],
                        'similarity': entropy_similarity[idx],
                        'matched_peak_number': matched_peaks_number[idx]
                    })
            else:
                results[i]['identity_search'].append({
                    'matched_ms2': None,
                    'similarity': 0,
                    'matched_peak_number': None
                })
    
    if search_mode == 'all' or search_mode == 'hybrid_search':
        for i in range(len(peak_list)):
            peaks = peak_list[i]
            if peaks is not None and len(peaks) > 0:
                if clean_spec:
                    peaks = entropy_search.clean_spectrum_for_search(precursor_mz_list[i], peaks, precursor_ions_removal_da=2.0)
                entropy_similarity = entropy_search.hybrid_search(precursor_mz=precursor_mz_list[i], peaks=peaks, 
                                                                ms1_tolerance_in_da=ms1_tol, ms2_tolerance_in_da=ms2_tol)
                matched_idx = np.argsort(entropy_similarity)[::-1][:top_n]
                for idx in matched_idx:
                    results[i]['hybrid_search'].append({
                        'matched_ms2': entropy_search[idx],
                        'similarity': entropy_similarity[idx]
                    })
            else:
                results[i]['hybrid_search'].append({
                    'matched_ms2': 0,
                    'similarity': None
                })
    
    return results
          

def feature_to_feature_search(feature_list, sim_tol=0.7):
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


def extract_signals_from_string(ms2):
    """
    Extract signals from MS2 spectrum in string format.

    Parameters
    ----------
    ms2 : str
        MS2 spectrum in string format. Format: "mz1;intensity1|mz2;intensity2|..."
        example: "100.0;1000.0|200.0;2000.0|300.0;3000.0|"
    
    returns
    ----------
    peaks : numpy.array
        Peaks in numpy array format.
    """
    
    # Use findall function to extract all numbers matching the pattern
    numbers = re.findall(r'\d+\.\d+', ms2)
    
    # Convert the extracted numbers from strings to floats
    numbers = [float(num) for num in numbers]
    
    numbers = np.array(numbers).reshape(-1, 2)

    return numbers


def _convert_ms2_signals_to_string(signals):
    """
    Convert peaks to string format.

    Parameters
    ----------
    signals : numpy.array
        MS2 signals organized as [[mz1, intensity1], [mz2, intensity2], ...]

    Returns
    -------
    ms2 : str
        MS2 spectrum in string format: "mz1;intensity1|mz2;intensity2|..."
    """
    
    ms2 = ""
    for i in range(len(signals)):
        ms2 += str(np.round(signals[i, 0], decimals=4)) + ";" + str(np.round(signals[i, 1], decimals=4)) + "|"
    ms2 = ms2[:-1]
    
    return ms2


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
    f.annotation = matched['name']
    f.formula = matched['formula'] if 'formula' in matched else None
    f.matched_peak_number = matched_peak_num
    f.smiles = matched['smiles'] if 'smiles' in matched else None
    f.inchikey = matched['inchikey'] if 'inchikey' in matched else None
    f.matched_ms2 = _convert_ms2_signals_to_string(matched['peaks'])
    f.matched_precursor_mz = matched['precursor_mz']
    f.matched_adduct_type = matched['precursor_type']
