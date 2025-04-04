# Author: Huaxu Yu

# A module to annotate metabolites based on their m/z, retention time and MS2 spectra

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

"""
Format of MS2 database in MassCube
====================================================================================

1. pickle format

A FlashEntropySearch object that contains the MS2 database. ms_entropy version 1.2.2 is highly recommended
to generate this object (other versions may not work). See masscube documentation for how to generate this object.

https://huaxuyu.github.io/masscubedocs/docs/workflows/database/

2. msp format

Within each block, key is defined as:
    - NAME: the name of the compound
    - PRECURSORMZ: the precursor m/z
    - PRECURSORTYPE: the adduct type
    - IONMODE: the ion mode
    - RETENTIONTIME: the retention time
    - CCS: collision cross section
    - FORMULA: the molecular formula
    - ONTOLOGY: the ontology of the compound
    - SMILES: the SMILES string
    - INCHIKEY: the InChIKey
    - INSTRUMENTTYPE: the instrument type
    - COLLISIONENERGY: the collision energy
    - DATABASE: the database name
    - COMMENT: the comment
    - Num Peaks: the number of peaks
    - [mz1 intensity1]: the m/z and intensity of each fragment
    - [mz2 intensity2]: the m/z and intensity of each fragment
    - ...

Example:
    NAME: L-PHENYLALANINE
    PRECURSORMZ: 166.086013793945
    PRECURSORTYPE: [M+H]+
    IONMODE: Positive
    RETENTIONTIME: 3.30520009994507
    CCS: 136.819671630859
    FORMULA: C9H11NO2
    ONTOLOGY: Phenylalanine and derivatives
    SMILES: C1=CC=C(C=C1)C[C@@H](C(=O)O)N
    INCHIKEY: COLNVLDHVKWLRT-QMMMGPOBSA-N
    INSTRUMENTTYPE: LC-ESI-QFT
    COLLISIONENERGY: 35.0 eV
    DATABASE: EMBL-MCF_spec98214
    COMMENT: DB#=EMBL-MCF_spec98214; origin=EMBL - Metabolomics Core Facility Spectral Library
    Num Peaks: 7
    103.054	15
    107.049	14
    120.081	1000
    121.084	16
    131.049	41
    149.059	16
    166.086	56

3. a list of dictionaries (or json format)

A list of dictionaries, each dictionary contains the following keys:

{
    "name": the name of the compound
    "precursor_mz": the precursor m/z
    "precursor_type": the precursor ion type
    "ion_mode": the ion mode
    "retention_time": the retention time
    "ccs": the collision cross section
    "formula": the molecular formula
    "ontology": the ontology of the compound
    "smiles": the SMILES string
    "inchikey": the InChIKey
    "instrument_type": the instrument type
    "collision_energy": the collision energy
    "database": the database name
    "comment": the comment
    "num_peaks": the number of peaks
    "peaks": a list of lists, each sublist contains two elements: m/z and intensity: [[mz1, intensity1], [mz2, intensity2], ...]
}

Example:
{
    "name": "L-PHENYLALANINE",
    "precursor_mz": 166.086013793945, 
    "precursor_type": "[M+H]+"
    "ion_mode": "Positive", 
    "retention_time": "3.30520009994507", 
    "ccs": "136.819671630859", 
    "formula": "C9H11NO2", 
    "ontology": "Phenylalanine and derivatives", 
    "smiles": "C1=CC=C(C=C1)C[C@@H](C(=O)O)N", 
    "inchikey": "COLNVLDHVKWLRT-QMMMGPOBSA-N", 
    "instrument_type": "LC-ESI-QFT", 
    "collision_energy": "35.0 eV", 
    "database": "EMBL-MCF_spec98214"
    "comment": "DB#=EMBL-MCF_spec98214; origin=EMBL - Metabolomics Core Facility Spectral Library", 
    "num_peaks": "7",
    "peaks": [["103.054", "15"], ["107.049", "14"], ["120.081", "1000"], ["121.084", "16"], ["131.049", "41"], ["149.059", "16"], ["166.086", "56"]], 
}

"""

"""
Search modes in MassCube
====================================================================================

Features (i.e. unique m/z-RT pairs) can be annotated in different ways with different confidence. Search modes summarize the way to search and annotate features.

1. mz_rt_ms2_match

Features are matched to database compounds with m/z, retention time and MS2 spectra.

2. mz_rt_match

Features are matched to database compounds with m/z and retention time.

3. mz_ms2_match

Features are matched to database compounds with m/z and MS2 spectra.

4. fuzzy_search (ms2 match or analog search)

Features are matched to database compounds with MS2 spectra using fuzzy search. Experimental and database precursor m/z values can be different.

Because matching by m/z value only is very likely to have false positives, this function is not provided in MassCube.

"""


def load_ms2_db(path):
    """
    Load MS2 database in pickle, msp, or json format.

    Parameters
    ----------
    path : str
        The path to the MS2 database.

    Returns
    -------
    entropy_search : FlashEntropySearch object
        The MS2 database.
    """

    print("\tLoading MS2 database...")
    # get extension of path
    ext = os.path.splitext(path)[1]

    if ext.lower() == '.msp':
        db =[]
        for a in read_one_spectrum(path):
            if 'precursortype' in a.keys():
                a['precursor_type'] = a.pop('precursortype')
            if 'ionmode' in a.keys():
                a['ion_mode'] = a.pop('ionmode')
            if 'retentiontime' in a.keys():
                a['retention_time'] = float(a.pop('retentiontime'))
            db.append(a)
        _correct_db(db)
        entropy_search = FlashEntropySearch(intensity_weight=None)
        entropy_search.build_index(db)
        print("\tMS2 database has been loaded.")
        return entropy_search
    
    elif ext.lower() == '.pkl':
        entropy_search = pickle.load(open(path, 'rb'))
        print("\tMS2 database has been loaded.")
        # check if intensity_weight is an attribute
        if not hasattr(entropy_search.entropy_search, 'intensity_weight'):
            raise ValueError("Please new MS/MS database is required for MassCube ver. 1.2 or later. Please download from: https://zenodo.org/records/14991522.")
        return entropy_search
    
    elif ext.lower() == '.json':
        db = json.load(open(path, 'r'))
        entropy_search = FlashEntropySearch(intensity_weight=None)
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

    Returns
    -------
    features : list
        A list of AlignedFeature objects with MS2 annotation.
    """

    entropy_search = load_ms2_db(params.ms2_library_path)

    if params.consider_rt:
        rt_arr = np.zeros(len(entropy_search.precursor_mz_array))+np.inf
        for i, ms2 in enumerate(entropy_search):
            if 'retention_time' in ms2:
                rt_arr[i] = ms2['retention_time']

    for f in tqdm(features):
        if len(f.ms2_seq) == 0:
            continue
        
        matched = None  # matched MS2 spectrum in the database

        parsed_ms2 = [] # experimental MS2 spectra (top num) to search
        for file_name, ms2 in f.ms2_seq:
            signals = extract_signals_from_string(ms2)
            signals = entropy_search.clean_spectrum_for_search(f.mz, signals, precursor_ions_removal_da=params.precursor_mz_offset)
            parsed_ms2.append([file_name, signals])
        
        parsed_ms2.sort(key=lambda x: np.sum(x[1][:, 1]), reverse=True)
        parsed_ms2 = parsed_ms2[:num]
        f.ms2_reference_file = parsed_ms2[0][0]
        f.ms2 = parsed_ms2[0][1]

        if params.consider_rt:
            rt_boo = np.abs(rt_arr - f.rt) < params.rt_tol_annotation

        similarities = []
        matched_nums = []

        for file_name, signals in parsed_ms2:
            similarity, matched_num = entropy_search.identity_search(precursor_mz=f.mz, peaks=signals, ms1_tolerance_in_da=params.mz_tol_ms1,
                                                                     ms2_tolerance_in_da=params.mz_tol_ms2, output_matched_peak_number=True)
            similarities.append(similarity)
            matched_nums.append(matched_num)
        
        if params.consider_rt:
            similarities_rt = [s*rt_boo for s in similarities]
            tmp = [np.max(s) for s in similarities_rt]
            if np.max(tmp) > params.ms2_sim_tol:
                idx_tmp = np.argmax(tmp)
                f.ms2_reference_file = parsed_ms2[idx_tmp][0]
                f.ms2 = parsed_ms2[idx_tmp][1]
                matched_idx = np.argmax(similarities_rt[idx_tmp])
                matched = entropy_search[matched_idx]
                matched = {k.lower():v for k,v in matched.items()}
                _assign_annotation_results_to_feature(f, score=similarities_rt[idx_tmp][matched_idx],matched=matched, 
                                                      matched_peak_num=matched_nums[idx_tmp][matched_idx], search_mode='identity_search_with_rt')
        
        # if the feature cannot be annotated by considering retention time
        if matched is None:
            tmp = [np.max(s) for s in similarities]
            if np.max(tmp) > params.ms2_sim_tol:    
                idx_tmp = np.argmax(tmp)
                f.ms2_reference_file = parsed_ms2[idx_tmp][0]
                f.ms2 = parsed_ms2[idx_tmp][1]
                matched_idx = np.argmax(similarities[idx_tmp])
                matched = entropy_search[matched_idx]
                matched = {k.lower():v for k,v in matched.items()}
                _assign_annotation_results_to_feature(f, score=similarities[idx_tmp][matched_idx], matched=matched,
                                                      matched_peak_num=matched_nums[idx_tmp][matched_idx], search_mode='identity_search')

        # if the feature cannot be annotated by MS2 identity search
        if matched is None and params.fuzzy_search:
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


def annotate_features(d, sim_tol=None, fuzzy_search=True, ms2_library_path=None, consider_rt=False):
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
    consider_rt : bool
        Whether to consider retention time in the annotation. Default is False.
    """

    if ms2_library_path is None:
        search_engine = load_ms2_db(d.params.ms2_library_path)
    else:
        search_engine = load_ms2_db(ms2_library_path)

    if sim_tol is None:
        sim_tol = d.params.ms2_sim_tol
    
    if consider_rt:
        rt_arr = np.zeros(len(search_engine.precursor_mz_array))+np.inf
        for i, ms2 in enumerate(search_engine):
            if 'retention_time' in ms2:
                rt_arr[i] = ms2['retention_time']

    for f in tqdm(d.features):
    
        if f.ms2 is None:
            continue
        
        matched = None
        matched_peak_num = None
        signals = search_engine.clean_spectrum_for_search(precursor_mz=f.mz, peaks=f.ms2.signals, precursor_ions_removal_da=2.0)
        scores, peak_nums = search_engine.identity_search(precursor_mz=f.mz, peaks=signals, ms1_tolerance_in_da=d.params.mz_tol_ms1, 
                                                          ms2_tolerance_in_da=d.params.mz_tol_ms2, output_matched_peak_number=True)
        if consider_rt:
            rt_boo = np.abs(rt_arr - f.rt) < d.params.rt_tol_annotation
            scores_rt = scores * rt_boo
            idx = np.argmax(scores_rt)
            if scores_rt[idx] > sim_tol:
                matched = search_engine[idx]
                matched = {k.lower():v for k,v in matched.items()}
                matched_peak_num = peak_nums[idx]
                _assign_annotation_results_to_feature(f, score=scores_rt[idx], matched=matched, matched_peak_num=matched_peak_num, 
                                                      search_mode='identity_search_with_rt')
        
        if matched is None:
            idx = np.argmax(scores)
            if scores[idx] > sim_tol:
                matched = search_engine[idx]
                matched = {k.lower():v for k,v in matched.items()}
                matched_peak_num = peak_nums[idx]
                _assign_annotation_results_to_feature(f, score=scores[idx], matched=matched, matched_peak_num=matched_peak_num,
                                                      search_mode='identity_search')

        if matched is None and fuzzy_search:
            scores = search_engine.hybrid_search(precursor_mz=f.mz, peaks=signals, ms1_tolerance_in_da=d.params.mz_tol_ms1, 
                                                             ms2_tolerance_in_da=d.params.mz_tol_ms2)
            idx = np.argmax(scores)
            if scores[idx] > sim_tol:
                matched = search_engine[idx]
                matched_peak_num = None
                _assign_annotation_results_to_feature(f, score=scores[idx], matched=matched, matched_peak_num=matched_peak_num, 
                                                      search_mode='fuzzy_search')


def feature_annotation_mzrt(features, path, mz_tol=0.01, rt_tol=0.3):
    """
    Annotate features based on a mzrt file (only .csv is supported now).

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
    features : list
        A list of features with annotation.
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
    if 'inchikey' not in df.columns:
        df['inchikey'] = None
    if 'formula' not in df.columns:
        df['formula'] = None
    if 'smiles' not in df.columns:
        df['smiles'] = None

    for i in range(len(df)):
        mz = df.iloc[i,1]
        rt = df.iloc[i,2]
        v1 = np.abs(feature_mz - mz) < mz_tol
        v2 = np.abs(feature_rt - rt) < rt_tol
        matched_v = np.where(v1 & v2 & to_anno)[0]
        if len(matched_v) > 0:
            matched_idx = matched_v[0]
            _assign_mzrt_annotation_results_to_feature(f=features[matched_idx], annotation=df.iloc[i,0], adduct=df['adduct'][i], 
                                                       inchikey=df['inchikey'][i], formula=df['formula'][i], smiles=df['smiles'][i],
                                                       matched_precursor_mz=mz, matched_retention_time=rt)
            to_anno[matched_idx] = False

    return features


def feature_to_feature_search(feature_list):
    """
    A function to calculate the MS2 similarity between features using fuzzy search.

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
    A helper function to index a list of features for spectrum entropy search.

    Parameters
    ----------
    feature_list : list
        A list of AlignedFeature objects.

    Returns
    -------
    entropy_search : FlashEntropySearch object
        The indexed feature list.
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


def index_msp_to_pkl(msp_path, output_path=None):
    """
    A function to index MSP file to pickle format.

    Parameters
    ----------
    msp_path : str
        The path to the MSP file.
    output_path : str
        The path to the output pickle file.
    """

    file_name = os.path.basename(msp_path).split(".")[0]

    if output_path is None:
        output_path = os.path.dirname(msp_path)

    db = []
    for a in read_one_spectrum(msp_path):
        if 'precursortype' in a.keys():
            a['precursor_type'] = a.pop('precursortype')
        if 'ionmode' in a.keys():
            a['ion_mode'] = a.pop('ionmode')
        if 'retentiontime' in a.keys():
            a['retention_time'] = a.pop('retentiontime')
        db.append(a)

    _correct_db(db)
    entropy_search = FlashEntropySearch()
    entropy_search.build_index(db)

    pickle.dump(entropy_search, open(os.path.join(output_path, file_name + ".pkl"), 'wb'))


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


def _assign_mzrt_annotation_results_to_feature(f, annotation, adduct, inchikey, formula, smiles, matched_precursor_mz, 
                                               matched_retention_time):
    """
    Assign annotation results to a feature.

    Parameters
    ----------
    f : Feature or AlignedFeature object
        Feature to be annotated.
    annotation : str
        The compound name.
    adduct : str
        The adduct type.
    inchikey : str
        The InChIKey.
    formula : str
        The molecular formula.
    smiles : str
        The SMILES string.
    matched_precursor_mz : float
        The matched precursor m/z.
    matched_retention_time : float
        The matched retention time.
    """

    f.search_mode = 'mzrt_search'
    f.similarity = None
    f.annotation = annotation
    f.formula = formula
    f.matched_peak_number = None
    f.smiles = smiles
    f.inchikey = inchikey
    f.matched_precursor_mz = matched_precursor_mz
    f.matched_retention_time = matched_retention_time
    f.matched_adduct_type = adduct
    f.adduct_type = adduct
    f.matched_ms2 = None