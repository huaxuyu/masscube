# Author: Huaxu Yu

# A module to annotate metabolites based on their m/z, retention time and MS2 spectra

# imports
import os
import pickle
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from ms_entropy import read_one_spectrum, FlashEntropySearch

from .utils_functions import extract_signals_from_string, convert_signals_to_string

"""
MS/MS database format
====================================================================================

1. pickle format

A FlashEntropySearch object that contains the MS/MS database. ms_entropy version 1.2.2 is highly recommended
to generate this object (other versions may not work). See masscube documentation for how to generate this object.

https://huaxuyu.github.io/masscubedocs/docs/untargeted_metabolomics/database/

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
    - mz1 intensity1: the m/z and intensity of each fragment
    - mz2 intensity2: the m/z and intensity of each fragment
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

Matched by precursor m/z, retention time and MS/MS spectra.

2. mz_rt_match

Matched by precursor m/z and retention time.

3. mz_ms2_match

Matched by precursor m/z and MS/MS spectra.

4. fuzzy_search (also called analog search or hybrid search)

More about hybrid search: https://www.nature.com/articles/s41592-023-02012-9
"""


def load_ms2_db(path: str):
    """
    Load a MS/MS database in either pickle, msp, or json format.

    Parameters
    ----------
    path : str
        The path to the MS/MS database.

    Returns
    -------
    entropy_search : FlashEntropySearch object
    """

    if path is None or _is_missing_value(path) or str(path).strip() == "":
        raise ValueError("MS/MS database path is required.")

    path = os.path.expanduser(str(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MS/MS database file does not exist: {path}")

    print("\tLoading MS/MS database...")

    ext = os.path.splitext(path)[1].lower()

    if ext == '.msp':
        entropy_search = _read_msp(path)
    
    elif ext == '.pkl':
        entropy_search = _read_pickle(path)
    
    elif ext == '.json':
        entropy_search = _read_json(path)
    else:
        raise ValueError(
            "Unsupported MS/MS database format '{}'. Please provide pkl, msp, or json.".format(ext)
        )

    if entropy_search is None or len(entropy_search.precursor_mz_array) == 0:
        raise ValueError(f"No valid MS/MS spectra were loaded from: {path}")
    
    print("\tMS/MS database has been loaded.")
    
    return entropy_search


def annotate_aligned_features(features: list, params, num: int = 5):
    """
    Annotate aligned features using MS/MS databases.
    
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

    ion_mode_mask = _build_ion_mode_mask(entropy_search, params.ion_mode)

    if params.consider_rt:
        rt_arr = _build_retention_time_array(entropy_search)

    for f in tqdm(features):
        
        if len(f.ms2_seq) == 0:
            continue
        
        parsed_ms2 = []
        for s in f.ms2_seq[:num]:
            signals = _clean_ms2_signals(
                entropy_search=entropy_search,
                precursor_mz=f.mz,
                signals=s.signals,
                precursor_mz_offset=params.precursor_mz_offset,
            )
            if len(signals) > 0:
                parsed_ms2.append((s, signals))

        if len(parsed_ms2) == 0:
            continue

        selected_scan, selected_signals = parsed_ms2[0]

        if params.consider_rt:
            rt_mask = np.abs(rt_arr - f.rt) < params.rt_tol_annotation

        similarities = []
        matched_nums = []
        matched = None  # matched MS2 spectrum in the database

        for scan, signals in parsed_ms2:
            similarity, matched_num = entropy_search.identity_search(precursor_mz=f.mz, peaks=signals, ms1_tolerance_in_da=params.mz_tol_ms1,
                                                                     ms2_tolerance_in_da=params.mz_tol_ms2, output_matched_peak_number=True)
            similarities.append(similarity * ion_mode_mask)
            matched_nums.append(matched_num)
        
        if params.consider_rt:
            similarities_rt = [s*rt_mask for s in similarities]
            tmp = [np.max(s) for s in similarities_rt]
            if np.max(tmp) > params.ms2_sim_tol:
                idx_tmp = np.argmax(tmp)
                selected_scan, selected_signals = parsed_ms2[idx_tmp]
                f.ms2_reference_file = getattr(selected_scan, "file_name", None)
                matched_idx = np.argmax(similarities_rt[idx_tmp])
                matched = _normalize_record(entropy_search[matched_idx])
                _assign_annotation_results_to_feature(f, score=similarities_rt[idx_tmp][matched_idx],matched=matched, 
                                                      matched_peak_num=matched_nums[idx_tmp][matched_idx], search_mode='identity_search_with_rt',
                                                      ms2_scan_idx=selected_scan.id, precursor_ion_fraction=selected_scan.precursor_ion_fraction)
                                                      
        
        # if the feature cannot be annotated by considering retention time
        if matched is None:
            tmp = [np.max(s) for s in similarities]
            if np.max(tmp) > params.ms2_sim_tol:    
                idx_tmp = np.argmax(tmp)
                selected_scan, selected_signals = parsed_ms2[idx_tmp]
                f.ms2_reference_file = getattr(selected_scan, "file_name", None)
                matched_idx = np.argmax(similarities[idx_tmp])
                matched = _normalize_record(entropy_search[matched_idx])
                _assign_annotation_results_to_feature(f, score=similarities[idx_tmp][matched_idx], matched=matched,
                                                      matched_peak_num=matched_nums[idx_tmp][matched_idx], search_mode='identity_search',
                                                      ms2_scan_idx=selected_scan.id, precursor_ion_fraction=selected_scan.precursor_ion_fraction)

        # if the feature cannot be annotated by MS2 identity search
        if matched is None and params.fuzzy_search:
            selected_scan, selected_signals = parsed_ms2[0]
            similarity = entropy_search.hybrid_search(precursor_mz=f.mz, peaks=selected_signals, ms1_tolerance_in_da=params.mz_tol_ms1, 
                                                      ms2_tolerance_in_da=params.mz_tol_ms2)
            similarity = similarity * ion_mode_mask
            idx = np.argmax(similarity)
            if similarity[idx] > params.ms2_sim_tol:
                matched = _normalize_record(entropy_search[idx])
                _assign_annotation_results_to_feature(f, score=similarity[idx], matched=matched, 
                                                      matched_peak_num=None, search_mode='fuzzy_search',
                                                      ms2_scan_idx=selected_scan.id, precursor_ion_fraction=selected_scan.precursor_ion_fraction)
        
        if getattr(f, "ms2_reference_file", None) is None:
            f.ms2_reference_file = getattr(selected_scan, "file_name", None)
        if getattr(f, "ms2_scan_idx", None) is None:
            f.ms2_scan_idx = getattr(selected_scan, "id", None)
        if getattr(f, "ms2_pif", None) is None:
            f.ms2_pif = getattr(selected_scan, "precursor_ion_fraction", None)
        f.ms2 = convert_signals_to_string(selected_signals)

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
        entropy_search = load_ms2_db(d.params.ms2_library_path)
    else:
        entropy_search = load_ms2_db(ms2_library_path)
    
    ion_mode_mask = _build_ion_mode_mask(entropy_search, d.params.ion_mode)

    if sim_tol is None:
        sim_tol = d.params.ms2_sim_tol
    
    if consider_rt:
        rt_arr = _build_retention_time_array(entropy_search)

    for f in tqdm(d.features):
    
        if f.ms2 is None:
            continue
        
        signals = _clean_ms2_signals(
            entropy_search=entropy_search,
            precursor_mz=f.mz,
            signals=f.ms2.signals,
            precursor_mz_offset=d.params.precursor_mz_offset,
        )
        if len(signals) == 0:
            continue
        
        matched = None
        matched_peak_num = None
        scores, peak_nums = entropy_search.identity_search(precursor_mz=f.mz, peaks=signals, ms1_tolerance_in_da=d.params.mz_tol_ms1, 
                                                          ms2_tolerance_in_da=d.params.mz_tol_ms2, output_matched_peak_number=True)
        
        scores = ion_mode_mask * scores
        if consider_rt:
            rt_boo = np.abs(rt_arr - f.rt) < d.params.rt_tol_annotation
            scores_rt = scores * rt_boo
            idx = np.argmax(scores_rt)
            if scores_rt[idx] > sim_tol:
                matched = _normalize_record(entropy_search[idx])
                matched_peak_num = peak_nums[idx]
                _assign_annotation_results_to_feature(f, score=scores_rt[idx], matched=matched, matched_peak_num=matched_peak_num, 
                                                      search_mode='identity_search_with_rt',
                                                      ms2_scan_idx=getattr(f.ms2, "id", None),
                                                      precursor_ion_fraction=getattr(f.ms2, "precursor_ion_fraction", None))
        
        if matched is None:
            idx = np.argmax(scores)
            if scores[idx] > sim_tol:
                matched = _normalize_record(entropy_search[idx])
                matched_peak_num = peak_nums[idx]
                _assign_annotation_results_to_feature(f, score=scores[idx], matched=matched, matched_peak_num=matched_peak_num,
                                                      search_mode='identity_search',
                                                      ms2_scan_idx=getattr(f.ms2, "id", None),
                                                      precursor_ion_fraction=getattr(f.ms2, "precursor_ion_fraction", None))

        if matched is None and fuzzy_search:
            scores = entropy_search.hybrid_search(precursor_mz=f.mz, peaks=signals, ms1_tolerance_in_da=d.params.mz_tol_ms1, 
                                                             ms2_tolerance_in_da=d.params.mz_tol_ms2)
            scores = ion_mode_mask * scores
            idx = np.argmax(scores)
            if scores[idx] > sim_tol:
                matched = _normalize_record(entropy_search[idx])
                matched_peak_num = None
                _assign_annotation_results_to_feature(f, score=scores[idx], matched=matched, matched_peak_num=matched_peak_num, 
                                                      search_mode='fuzzy_search',
                                                      ms2_scan_idx=getattr(f.ms2, "id", None),
                                                      precursor_ion_fraction=getattr(f.ms2, "precursor_ion_fraction", None))


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

    df = pd.read_csv(path)
    name_col, mz_col, rt_col = _resolve_mzrt_columns(df)
    features.sort(key=lambda x: getattr(x, "highest_intensity", getattr(x, "peak_height", 0)) or 0, reverse=True)
    
    # match and annotate features
    feature_mz = np.array([f.mz for f in features])
    feature_rt = np.array([f.rt for f in features])
    to_anno = np.ones(len(features), dtype=bool)

    adduct_col = _find_optional_column(df, ["adduct", "precursor_type", "precursortype"])
    inchikey_col = _find_optional_column(df, ["inchikey", "inchi_key"])
    formula_col = _find_optional_column(df, ["formula", "molecular_formula"])
    smiles_col = _find_optional_column(df, ["smiles"])

    for _, row in df.iterrows():
        mz = _safe_float(row[mz_col])
        rt = _safe_float(row[rt_col])
        if not np.isfinite(mz) or not np.isfinite(rt):
            continue
        v1 = np.abs(feature_mz - mz) < mz_tol
        v2 = np.abs(feature_rt - rt) < rt_tol
        matched_v = np.where(v1 & v2 & to_anno)[0]
        if len(matched_v) > 0:
            mz_score = np.abs(feature_mz[matched_v] - mz) / max(mz_tol, 1e-12)
            rt_score = np.abs(feature_rt[matched_v] - rt) / max(rt_tol, 1e-12)
            matched_idx = matched_v[np.argmin(mz_score + rt_score)]
            _assign_mzrt_annotation_results_to_feature(f=features[matched_idx],
                                                       annotation=row[name_col],
                                                       adduct=_row_value(row, adduct_col),
                                                       inchikey=_row_value(row, inchikey_col),
                                                       formula=_row_value(row, formula_col),
                                                       smiles=_row_value(row, smiles_col),
                                                       matched_precursor_mz=mz, matched_retention_time=rt)
            to_anno[matched_idx] = False

    return features


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

    required_cols = ["MS2", "m/z", "adduct", "RT", "search_mode", "formula", "InChIKey", "SMILES"]
    missing_cols = [col for col in required_cols if col not in feature_table.columns]
    if len(missing_cols) > 0:
        raise ValueError("Feature table is missing required columns: {}".format(", ".join(missing_cols)))

    with open(output_path, "w") as f:
        for _, row in feature_table.iterrows():
            ms2 = row["MS2"]
            if _is_missing_value(ms2):
                continue

            name = _format_text(row["annotation"], default="Unknown") if "annotation" in feature_table.columns else "Unknown"
            peaks = extract_signals_from_string(str(ms2))

            f.write("NAME: " + name + "\n")
            f.write("PRECURSORMZ: " + _format_text(row["m/z"]) + "\n")
            f.write("PRECURSORTYPE: " + _format_text(row["adduct"]) + "\n")
            f.write("RETENTIONTIME: " + _format_text(row["RT"]) + "\n")
            f.write("SEARCHMODE: " + _format_text(row["search_mode"]) + "\n")
            f.write("FORMULA: " + _format_text(row["formula"]) + "\n")
            f.write("INCHIKEY: " + _format_text(row["InChIKey"]) + "\n")
            f.write("SMILES: " + _format_text(row["SMILES"]) + "\n")
            f.write("Num Peaks: " + str(len(peaks)) + "\n")
            for peak in peaks:
                f.write(str(peak[0]) + "\t" + str(peak[1]) + "\n")
            f.write("\n")


def _is_missing_value(value):
    """
    Return True for scalar missing values while leaving arrays/lists alone.
    """

    if value is None:
        return True
    if isinstance(value, (list, tuple, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _safe_float(value, default=np.nan):
    """
    Convert a scalar to float, returning default for missing or invalid values.
    """

    if _is_missing_value(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_record(record):
    """
    Normalize database keys and common aliases for downstream access.
    """

    normalized = {str(k).strip().lower(): v for k, v in dict(record).items()}

    aliases = {
        "precursortype": "precursor_type",
        "ionmode": "ion_mode",
        "retentiontime": "retention_time",
    }
    for old_key, new_key in aliases.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized[old_key]

    if "precursor_mz" not in normalized:
        mz_keys = [k for k in normalized.keys() if "prec" in k and "mz" in k]
        if len(mz_keys) > 0:
            normalized["precursor_mz"] = normalized[mz_keys[0]]

    return normalized


def _normalize_ion_mode(value):
    if _is_missing_value(value):
        return None
    value = str(value).strip().lower()
    if "positive" in value or value in {"pos", "+"}:
        return "positive"
    if "negative" in value or value in {"neg", "-"}:
        return "negative"
    return value


def _build_ion_mode_mask(entropy_search, ion_mode):
    """
    Build a mask for database entries matching ion mode.

    Missing ion-mode metadata is treated as compatible rather than excluding
    otherwise valid libraries.
    """

    target = _normalize_ion_mode(ion_mode)
    mask = np.ones(len(entropy_search.precursor_mz_array), dtype=bool)
    if target in {None, "", "unknown", "none"}:
        return mask

    for i in range(len(entropy_search.precursor_mz_array)):
        ms2 = _normalize_record(entropy_search[i])
        db_ion_mode = _normalize_ion_mode(ms2.get('ion_mode'))
        if db_ion_mode is not None:
            mask[i] = db_ion_mode == target

    return mask


def _build_retention_time_array(entropy_search):
    """
    Return database retention times, using inf where RT is missing.
    """

    rt_arr = np.zeros(len(entropy_search.precursor_mz_array)) + np.inf
    for i, ms2 in enumerate(entropy_search):
        ms2 = _normalize_record(ms2)
        if 'retention_time' in ms2:
            rt_arr[i] = _safe_float(ms2['retention_time'], default=np.inf)
    return rt_arr


def _clean_ms2_signals(entropy_search, precursor_mz, signals, precursor_mz_offset):
    """
    Clean a spectrum for search without mutating the original Scan object.
    """

    if signals is None or len(signals) == 0:
        return np.empty((0, 2), dtype=np.float32)

    cleaned = entropy_search.clean_spectrum_for_search(
        precursor_mz=precursor_mz,
        peaks=signals,
        precursor_ions_removal_da=precursor_mz_offset,
    )
    if cleaned is None:
        return np.empty((0, 2), dtype=np.float32)
    return cleaned


def _normalize_column_name(name):
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("/", "")


def _find_optional_column(df, candidates):
    normalized = {_normalize_column_name(col): col for col in df.columns}
    for candidate in candidates:
        col = normalized.get(_normalize_column_name(candidate))
        if col is not None:
            return col
    return None


def _resolve_mzrt_columns(df):
    name_col = _find_optional_column(df, ["annotation", "name", "compound", "compound_name"])
    mz_col = _find_optional_column(df, ["mz", "m/z", "precursor_mz", "precursormz", "matched_mz"])
    rt_col = _find_optional_column(df, ["rt", "retention_time", "retentiontime"])

    if name_col is None and df.shape[1] > 0:
        name_col = df.columns[0]
    if mz_col is None and df.shape[1] > 1:
        mz_col = df.columns[1]
    if rt_col is None and df.shape[1] > 2:
        rt_col = df.columns[2]

    missing = [
        label for label, col in [("annotation/name", name_col), ("m/z", mz_col), ("RT", rt_col)]
        if col is None
    ]
    if len(missing) > 0:
        raise ValueError("mzRT file is missing required columns: {}".format(", ".join(missing)))

    return name_col, mz_col, rt_col


def _row_value(row, col, default=None):
    if col is None:
        return default
    value = row[col]
    return default if _is_missing_value(value) else value


def _format_text(value, default=""):
    if _is_missing_value(value):
        return default
    return str(value)


def index_json_to_pkl(json_path, output_path=None):
    """
    A function to index JSON file to pickle format.

    Parameters
    ----------
    json_path : str
        The path to the JSON file.
    output_path : str
        The path to the output pickle file.
    """

    file_name = os.path.basename(json_path).split(".")[0]

    if output_path is None:
        output_path = os.path.dirname(json_path)

    with open(json_path, 'r') as f:
        db = json.load(f)
    db = _preprocess_msp_list(db)
    entropy_search = FlashEntropySearch()
    entropy_search.build_index(db)

    with open(os.path.join(output_path, file_name + ".pkl"), 'wb') as f:
        pickle.dump(entropy_search, f)


def _preprocess_msp_list(db: list):
    """
    Preprocess the MSP format MS/MS database.
    """

    processed = []
    for item in db:
        a = _normalize_record(item)

        if 'precursortype' in a and 'precursor_type' not in a:
            a['precursor_type'] = a.pop('precursortype')

        if 'precursor_mz' not in a:
            mz_keys = [k for k in a.keys() if 'prec' in k and 'mz' in k]
            a['precursor_mz'] = a.pop(mz_keys[0]) if len(mz_keys) > 0 else None
        a['precursor_mz'] = _safe_float(a.get('precursor_mz'))

        if 'ionmode' in a and 'ion_mode' not in a:
            a['ion_mode'] = a.pop('ionmode')

        if 'retentiontime' in a and 'retention_time' not in a:
            a['retention_time'] = a.pop('retentiontime')
        if 'retention_time' in a:
            rt = _safe_float(a.get('retention_time'))
            if np.isfinite(rt):
                a['retention_time'] = rt
            else:
                a.pop('retention_time', None)

        if not np.isfinite(a['precursor_mz']) or 'peaks' not in a or len(a['peaks']) == 0:
            continue
        processed.append(a)

    return processed


def _read_msp(path: str):
    """
    A helper function to read MSP file and return a list of dictionaries.

    Parameters
    ----------
    path : str
        The path to the MSP file.

    Returns
    -------
    entropy_search : FlashEntropySearch object
         The FlashEntropySearch object built from the MSP file.
    """
    db = _preprocess_msp_list([a for a in read_one_spectrum(path)])
    if len(db) == 0:
        raise ValueError(f"No valid spectra were found in MSP file: {path}")
    entropy_search = FlashEntropySearch(intensity_weight=None)
    entropy_search.build_index(db)
    return entropy_search


def _read_pickle(path: str):
    """
    A helper function to read pickle file and return a FlashEntropySearch object.

    Parameters
    ----------
    path : str
        The path to the pickle file.

    Returns
    -------
    entropy_search : FlashEntropySearch object
         The FlashEntropySearch object built from the pickle file.
    """

    with open(path, 'rb') as f:
        entropy_search = pickle.load(f)

    if not hasattr(entropy_search, 'precursor_mz_array'):
        raise ValueError(f"Invalid MS/MS database pickle: {path}")

    # check if intensity_weight is an attribute
    if hasattr(entropy_search, 'entropy_search') and not hasattr(entropy_search.entropy_search, 'intensity_weight'):
        raise ValueError("Please download the newest MS/MS database from: https://zenodo.org/records/14991522.")
    return entropy_search


def _read_json(path: str):
    """
    A helper function to read json file and return a FlashEntropySearch object.

    Parameters
    ----------
    path : str
        The path to the json file.

    Returns
    -------
    entropy_search : FlashEntropySearch object
         The FlashEntropySearch object built from the json file.
    """

    with open(path, 'r') as f:
        db = json.load(f)
    db = _preprocess_msp_list(db)
    if len(db) == 0:
        raise ValueError(f"No valid spectra were found in JSON file: {path}")
    entropy_search = FlashEntropySearch(intensity_weight=None)
    entropy_search.build_index(db)
    return entropy_search


def _assign_annotation_results_to_feature(f, score, matched, matched_peak_num, search_mode,
                                          ms2_scan_idx=None, precursor_ion_fraction=None):
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
    f.annotation = matched.get('name')
    f.formula = matched.get('formula')
    f.matched_peak_number = matched_peak_num
    f.smiles = matched.get('smiles')
    f.inchikey = matched.get('inchikey')
    f.matched_ms2 = convert_signals_to_string(matched.get('peaks'))
    f.matched_precursor_mz = matched.get('precursor_mz')
    f.matched_adduct_type = matched.get('precursor_type')
    if search_mode.startswith('identity_search'):
        f.adduct_type = matched.get('precursor_type')
    f.ms2_pif = precursor_ion_fraction
    f.ms2_scan_idx = ms2_scan_idx
    f.database = matched.get('database')
    if hasattr(f, 'matched_spectra'):
        f.matched_spectra.append(matched)


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
