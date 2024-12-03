# Utility functions for feature table manipulation
import pandas as pd
import re

def output_feature_to_msp(feature_table, output_path):
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
            f.write("ID: " + str(feature_table['feature_ID'][i]) + "\n")
            if feature_table['MS2'][i] is None or feature_table['MS2'][i]!=feature_table['MS2'][i]:
                f.write("NAME: Unknown\n")
                f.write("PRECURSORMZ: " + str(feature_table['m/z'][i]) + "\n")
                f.write("PRECURSORTYPE: " + str(feature_table['adduct'][i]) + "\n")
                f.write("RETENTIONTIME: " + str(feature_table['RT'][i]) + "\n")
                f.write("Num Peaks: " + "0\n")
                f.write("\n")
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


def convert_features_to_df(features, sample_names, quant_method="peak_height"):
    """
    convert feature list to DataFrame

    Parameters
    ----------
    features : list
        list of features
    sample_names : list
        list of sample names
    quant_method : str
        quantification method, "peak_height", "peak_area" or "top_average"

    Returns
    -------
    feature_table : pd.DataFrame
        feature DataFrame
    """

    results = []
    sample_names = list(sample_names)
    columns=["group_ID", "feature_ID", "m/z", "RT", "adduct", "is_isotope", "is_in_source_fragment", "Gaussian_similarity", "noise_score", 
             "asymmetry_factor", "detection_rate", "detection_rate_gap_filled", "alignment_reference_file", "charge", "isotopes", "MS2_reference_file", "MS2", "matched_MS2", 
             "search_mode", "annotation", "formula", "similarity", "matched_peak_number", "SMILES", "InChIKey"] + sample_names

    for f in features:
        if quant_method == "peak_height":
            quant = list(f.peak_height_arr)
        elif quant_method == "peak_area":
            quant = list(f.peak_area_arr)
        elif quant_method == "top_average":
            quant = list(f.top_average_arr)
        
        results.append([f.feature_group_id, f.id, f.mz, f.rt, f.adduct_type, f.is_isotope, f.is_in_source_fragment, f.gaussian_similarity, f.noise_score,
                        f.asymmetry_factor, f.detection_rate, f.detection_rate_gap_filled, f.reference_file, f.charge_state, f.isotope_signals, f.ms2_reference_file,
                        f.ms2, f.matched_ms2, f.search_mode, f.annotation, f.formula, f.similarity, f.matched_peak_number, f.smiles, f.inchikey] + quant)
        
    feature_table = pd.DataFrame(results, columns=columns)
    
    return feature_table