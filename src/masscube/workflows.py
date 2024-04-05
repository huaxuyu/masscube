# Author: Hauxu Yu

# A module to summarize the premade data processing workflows.

# Import modules
import os
import multiprocessing
import pickle
from copy import deepcopy

from .raw_data_utils import MSData
from .params import Params, find_ms_info
from .feature_grouping import annotate_isotope, annotate_adduct, annotate_in_source_fragment
from .alignment import feature_alignment, gap_filling
from .annotation import feature_annotation, annotate_rois
from .normalization import sample_normalization
from .visualization import plot_ms2_matching_from_feature_table
from .network import network_analysis
from .stats import statistical_analysis
from .feature_table_utils import calculate_fill_percentage


# Untargeted feature detection for a single file
def feature_detection(file_name, params=None, cal_gss=True, annotation=False):
    """
    Untargeted feature detection from a single file (.mzML or .mzXML).

    Parameters
    ----------
    file_name : str
        Path to the raw file.
    parameters : Params object
        The parameters for the workflow.
    annotation : bool
        Whether to annotate the MS2 spectra.

    Returns
    -------
    d : MSData object
        The MSData object with the detected features. The detected features are stored in d.rois.
    """

    # create a MSData object
    d = MSData()

    # set parameters
    # if params is None, use the default parameters
    if params is None:
        params = Params()
        ms_type, ion_mode = find_ms_info(file_name)
        print("MS type: " + ms_type, "Ion mode: " + ion_mode)
        params.set_default(ms_type, ion_mode)

    # read raw data
    d.read_raw_data(file_name, params)

    # drop ions by intensity (defined in params.int_tol)
    d.drop_ion_by_int()
    # detect region of interests (ROIs)
    d.find_rois()

    # cut ROIs
    d.cut_rois()

    # label short ROIs, find the best MS2, and sort ROIs by m/z
    d.summarize_roi(cal_gss=cal_gss)

    print("Number of detected features: " + str(len(d.rois)))

    # # annotate isotopes, adducts, and in-source fragments
    annotate_isotope(d)
    annotate_in_source_fragment(d)
    annotate_adduct(d)

    # annotate MS2 spectra
    if annotation and d.params.msms_library is not None:
        annotate_rois(d)

    if params.plot_bpc:
        d.plot_bpc(label_name=True, output=os.path.join(params.bpc_dir, d.file_name + "_bpc.png"))

    # output single file to a csv file
    if d.params.output_single_file:
        d.output_single_file()

    return d


# Untargeted metabolomics workflow
def untargeted_metabolomics_workflow(path=None):
    """
    A function for the untargeted metabolomics workflow.
    """
    params = Params()
    # obtain the working directory
    if path is not None:
        params.project_dir = path
    else:
        params.project_dir = os.getcwd()
    params._untargeted_metabolomics_workflow_preparation()

    with open(os.path.join(params.project_dir, "project.mc"), "wb") as f:
        pickle.dump(params, f)
    
    raw_file_names = os.listdir(params.sample_dir)
    raw_file_names = [f for f in raw_file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
    raw_file_names = [os.path.join(params.sample_dir, f) for f in raw_file_names]

    # process files by multiprocessing, each batch contains 300 files
    print("Processing files by multiprocessing...")
    workers = int(multiprocessing.cpu_count() * 0.8)
    for i in range(0, len(raw_file_names), 300):
        if len(raw_file_names) - i < 300:
            print("Processing files from " + str(i) + " to " + str(len(raw_file_names)))
        else:
            print("Processing files from " + str(i) + " to " + str(i+300))
        p = multiprocessing.Pool(workers)
        p.starmap(feature_detection, [(f, params) for f in raw_file_names[i:i+300]])
        p.close()
        p.join()

    # feature alignment
    print("Aligning features...")
    feature_table = feature_alignment(params.single_file_dir, params)

    # gap filling
    print("Filling gaps...")
    feature_table = gap_filling(feature_table, params)
    
    # calculate fill percentage
    feature_table = calculate_fill_percentage(feature_table, params.individual_sample_groups)

    # annotation
    print("Annotating features...")
    if params.msms_library is not None and os.path.exists(params.msms_library):
        feature_annotation(feature_table, params)
    else:
        print("No MS2 library is found. Skipping annotation...")

    # normalization
    if params.run_normalization:
        feature_table.to_csv(os.path.join(params.project_dir, "aligned_feature_table_before_normalization.csv"), index=False)
        feature_table_before_normalization = deepcopy(feature_table)
        print("Running normalization...")
        feature_table = sample_normalization(feature_table, params.individual_sample_groups, params.normalization_method)

    # statistical analysis
    if params.run_statistics:
        print("Running statistical analysis...")
        feature_table_before_normalization = statistical_analysis(feature_table_before_normalization, params, before_norm=True)
        feature_table = statistical_analysis(feature_table, params)

    # network analysis
    if params.run_network:
        print("Running network analysis...This may take several minutes...")
        network_analysis(feature_table)

    # plot annoatated metabolites
    if params.plot_ms2:
        print("Plotting annotated metabolites...")
        plot_ms2_matching_from_feature_table(feature_table, params)
    
    # output feature table
    feature_table.to_csv(os.path.join(params.project_dir, "aligned_feature_table.csv"), index=False)
    print("The workflow is completed.")