# Author: Hauxu Yu

# A module to summarize the premade data processing workflows.

# Import modules
import os
import multiprocessing
import pickle
from copy import deepcopy
import pandas as pd
import numpy as np
from importlib.metadata import version
from scipy.stats import zscore
import time
import json

from .raw_data_utils import read_raw_file_to_obj
from .params import Params
from .feature_grouping import annotate_isotope, annotate_adduct, annotate_in_source_fragment
from .alignment import feature_alignment, gap_filling, output_feature_table
from .annotation import annotate_aligned_features, annotate_features, feature_annotation_mzrt
from .normalization import sample_normalization, signal_normalization
from .visualization import plot_ms2_matching_from_feature_table
from .stats import statistical_analysis
from .feature_table_utils import convert_features_to_df, output_feature_to_msp
from .utils_functions import get_timestamps


# 1. Untargeted feature detection for a single file
def process_single_file(file_name, params=None, segment_feature=True, group_features=False, evaluate_peak_shape=False,
                        annotate_ms2=False, ms2_library_path=None, output_dir=None):
    """
    Untargeted data processing for a single file (mzML, mzXML, mzjson or compressed mzjson).

    Parameters
    ----------
    file_name : str
        Path to the raw file.
    params : Params object
        Parameters for feature detection. If None, the default parameters are used
        based on the type of mass spectrometer.
    segment_feature : bool
        Whether to segment the feature to peaks for distinguishing possible isomers. Default is True.
    group_features : bool
        Whether to group features by isotopes, adducts and in-source fragments. Default is True.
    evaluate_peak_shape : bool
        Whether to evaluate the peak shape by calculating noise score and asymmetry factor. Default is True.
    annotate_ms2 : bool
        Whether to annotate MS2 spectra. Default is False.
    ms2_library_path : str
        Another way to specify the MS2 library path.
    output_dir : str
        The output directory for the single file. If None, the output is saved to the same directory as the raw file.

    Returns
    -------
    d : MSData object
        An MSData object containing the processed data.
    """

    try:
        # STEP 1. data reading, parsing, and parameter preparation
        d = read_raw_file_to_obj(file_name, params=params)
        # check if the file is centroided
        if not d.params.is_centroid:
            print("File: " + file_name + " is not centroided and skipped.")
            return None
        # set ms2 library path
        if ms2_library_path is not None:
            d.params.ms2_library_path = ms2_library_path

        # STEP 2. feature detection and segmentation
        d.detect_features()
        if segment_feature:
            d.segment_features()

        # STEP 3. feature evaluation
        if evaluate_peak_shape:
            d.summarize_features(cal_g_score=True, cal_a_score=True)
        else:
            d.summarize_features(cal_g_score=False, cal_a_score=False)

        # STEP4. feature grouping
        if group_features:
            annotate_isotope(d)
            annotate_in_source_fragment(d)
            annotate_adduct(d)

        # STEP 5. MS2 annotation
        if annotate_ms2:
            if ms2_library_path is None:
                ms2_library_path = d.params.ms2_library_path
            if ms2_library_path is not None:
                annotate_features(d=d, sim_tol=d.params.ms2_sim_tol, fuzzy_search=True, ms2_library_path=ms2_library_path)

        # STEP 6. Visualization and output
        if d.params.plot_bpc and d.params.bpc_dir is not None:
            d.plot_bpc(output_dir=os.path.join(d.params.bpc_dir, d.params.file_name + "_bpc.png"))
        if d.params.output_single_file:
            if output_dir is not None:
                d.output_single_file(os.path.join(output_dir, d.params.file_name + ".txt"))
            elif d.params.single_file_dir is not None:
                d.output_single_file()
            
        # for faster data reloading
        if d.params.output_ms1_scans:
            d.output_ms1_to_pickle()

        return d
    
    except:
        print("Error: " + file_name + " is skipped.")
        return None


# 2. Untargeted metabolomics workflow
def untargeted_metabolomics_workflow(path=None, return_results=False):
    """
    The untargeted metabolomics workflow. See the documentation for details.

    Parameters
    ----------
    path : str
        The working directory. If None, the current working directory is used.
    return_results : bool
        Whether to return the results. Default is False.

    Returns
    -------
    features : list
        A list of features.
    params : Params object
        Parameters for the workflow.
    """

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Welcome to the untargeted metabolomics workflow.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # STEP 1. Prepare parameters for the workflow
    print("Step 1: Preparing the workflow...")
    
    metadata = deepcopy(DATA_PROCESSING_METADATA)
    metadata[0]['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    metadata[0]['dependencies'] = {item: version(item) for item in DEPENDENCIES}
    params = Params()
   
    # obtain the working directory
    if path is not None:
        params.project_dir = path
    else:
        params.project_dir = os.getcwd()
    
    params._untargeted_metabolomics_workflow_preparation()

    # save the parameters to metadata
    for key, value in params.__dict__.items():
        metadata[1][key] = value
    print("\tWorkflow is prepared.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # STEP 2. Process individual files
    print("Step 2: Processing individual files for feature detection...")
    processed_files = [f.split(".")[0] for f in os.listdir(params.single_file_dir) if f.lower().endswith(".txt")]
    to_be_processed = []
    for i, f in enumerate(params.sample_names):
        if f not in processed_files:
            to_be_processed.append(params.sample_abs_paths[i])
    print("\t{} files to process out of {} files.".format(len(to_be_processed), len(params.sample_abs_paths)))
    
    workers = int(multiprocessing.cpu_count() * params.percent_cpu_to_use)
    print("\tA total of {} CPU cores are detected, {} cores are used.".format(multiprocessing.cpu_count(), workers))
    for i in range(0, len(to_be_processed), params.batch_size):
        if len(to_be_processed) - i < params.batch_size:
            print("\tProcessing files from " + str(i) + " to " + str(len(to_be_processed)))
        else:
            print("\tProcessing files from " + str(i) + " to " + str(i+to_be_processed))
        p = multiprocessing.Pool(workers)
        p.starmap(process_single_file, [(f, params) for f in to_be_processed[i:i+params.batch_size]])
        p.close()
        p.join()
        
    metadata[2]["status"] = "completed"
    print("\tIndividual file processing is completed.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # STEP 3. Feature alignment
    print("Step 3: Aligning features...")
    features = feature_alignment(params.single_file_dir, params)
    metadata[3]["status"] = "completed"
    print("\tFeature alignment is completed.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    # STEP 4. Feature annotation
    print("Step 4: Annotating features...")
    # annotation (using MS2 library)
    if params.ms2_library_path is not None and os.path.exists(params.ms2_library_path):
        features = annotate_aligned_features(features, params)
        print("\tMS2 annotation is completed.")
    else:
        print("\tNo MS2 library is found. MS2 annotation is skipped.")
    # annotation (using mzrt list)
    if os.path.exists(os.path.join(params.project_dir, "mzrt_list.csv")):
        print("\tAnnotating features using the extra mzrt list...")
        default_adduct = "[M+H]+" if params.ion_mode == "positive" else "[M-H]-"
        features = feature_annotation_mzrt(features, os.path.join(params.project_dir, "mzrt_list.csv"), default_adduct, params.align_mz_tol, params.align_rt_tol)
        print("\tmz/rt annotation is completed.")
    metadata[4]["status"] = "completed"

    feature_table = convert_features_to_df(features=features, sample_names=params.sample_names, quant_method=params.quant_method)
    # output feature table to a txt file
    output_path = os.path.join(params.project_dir, "aligned_feature_table.txt")
    output_feature_table(feature_table, output_path)
    # output the acquired MS2 spectra to a MSP file (designed for MassWiki)
    output_path = os.path.join(params.project_file_dir, "features.msp")
    output_feature_to_msp(feature_table, output_path)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # STEP 5. signal normalization
    if params.signal_normalization:
        print("Step 5: Running signal normalization...")
        feature_table_before_normalization = deepcopy(feature_table)
        sample_names = params.sample_names
        df_time = get_timestamps(params.project_dir, output=False)
        time_by_name = df_time.iloc[:,0].values
        analytical_order = [np.where(time_by_name == name)[0][0] for name in sample_names]
        feature_table = signal_normalization(feature_table, params.individual_sample_groups, analytical_order, params.signal_norm_method)
        metadata[5]["status"] = "completed"
        print("\tMS signal drift normalization is completed.")
    else:
        metadata[5]["status"] = "skipped"
        print("Step 6: MS signal drift normalization is skipped.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # # STEP 6. sample normalization
    # if params.sample_normalization:
    #     print("Step 6: Running sample normalization...")
    #     feature_table = sample_normalization(feature_table, params.individual_sample_groups, params.sample_norm_method)
    #     metadata[6]["status"] = "completed"
    #     print("\tSample Normalization is completed.")
    # else:
    #     metadata[6]["status"] = "skipped"
    #     print("Step 6: Sample normalization is skipped.")

    # # STEP 7. statistical analysis
    # if params.run_statistics:
    #     print("Step 7: Running statistical analysis...")
    #     feature_table_before_normalization = statistical_analysis(feature_table_before_normalization, params, before_norm=True)
    #     feature_table = statistical_analysis(feature_table, params)
    #     metadata[7]["status"] = "completed"
    #     print("\tStatistical analysis is completed.")
    # else:
    #     metadata[7]["status"] = "skipped"
    #     print("Step 7: Statistical analysis is skipped.")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    # STEP 8. output and visualization
    metadata[0]['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    time_label = time.strftime("%Y%m%d%H%M%S", time.localtime())
    metadata_file_name = "data_processing_metadata_" + time_label + ".pkl"
    with open(os.path.join(params.project_file_dir, metadata_file_name), "wb") as f:
        pickle.dump(metadata, f)
    with open(os.path.join(params.project_file_dir, "project.masscube"), "wb") as f:
        pickle.dump(params, f)

    # plot annoatated metabolites
    if params.plot_ms2:
        print("Plotting MS2 matching...")
        plot_ms2_matching_from_feature_table(feature_table, params)
        print("\tMS2 plotting is completed.")

    print("The workflow is completed.")

    if return_results:
        return features, params


# 4. Evaluate the data quality of the raw files
def run_evaluation(path=None):
    """
    Evaluate the run and report the problematic files.

    Parameters
    ----------
    path : str
        Path to the project directory.
    """

    if path is None:
        path = os.getcwd()

    # check if sample table exists
    if os.path.exists(os.path.join(path, "sample_table.csv")):
        sample_table = pd.read_csv(os.path.join(path, "sample_table.csv"))
        blank_samples = sample_table.iloc[:,0].values[sample_table.iloc[:,1] == 'blank'].tolist()
    else:
        print("Sample table is not found. Problematic files may include blank samples.")
        blank_samples = []

    # get all .txt files
    txt_path = os.path.join(path, "single_files")
    txt_files = [f for f in os.listdir(txt_path) if f.lower().endswith('.txt')]
    txt_files = [f for f in txt_files if not f.startswith(".")]
    int_array = np.zeros(len(txt_files))
    for i in range(len(txt_files)):
        df = pd.read_csv(os.path.join(txt_path, txt_files[i]), sep="\t", low_memory=False)
        int_array[i] = np.sum(df['peak_height'].values)
        
    z = zscore(int_array)
    idx = np.where(z < -2)[0]

    problematic_files = []
    for i in idx:
        problematic_files.append(txt_files[i].split(".")[0])

    problematic_files = [f for f in problematic_files if f not in blank_samples]
    
    # output the names of problematic files
    if len(problematic_files) > 0:
        print("The following files are problematic:")
        for f in problematic_files:
            print(f)
        # output to a txt file
        df = pd.DataFrame(problematic_files, columns=["file_name"])
        output_path = os.path.join(path, "problematic_files.txt")
        df.to_csv(output_path, sep="\t", index=False)
    else:
        print("No problematic files are found.")


# 5. Single-file peak picking (batch mode)
def batch_file_processing(path=None, batch_size=100, cpu_ratio=0.8):
    """
    The untargeted metabolomics workflow. See the documentation for details.

    Parameters
    ----------
    path : str
        The working directory. If None, the current working directory is used.
    batch_size : int
        The number of files to be processed in each batch.
    cpu_ratio : float
        The ratio of CPU cores to be used.
    """

    params = Params()
    # obtain the working directory
    if path is not None:
        params.project_dir = path
    else:
        params.project_dir = os.getcwd()
    params._batch_processing_preparation()
    
    raw_file_names = os.listdir(params.sample_dir)
    raw_file_names = [f for f in raw_file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")
                      or f.lower().endswith(".mzjson.gz") or f.lower().endswith(".mzjson")]
    raw_file_names = [f for f in raw_file_names if not f.startswith(".")]   # for Mac OS
    
    # skip the files that have been processed
    txt_files = os.listdir(params.single_file_dir)
    txt_files = [f.split(".")[0] for f in txt_files if f.lower().endswith(".txt")]
    txt_files = [f for f in txt_files if not f.startswith(".")]  # for Mac OS
    raw_file_names = [f for f in raw_file_names if f.split(".")[0] not in txt_files]
    
    raw_file_names = [os.path.join(params.sample_dir, f) for f in raw_file_names]
    print("Total number of files to be processed: " + str(len(raw_file_names)))

    # process files by multiprocessing, each batch contains 100 files by default (tunable in batch_size)
    print("Processing files by multiprocessing...")

    workers = int(multiprocessing.cpu_count() * cpu_ratio)
    for i in range(0, len(raw_file_names), batch_size):
        if len(raw_file_names) - i < batch_size:
            print("Processing files from " + str(i) + " to " + str(len(raw_file_names)))
        else:
            print("Processing files from " + str(i) + " to " + str(i+batch_size))
        p = multiprocessing.Pool(workers)
        p.starmap(process_single_file, [(f, params, True, True, False, False, False ) for f in raw_file_names[i:i+batch_size]])
        p.close()
        p.join()


DEPENDENCIES = ('masscube', 'numpy', 'pandas', 'scipy', 'matplotlib', 'pyteomics', 'scikit-learn', 'ms_entropy', 'lxml')

DATA_PROCESSING_METADATA = [
    {
        "name": "overview",
        "layer": 0,
        "dependencies": None,
        "start_time": None,
        "end_time": None,
    },
    {
        "name": "parameters",
        "layer": 1,
    },
    {
        "name": "feature_detection",
        "layer": 2,
        "status": "not completed",
    },
    {
        "name": "feature_alignment",
        "layer": 3,
        "status": "not completed"
    },
    {
        "name": "feature_annotation",
        "layer": 4,
        "status": "not completed",
    },
    {
        "name": "signal_normalization",
        "layer": 5,
        "status": "not completed"
    },
    {
        "name": "sample_normalization",
        "layer": 6,
        "status": "not completed"
    },
    {
        "name": "statistical_analysis",
        "layer": 7,
        "status": "not completed"
    }
]