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
from .annotation import feature_annotation, annotate_rois, feature_annotation_mzrt
from .normalization import sample_normalization
from .visualization import plot_ms2_matching_from_feature_table
from .stats import statistical_analysis
from .feature_table_utils import convert_features_to_df, output_feature_to_msp


# 1. Untargeted feature detection for a single file
def feature_detection(file_name, params=None, cal_g_score=True, cal_a_score=True,
                      anno_isotope=True, anno_adduct=True, anno_in_source_fragment=True, 
                      annotation=False, ms2_library_path=None, output_dir=None, cut_roi=True):
    """
    Untargeted feature detection from a single file (.mzML or .mzXML).

    Parameters
    ----------
    file_name : str
        Path to the raw file.
    params : Params object
        Parameters for feature detection.
    cal_gss : bool
        Whether to calculate the Gaussian similarity score (GSS) for each ROI.
    anno_isotope : bool
        Whether to annotate isotopes.
    anno_adduct : bool
        Whether to annotate adducts.
    anno_in_source_fragment : bool
        Whether to annotate in-source fragments.
    annotation : bool
        Whether to annotate MS2 spectra. If True, the MS2 library should be provided in the params object.

    Returns
    -------
    d : MSData object
        An MSData object containing the processed data.
    """

    try:
        # create a MSData object
        d = read_raw_file_to_obj(file_name, params=params, centroid_mz=True, read_ms2=True, clean_ms2=True)

        if not d.centroid:
            print("File: " + file_name + " is not centroided and skipped.")
            return None
        
        if ms2_library_path is not None:
            d.params.msms_library = ms2_library_path

        # detect region of interests (ROIs)
        d.find_rois()

        # cut ROIs
        if cut_roi:
            d.cut_rois()

        # label short ROIs, find the best MS2, and sort ROIs by m/z
        d.summarize_roi(cal_g_score=cal_g_score, cal_a_score=cal_a_score)

        # # annotate isotopes, adducts, and in-source fragments
        if anno_isotope:
            annotate_isotope(d)
        if anno_in_source_fragment:
            annotate_in_source_fragment(d)
        if anno_adduct:
            annotate_adduct(d)

        # annotate MS2 spectra
        if annotation and d.params.msms_library is not None:
            annotate_rois(d)

        if d.params.plot_bpc:
            d.plot_bpc(label_name=True, output_dir=os.path.join(params.bpc_dir, d.file_name + "_bpc.png"))

        # output single file to a txt file
        if d.params.output_single_file:
            d.output_single_file()
        elif output_dir is not None:
            d.output_single_file(os.path.join(output_dir, d.file_name + ".txt"))

        return d
    
    except Exception as e:
        print("Error: " + str(e))
        return None


# 2. Untargeted metabolomics workflow
def untargeted_metabolomics_workflow(path=None, batch_size=100, cpu_ratio=0.8):
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

    # start of the workflow
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Welcome to the untargeted metabolomics workflow.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # define the metadata and parameters
    medadata = [
        {
            "name": "overview",
            "layer": 0,
            "packages": [
                {"name": "masscube", "version": version("masscube")},
                {"name": "numpy", "version": version("numpy")},
                {"name": "pandas", "version": version("pandas")},
                {"name": "scipy", "version": version("scipy")},
                {"name": "matplotlib", "version": version("matplotlib")},
                {"name": "pyteomics", "version": version("pyteomics")},
                {"name": "scikit-learn", "version": version("scikit-learn")},
                {"name": "ms_entropy", "version": version("ms_entropy")},
                {"name": "lxml", "version": version("lxml")}
            ],
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
        }
    ]

    print("Step 1: Preparing the workflow...")
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
    raw_file_names = [f for f in raw_file_names if not f.startswith(".")]   # for Mac OS
    total_file_num = len(raw_file_names)
    # skip the files that have been processed
    txt_files = os.listdir(params.single_file_dir)
    txt_files = [f.split(".")[0] for f in txt_files if f.lower().endswith(".txt")]
    txt_files = [f for f in txt_files if not f.startswith(".")]  # for Mac OS
    raw_file_names = [f for f in raw_file_names if f.split(".")[0] not in txt_files]
    raw_file_names = [os.path.join(params.sample_dir, f) for f in raw_file_names]

    print("\t{} raw file are found, {} files to be processed.".format(total_file_num, len(raw_file_names)))

    # save the parameters to metadata
    p_meta = {
        "name": "parameters",
        "layer": 1,
    }
    for key, value in params.__dict__.items():
        p_meta[key] = value
    medadata.append(p_meta)
    print("\tWorkflow is prepared.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # process files by multiprocessing, each batch contains 100 files by default (tunable in batch_size)
    print("Step 2: Processing individual files for feature detection, evaluation, and grouping...")
    workers = int(multiprocessing.cpu_count() * cpu_ratio)
    print("\tA total of {} CPU cores are detected, {} cores are used.".format(multiprocessing.cpu_count(), workers))
    for i in range(0, len(raw_file_names), batch_size):
        if len(raw_file_names) - i < batch_size:
            print("Processing files from " + str(i) + " to " + str(len(raw_file_names)))
        else:
            print("Processing files from " + str(i) + " to " + str(i+batch_size))
        p = multiprocessing.Pool(workers)
        p.starmap(feature_detection, [(f, params) for f in raw_file_names[i:i+batch_size]])
        p.close()
        p.join()
    medadata.append({
        "name": "feature_detection",
        "layer": 2,
        "core_num": workers,
        "batch_size": batch_size,
        "status": "completed"
    })
    print("\tIndividual file processing is completed.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    if not os.path.exists(os.path.join(params.project_dir, "aligned_feature_table_before_normalization.txt")) and not os.path.exists(os.path.join(params.project_dir, "aligned_feature_table.txt")):
        # feature alignment
        print("Step 3: Aligning features...")
        features = feature_alignment(params.single_file_dir, params)
        medadata.append({
            "name": "feature_alignment",
            "layer": 3,
            "status": "completed"
        })
        print("\tFeature alignment is completed.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # gap filling
        print("Step 4: Filling gaps...")
        features = gap_filling(features, params)
        medadata.append({
            "name": "gap_filling",
            "layer": 4,
            "status": "completed"
        })
        print("\tGap filling is completed.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # annotation (using MS2 library)
        print("Step 5: Annotating features...")
        ms2_anno = False
        mzrt_anno = False
        if params.msms_library is not None and os.path.exists(params.msms_library):
            features = feature_annotation(features, params)
            print("\tMS2 annotation is completed.")
            ms2_anno = True
        else:
            print("\tNo MS2 library is found. MS2 annotation is skipped.")
        
        # annotation (using mzrt list)
        if os.path.exists(os.path.join(params.project_dir, "mzrt_list.csv")):
            print("\tAnnotating features using the extra mzrt list...")
            mzrt_anno = True
            default_adduct = "[M+H]+" if params.ion_mode == "positive" else "[M-H]-"
            features = feature_annotation_mzrt(features, os.path.join(params.project_dir, "mzrt_list.csv"), default_adduct, params.align_mz_tol, params.align_rt_tol)
            print("\tmz/rt annotation is completed.")
        medadata.append({
            "name": "feature_annotation",
            "layer": 5,
            "status": "completed",
            "ms2_annotation": "applied" if ms2_anno else "skipped",
            "mzrt_annotation": "applied" if mzrt_anno else "skipped"
        })

        feature_table = convert_features_to_df(features, params.sample_names)
        # output the acquired MS2 spectra to a MSP file (designed for MassWiki)
        output_path = os.path.join(params.project_dir, "MassWiki_input.msp")
        output_feature_to_msp(feature_table, output_path)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    else:
        print("The aligned feature table is found. Step 3 (feature alignment), Step 4 (gap filling), and Step 5 (annotation) are skipped.")
        if os.path.exists(os.path.join(params.project_dir, "aligned_feature_table.txt")):
            feature_table = pd.read_csv(os.path.join(params.project_dir, "aligned_feature_table.txt"), sep="\t", low_memory=False)
        elif os.path.exists(os.path.join(params.project_dir, "aligned_feature_table_before_normalization.txt")):
            feature_table = pd.read_csv(os.path.join(params.project_dir, "aligned_feature_table_before_normalization.txt"), sep="\t", low_memory=False)
        medadata.append({
            "name": "feature_alignment",
            "layer": 3,
            "status": "use previous result"
        })
        medadata.append({
            "name": "gap_filling",
            "layer": 4,
            "status": "use previous result"
        })
        medadata.append({
            "name": "feature_annotation",
            "layer": 5,
            "status": "use previous result",
            "ms2_annotation": "use previous result",
            "mzrt_annotation": "use previous result"
        })
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # normalization
    if params.run_normalization:
        print("Step 6: Running normalization...")
        output_path = os.path.join(params.project_dir, "aligned_feature_table_before_normalization.txt")
        output_feature_table(feature_table, output_path)
        feature_table_before_normalization = deepcopy(feature_table)
        feature_table = sample_normalization(feature_table, params.individual_sample_groups, params.normalization_method)
        medadata.append({
            "name": "normalization",
            "layer": 6,
            "status": "completed"
        })
        print("\tNormalization is completed.")
    else:
        medadata.append({
            "name": "normalization",
            "layer": 6,
            "status": "skipped"
        })
        print("Step 6: Normalization is skipped.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # statistical analysis
    if params.run_statistics:
        print("Step 7: Running statistical analysis...")
        feature_table_before_normalization = statistical_analysis(feature_table_before_normalization, params, before_norm=True)
        feature_table = statistical_analysis(feature_table, params)
        medadata.append({
            "name": "statistical_analysis",
            "layer": 7,
            "status": "completed"
        })
        print("\tStatistical analysis is completed.")
    else:
        medadata.append({
            "name": "statistical_analysis",
            "layer": 7,
            "status": "skipped"
        })
        print("Step 7: Statistical analysis is skipped.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    # output feature table
    output_path = os.path.join(params.project_dir, "aligned_feature_table.txt")
    output_feature_table(feature_table, output_path)

    # output parameters and metadata
    medadata[0]['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    time_label = time.strftime("%Y%m%d%H%M%S", time.localtime())
    metadata_file_name = "data_processing_metadata_" + time_label + ".json"
    with open(os.path.join(params.project_dir, metadata_file_name), "w") as f:
        json.dump(medadata, f)

    print("Data processing is completed.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # plot annoatated metabolites
    if params.plot_ms2:
        print("Visualization step 1: Plotting MS2 matching...")
        plot_ms2_matching_from_feature_table(feature_table, params)
        print("\tMS2 plotting is completed.")

    print("The workflow is completed.")
    

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
        p.starmap(feature_detection, [(f, params, True, True, False, False, False ) for f in raw_file_names[i:i+batch_size]])
        p.close()
        p.join()