# Author: Hauxu Yu

# A module to summarize the premade data processing workflows.

# Import modules
import os
import multiprocessing
import pickle
from copy import deepcopy
import pandas as pd
import numpy as np
from tqdm import tqdm

from .raw_data_utils import MSData, get_start_time
from .params import Params, find_ms_info
from .feature_grouping import annotate_isotope, annotate_adduct, annotate_in_source_fragment
from .alignment import feature_alignment, gap_filling, output_feature_table
from .annotation import feature_annotation, annotate_rois, output_ms2_to_msp
from .normalization import sample_normalization
from .visualization import plot_ms2_matching_from_feature_table
from .network import network_analysis
from .stats import statistical_analysis
from .feature_table_utils import calculate_fill_percentage


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
        d = MSData()

        # set parameters
        ms_type, ion_mode, centrod = find_ms_info(file_name)
        if not centrod:
            print("File: " + file_name + " is not centroided and skipped.")
            return None
        
        # if params is None, use the default parameters
        if params is None:
            params = Params()
            params.set_default(ms_type, ion_mode)
        
        if ms2_library_path is not None:
            params.msms_library = ms2_library_path

        # read raw data
        d.read_raw_data(file_name, params)

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

        if params.plot_bpc:
            d.plot_bpc(label_name=True, output=os.path.join(params.bpc_dir, d.file_name + "_bpc.png"))

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
        p.starmap(feature_detection, [(f, params) for f in raw_file_names[i:i+batch_size]])
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
    
    output_path = os.path.join(params.project_dir, "ms2.msp")
    output_ms2_to_msp(feature_table, output_path)

    # normalization
    if params.run_normalization:
        output_path = os.path.join(params.project_dir, "aligned_feature_table_before_normalization.txt")
        output_feature_table(feature_table, output_path)
        feature_table_before_normalization = deepcopy(feature_table)
        print("Running normalization...")
        feature_table = sample_normalization(feature_table, params.individual_sample_groups, params.normalization_method)

    # statistical analysis
    if params.run_statistics:
        print("Running statistical analysis...")
        feature_table_before_normalization = statistical_analysis(feature_table_before_normalization, params, before_norm=True)
        feature_table = statistical_analysis(feature_table, params)
    
    # output feature table
    output_path = os.path.join(params.project_dir, "aligned_feature_table.txt")
    output_feature_table(feature_table, output_path)

    # network analysis
    if params.run_network:
        print("Running network analysis...This may take several minutes...")
        network_analysis(feature_table)

    # plot annoatated metabolites
    if params.plot_ms2:
        print("Plotting annotated metabolites...")
        plot_ms2_matching_from_feature_table(feature_table, params)
    
    # output feature table
    output_path = os.path.join(params.project_dir, "aligned_feature_table.txt")
    output_feature_table(feature_table, output_path)

    # output parameters and metadata
    params.output_parameters(os.path.join(params.project_dir, "data_processing_metadata.json"))
    print("The workflow is completed.")


# 3. Get analytical metadata from mzML files
def get_analytical_metadata(path):
    """
    Get metadata from mzML or mzXML files.

    Parameters
    ----------
    path : str
        Path to the mzML or mzXML file.
    """

    file_names = os.listdir(path)
    file_names = [f for f in file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
    file_names = [f for f in file_names if not f.startswith(".")]

    times = []
    for f in file_names:
        tmp = os.path.join(path, f)
        times.append(get_start_time(tmp))
    
    # sort the files by time
    file_times = list(zip(file_names, times))
    file_times = sorted(file_times, key=lambda x: x[1])

    # output to a txt file using pandas
    df = pd.DataFrame(file_times, columns=["file_name", "aquisition_time"])
    output_path = os.path.join(path, "analytical_metadata.txt")
    df.to_csv(output_path, sep="\t", index=False)


# 4. Evaluate the data quality of the raw files
def run_evaluation(path):
    """
    Evaluate the run and report the problematic files.

    Parameters
    ----------
    path : str
        Path to the mzML or mzXML files.
    """

    file_names = os.listdir(path)
    file_names = [f for f in file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
    file_names = [f for f in file_names if not f.startswith(".")]
    file_names = [f for f in file_names if "blank" not in f.lower() and "mb" not in f.lower()]

    feature_num = []
    for f in tqdm(file_names):
        d = feature_detection(os.path.join(path, f), params=None, cal_gss=False, anno_isotope=False,  
                              anno_adduct=False, anno_in_source_fragment=False, annotation=False)
        feature_num.append(len(d.rois))

    # find outliers
    problematic_files = []
    mean = np.mean(feature_num)
    std = np.std(feature_num)

    outliers = np.where(np.abs(feature_num - mean) > 3 * std)[0]

    print(outliers)
    for i in outliers[::-1]:
        problematic_files.append(file_names[i])

    if len(problematic_files) == 0:
        print("No problematic files are found.")
    else:
        print("Problematic files:")
        for f in problematic_files:
            print(f)
        
        # output to a txt file
        with open(os.path.join(path, "problematic_files.txt"), "w") as f:
            for file in problematic_files:
                f.write(file + "\n")


# 5. Targeted metabolomics workflow
def targeted_metabolomics_workflow(path=None):
    """
    The targeted metabolomics workflow. The function is under development.

    Parameters
    ----------
    path : str
        The working directory. If None, the current working directory is used.
    """

    pass


# 6. Single-file peak picking (batch mode)
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
    params._untargeted_metabolomics_workflow_preparation()

    with open(os.path.join(params.project_dir, "project.mc"), "wb") as f:
        pickle.dump(params, f)
    
    raw_file_names = os.listdir(params.sample_dir)
    raw_file_names = [f for f in raw_file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
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
        p.starmap(feature_detection, [(f, params) for f in raw_file_names[i:i+batch_size]])
        p.close()
        p.join()


# 7. Determine sample total amount
def sample_total_amount(path):
    """
    Determine the total amount of the samples based on LC-MS data. See the documentation for details.

    Parameters
    ----------
    path : str
        Path to the mzML or mzXML files.
    """

    pass

# 8. Serial QC calibration
def qc_calibration(path):
    """
    Serial QC calibration. See the documentation for details.

    Parameters
    ----------
    path : str
        Path to the mzML or mzXML files.
    """

    pass



# 9. Mass calibration
def mass_calibration(path):
    """
    Mass calibration. See the documentation for details.

    Parameters
    ----------
    path : str
        Path to the mzML or mzXML files.
    """

    pass


# 10. Untargeted metabolomics workflow with multiple analytical modes (RP+, RP-, HILIC+, HILIC-)
def untargeted_metabolomics_workflow_multi_mode(path=None):
    """
    The untargeted metabolomics workflow with multiple analytical modes. See the documentation for details.

    Parameters
    ----------
    path : str
        The working directory. If None, the current working directory is used.
    """

    pass
