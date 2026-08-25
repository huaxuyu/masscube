# Author: Huaxu Yu

# A module to summarize the premade data processing workflows.

# Import modules
import os
from copy import deepcopy
from importlib.metadata import version
import multiprocessing
import pickle
import time

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from tqdm import tqdm

from .raw_data_utils import (
    find_raw_data_info,
    get_raw_data_format,
    is_supported_raw_data_path,
    read_raw_file_to_obj,
)
from .raw_data_utils.cache import (
    _mcraw_cache_is_current,
    _prepare_mcraw_cache,
)

from .params import Params
from .project import ProjectContext, ProjectMetadata, WorkflowState
from .alignment import (
    convert_features_to_df,
    feature_alignment,
    output_feature_table,
    output_feature_to_msp,
)
from .annotation import (
    MS2LibraryError,
    annotate_aligned_features,
    annotate_features,
    feature_annotation_mzrt,
)
from .feature_grouping import (
    group_features_after_alignment,
    group_features_single_file,
)
from .normalization import sample_normalization, signal_normalization
from .stats import full_statistical_analysis
from .utils_functions import (
    convert_signals_to_string,
)
from .visualization import plot_ms2_matching_from_feature_table


# 1. Untargeted metabolomics data processing for a single file
class SingleFileProcessingSkipped(RuntimeError):
    """A valid request that cannot produce a processed feature table."""


def process_single_file(file_name: str, params: Params = None, segment_feature: bool = None,
                        group_features: bool = None, annotate_ms2: bool = None, ms2_library_path: str = None,
                        output_dir: str = None, bpc_output_dir: str = None,
                        cache_dir: str = None, return_data: bool = True):
    """
    Untargeted data processing for one centroid mzML, Bruker TDF2 ``.d`` source,
    or MassCube ``.mcraw`` cache.
    Override rule: explicit function argument overrides Params.

    Parameters
    ----------
    file_name : str
        Path to the raw mzML file, Bruker ``.d`` directory, or mcraw cache.
    params : Params object
        Parameters for single file processing. If None, the default parameters are used
        based on the type of mass spectrometer.
    segment_feature : bool or None
        Whether to segment the feature's ion trace for distinguishing possible isomers. 
        If None, ``params.segment_features`` is used.
    group_features : bool or None
        Whether to group features by isotopes, adducts and in-source fragments.
        If None, ``params.group_features_single_file`` is used.
    annotate_ms2 : bool or None
        Whether to annotate MS2 spectra. If None, ``params.annotate_ms2`` is used.
    ms2_library_path : str
        Absolute path to the MS2 library.
    output_dir : str
        Explicit output directory for the feature table. Supplying it enables
        table output even when ``params.output_single_file`` is False.
    bpc_output_dir : str
        Explicit output directory for a base-peak chromatogram. Supplying it
        enables BPC output even when ``params.plot_bpc`` is False.
    cache_dir : str
        Optional directory in which a non-mcraw input is cached as mcraw.
    return_data : bool
        Whether to return the processed data. Default is True.

    Returns
    -------
    d : MSData object
        An MSData object containing the processed data. If return_data is False, returns None.

    Raises
    ------
    SingleFileProcessingSkipped
        The input is recognized but cannot produce a processed feature table,
        for example because it is profile mode or contains no selected MS1 scans.
    """

    step = "INIT"

    try:
        # STEP 1. data reading, parsing, and parameter preparation
        step = "STEP 1: data reading and parameter preparation"
        analysis_file = os.path.abspath(os.fspath(file_name))
        ms_info = find_raw_data_info(analysis_file)
        ms_type, ion_mode, is_centroid, _ = ms_info
        # skip the file if it is not centroid
        if not is_centroid:
            raise SingleFileProcessingSkipped(
                f"File is not centroid: {analysis_file}"
            )
        if params is None:
            params = Params()
            params.set_default(ms_type, ion_mode)
        else:
            # A processed MSData records the effective parameters for this
            # file without mutating the caller's reusable configuration.
            params = deepcopy(params)

        params.segment_features = (
            params.segment_features
            if segment_feature is None
            else _require_boolean_override(segment_feature, "segment_feature")
        )
        params.group_features_single_file = (
            params.group_features_single_file
            if group_features is None
            else _require_boolean_override(group_features, "group_features")
        )
        params.annotate_ms2 = (
            params.annotate_ms2
            if annotate_ms2 is None
            else _require_boolean_override(annotate_ms2, "annotate_ms2")
        )
        if ms2_library_path is not None:
            params.ms2_library_path = os.fspath(ms2_library_path)
        params.check_parameters()

        segment_feature = params.segment_features
        group_features = params.group_features_single_file
        annotate_ms2 = params.annotate_ms2

        # A direct caller with a workflow tmp directory gets the same native
        # cache behavior as the project workflow. Existing mcraw paths pass
        # through unchanged.
        if (
            get_raw_data_format(analysis_file) != "mcraw"
            and cache_dir is not None
        ):
            analysis_file = _prepare_mcraw_cache(
                analysis_file, cache_dir, params, ms_info=ms_info
            )

        d = read_raw_file_to_obj(
            analysis_file,
            params=params,
            ms_info=ms_info,
        )

        # check if the MS1 data is valid (no MS1 data found when intensity tolerance is too high)
        if len(d.ms1_idx_arr) == 0:
            raise SingleFileProcessingSkipped(
                "No valid MS1 data were found. Please check the file and "
                f"MS1 intensity tolerance: {file_name}"
            )

        # check if the file is centroid
        if not d.metadata.is_centroid:
            raise SingleFileProcessingSkipped(
                f"File is not centroid after loading: {file_name}"
            )
        # STEP 2. feature detection and segmentation
        step = "STEP 2: feature detection and segmentation"
        d.detect_features()

        if segment_feature:
            d.segment_features()

        # STEP 3. feature evaluation
        step = "STEP 3: feature finalization"
        d.finalize_features()

        # STEP 4. MS2 annotation
        step = "STEP 4: MS2 annotation"
        if annotate_ms2:
            if ms2_library_path is None:
                ms2_library_path = d.params.ms2_library_path
            if ms2_library_path is not None:
                try:
                    annotate_features(
                        d=d,
                        sim_tol=d.params.ms2_sim_tol,
                        fuzzy_search=d.params.fuzzy_search,
                        ms2_library_path=ms2_library_path,
                        consider_rt=d.params.consider_rt,
                        similarity_method=d.params.spectral_similarity_method,
                    )
                except MS2LibraryError as e:
                    _print_ms2_annotation_skip_message(e, ms2_library_path, indent="\t")
            else:
                print("\tMS2 annotation is skipped because no MS2 library path is configured.")

        # STEP 5. feature grouping
        step = "STEP 5: feature grouping"
        if group_features:
            group_features_single_file(d, rt_tol=d.params.rt_tol_feature_grouping)

        # STEP 6. visualization and output
        step = "STEP 6: visualization and output"
        default_output_dir = os.path.dirname(os.path.abspath(os.fspath(file_name)))
        if d.params.plot_bpc or bpc_output_dir is not None:
            bpc_output_dir = bpc_output_dir or default_output_dir
            os.makedirs(bpc_output_dir, exist_ok=True)
            d.plot_bpc(
                output_dir=os.path.join(
                    bpc_output_dir, d.metadata.file_name + "_bpc.png"
                )
            )

        if d.params.output_single_file or output_dir is not None:
            output_dir = output_dir or default_output_dir
            os.makedirs(output_dir, exist_ok=True)
            d.output_single_file(
                os.path.join(output_dir, d.metadata.file_name + ".txt")
            )
        
        if return_data:
            return d
        else:
            return None
    
    except SingleFileProcessingSkipped as error:
        print(f"\tSkipped: {file_name}")
        print(f"\t\t{error}")
        raise
    except Exception as e:
        print(f"\tError occurred: {file_name}")
        print(f"\t\tFailed at {step}.")
        print(f"\t\t{type(e).__name__}: {e}")
        raise


def _require_boolean_override(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{name} must be a boolean or None.")


def _process_project_single_file(
    file_name,
    params,
    output_dir,
    bpc_output_dir,
):
    """Process one project file and return a lightweight status outcome."""

    sample_name = _pure_file_name(file_name)
    started_at = _time_string()
    try:
        data = process_single_file(
            file_name,
            params,
            output_dir=output_dir,
            bpc_output_dir=bpc_output_dir,
            return_data=True,
        )
        feature_count = len(data.features)
    except SingleFileProcessingSkipped as error:
        return {
            "sample_name": sample_name,
            "status": "skipped",
            "reason": str(error),
            "feature_count": None,
            "started_at": started_at,
            "ended_at": _time_string(),
        }
    except Exception as error:
        return {
            "sample_name": sample_name,
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "feature_count": None,
            "started_at": started_at,
            "ended_at": _time_string(),
        }
    return {
        "sample_name": sample_name,
        "status": "completed",
        "reason": "",
        "feature_count": feature_count,
        "started_at": started_at,
        "ended_at": _time_string(),
    }


def _prepare_project_mcraw_cache(file_name, tmp_file_dir, params, ms_info):
    """Prepare one cache without losing the source name on failure."""

    sample_name = _pure_file_name(file_name)
    try:
        cache_path = _prepare_mcraw_cache(
            file_name,
            tmp_file_dir,
            params,
            ms_info=ms_info,
        )
    except Exception as error:
        return {
            "sample_name": sample_name,
            "cache_path": None,
            "reason": f"{type(error).__name__}: {error}",
        }
    return {
        "sample_name": sample_name,
        "cache_path": cache_path,
        "reason": "",
    }

# 2. Project-level untargeted metabolomics workflow
DEPENDENCIES = (
    "masscube",
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "pyteomics",
    "scikit-learn",
    "ms_entropy",
    "lxml",
)

DATA_PROCESSING_METADATA = [
    {
        "name": "overview",
        "layer": 0,
        "dependencies": None,
        "start_time": None,
        "end_time": None,
    },
    {"name": "parameters", "layer": 1},
    {"name": "feature_detection", "layer": 2, "status": "not completed"},
    {"name": "feature_alignment", "layer": 3, "status": "not completed"},
    {"name": "feature_annotation", "layer": 4, "status": "not completed"},
    {"name": "signal_normalization", "layer": 5, "status": "not completed"},
    {"name": "sample_normalization", "layer": 6, "status": "not completed"},
    {"name": "statistical_analysis", "layer": 7, "status": "not completed"},
]


class UntargetedMetabolomicsWorkflow:
    """A complete MassCube project and its processing workflow.

    ``params`` contains processing choices only. Filesystem locations are on
    ``context``; project/sample facts are on ``metadata``; mutable execution
    information is on ``state``.
    """

    def __init__(self, path=None, params=None):
        self.context = ProjectContext(path if path is not None else os.getcwd())
        self.params = params if params is not None else Params()
        self.metadata = ProjectMetadata(
            project_name=os.path.basename(self.context.project_dir.rstrip(os.sep))
        )
        self.state = WorkflowState(
            processing_metadata=deepcopy(DATA_PROCESSING_METADATA)
        )
        self.features = None
        self.feature_table = None
        self._prepared = False
        self._params_supplied = params is not None

    @property
    def project_dir(self):
        return self.context.project_dir

    def prepare(self):
        """Validate and initialize project context, parameters, and metadata."""

        self.state.start_step("preparation")
        context = self.context
        if not os.path.isdir(context.project_dir):
            raise ValueError(
                "The project directory does not exist. Please create the "
                "directory first."
            )
        if not os.path.isdir(context.sample_dir):
            raise ValueError("No raw MS data is found in the project directory.")

        sample_files = sorted(
            file_name
            for file_name in os.listdir(context.sample_dir)
            if is_supported_raw_data_path(
                os.path.join(context.sample_dir, file_name)
            )
        )
        if not sample_files:
            raise ValueError("No raw MS data is found in the project directory.")

        sample_table_path = os.path.join(context.project_dir, "sample_table.csv")
        parameter_path = os.path.join(context.project_dir, "parameters.csv")
        if not os.path.exists(sample_table_path):
            print(
                "\tNo sample table is found in the project directory. "
                "Normalization and statistical analysis will NOT be performed."
            )
            self.params.run_statistics = False
            self.params.sample_normalization = False
            self.params.signal_normalization = False
        if not os.path.exists(parameter_path):
            print(
                "\tNo parameter file is found in the project directory. "
                "Default parameters will be used."
            )
            print(
                "\tTo perform feature annotation, please specify the path of "
                "the MS/MS library in the parameter file."
            )

        context.create_output_directories()
        self.metadata = ProjectMetadata.from_sources(
            context,
            sample_files,
            sample_table_path,
        )
        first_valid = self.metadata.valid_samples.iloc[0]
        if not self._params_supplied:
            self.params.set_default(
                first_valid["MS_TYPE"], first_valid["ION_MODE"]
            )
        if os.path.exists(parameter_path):
            self.params.read_parameters_from_csv(
                parameter_path, ignore_metadata_fields=True
            )
        else:
            print("Using default parameters...")
            self.params.plot_bpc = True
        self.params.output_single_file = True
        self.params.check_parameters()

        self.metadata.sample_metadata.to_csv(
            os.path.join(context.project_file_dir, "sample_table_with_time.csv"),
            index=False,
        )

        run_metadata = self.state.processing_metadata
        self.state.run_id = _new_run_id()
        self.state.started_at = _time_string()
        run_metadata[0]["start_time"] = self.state.started_at
        run_metadata[0]["run_id"] = self.state.run_id
        run_metadata[0]["dependencies"] = {
            item: version(item) for item in DEPENDENCIES
        }
        run_metadata[0]["project"] = self.metadata.as_dict()
        run_metadata[1].update(self.params.as_dict())
        self._prepared = True
        return self

    def process_single_files(self):
        """Prepare native caches and process files that need regeneration."""

        self._require_prepared()
        self.state.start_step("feature_detection")
        context = self.context
        params = self.params
        sample_metadata = self.metadata.sample_metadata
        self.state.run_id = _new_run_id()
        self.state.processing_metadata[0]["run_id"] = self.state.run_id
        self.state.problematic_files = {}
        workers = max(
            1, int(multiprocessing.cpu_count() * params.percent_cpu_to_use)
        )
        print(
            "\tA total of {} CPU cores are detected, {} cores are used.".format(
                multiprocessing.cpu_count(), workers
            )
        )

        valid_samples = self.metadata.valid_samples
        source_entries = [
            (
                os.fspath(sample["ABSOLUTE_PATH"]),
                (
                    sample["MS_TYPE"],
                    sample["ION_MODE"],
                    bool(sample["IS_CENTROID"]),
                    None if pd.isna(sample["time"]) else sample["time"],
                ),
            )
            for _, sample in valid_samples.iterrows()
        ]
        source_files = [source for source, _ in source_entries]
        cache_was_current = {
            _pure_file_name(source): _mcraw_cache_is_current(
                source, context.tmp_file_dir
            )
            for source in source_files
        }
        processed_files = {
            os.path.splitext(file_name)[0]
            for file_name in os.listdir(context.single_file_dir)
            if file_name.lower().endswith(".txt")
        }
        status_path = os.path.join(
            context.project_file_dir, "single_file_processing_status.csv"
        )
        previous_status = _read_single_file_status(status_path)
        reusable_files = _reusable_single_file_results(
            processed_files, previous_status
        )
        status_table = _initialize_single_file_status(
            sample_metadata=sample_metadata,
            single_file_dir=context.single_file_dir,
            cache_was_current=cache_was_current,
            reusable_files=reusable_files,
            previous_status=previous_status,
            run_id=self.state.run_id,
        )
        self.state.artifacts["single_file_processing_status"] = status_path
        _write_single_file_status(status_table, status_path)
        for _, row in status_table[status_table["status"] == "skipped"].iterrows():
            self.state.problematic_files[str(row["sample_name"])] = str(
                row["reason"]
            )

        to_be_processed_names = set(
            status_table.loc[
                status_table["status"] == "pending", "sample_name"
            ].astype(str)
        )
        print(
            f"\t{len(to_be_processed_names)} files to process out of "
            f"{len(sample_metadata)} files."
        )

        if to_be_processed_names:
            started_at = _time_string()
            pending_mask = status_table["sample_name"].isin(
                to_be_processed_names
            )
            status_table.loc[pending_mask, "status"] = "running"
            status_table.loc[pending_mask, "started_at"] = started_at
            _write_single_file_status(status_table, status_path)

        print(f"\tPreparing or validating {len(source_files)} mcraw caches...")
        cache_results = Parallel(
            n_jobs=workers,
            backend="loky",
            return_as="generator",
        )(
            delayed(_prepare_project_mcraw_cache)(
                source,
                context.tmp_file_dir,
                params,
                ms_info,
            )
            for source, ms_info in source_entries
        )
        cache_by_sample = {}
        cache_failures = []
        for outcome in tqdm(
            cache_results,
            total=len(source_files),
            desc="mcraw",
            unit="file",
        ):
            sample_name = outcome["sample_name"]
            if outcome["reason"]:
                cache_failures.append(outcome)
                _update_single_file_status(
                    status_table,
                    {
                        "sample_name": sample_name,
                        "status": "failed",
                        "reason": outcome["reason"],
                        "feature_count": None,
                        "started_at": status_table.loc[
                            status_table["sample_name"] == sample_name,
                            "started_at",
                        ].iloc[0],
                        "ended_at": _time_string(),
                    },
                )
                self.state.problematic_files[sample_name] = outcome["reason"]
            else:
                cache_by_sample[sample_name] = outcome["cache_path"]
        _write_single_file_status(status_table, status_path)
        if cache_failures:
            failed_names = ", ".join(
                outcome["sample_name"] for outcome in cache_failures
            )
            raise RuntimeError(
                f"Failed to prepare mcraw cache for: {failed_names}"
            )

        to_be_processed = [
            cache_by_sample[sample_name]
            for sample_name in status_table["sample_name"].astype(str)
            if sample_name in to_be_processed_names
        ]

        import gc
        import math

        n_batches = math.ceil(len(to_be_processed) / params.batch_size)
        for batch_index in range(n_batches):
            start = batch_index * params.batch_size
            end = min(start + params.batch_size, len(to_be_processed))
            batch = to_be_processed[start:end]
            print(
                f"\nProcessing batch {batch_index + 1}/{n_batches} "
                f"({len(batch)} files)"
            )
            results = Parallel(
                n_jobs=workers,
                backend="loky",
                return_as="generator",
            )(
                delayed(_process_project_single_file)(
                    file_name,
                    params,
                    output_dir=context.single_file_dir,
                    bpc_output_dir=(context.bpc_dir if params.plot_bpc else None),
                )
                for file_name in batch
            )
            failed_outcomes = []
            for outcome in tqdm(
                results,
                total=len(batch),
                desc=f"Batch {batch_index + 1}",
                unit="file",
            ):
                _update_single_file_status(status_table, outcome)
                if outcome["status"] in {"skipped", "failed"}:
                    sample_name = outcome["sample_name"]
                    self.state.problematic_files[sample_name] = outcome["reason"]
                if outcome["status"] == "skipped":
                    sample_mask = sample_metadata["sample_name"] == sample_name
                    sample_metadata.loc[sample_mask, "VALID"] = False
                if outcome["status"] == "failed":
                    failed_outcomes.append(outcome)
            _write_single_file_status(status_table, status_path)
            gc.collect()
            if failed_outcomes:
                details = "; ".join(
                    f"{outcome['sample_name']}: {outcome['reason']}"
                    for outcome in failed_outcomes
                )
                raise RuntimeError(
                    f"Single-file processing failed. {details}"
                )

        status_counts = {
            str(name): int(count)
            for name, count in status_table["status"].value_counts().items()
        }
        self.state.processing_metadata[2]["file_status_counts"] = status_counts
        self.state.processing_metadata[2]["status"] = (
            "completed with skipped files"
            if status_counts.get("skipped", 0) > 0
            else "completed"
        )
        return self

    def align_features(self):
        """Align individual feature tables or reload an existing result."""

        self._require_prepared()
        self.state.start_step("feature_alignment")
        aligned_table_path = os.path.join(
            self.context.project_dir, "aligned_feature_table.txt"
        )
        aligned_features_path = os.path.join(
            self.context.project_file_dir, "aligned_features.pkl"
        )
        if os.path.exists(aligned_table_path):
            self.feature_table = pd.read_csv(
                aligned_table_path, sep="\t", low_memory=False
            )
            if os.path.exists(aligned_features_path):
                with open(aligned_features_path, "rb") as file:
                    self.features = pickle.load(file)
            self.state.processing_metadata[3]["status"] = "skipped"
            return False

        self.features = feature_alignment(
            self.context.single_file_dir,
            self.params,
            sample_metadata=self.metadata.sample_metadata,
            project_file_dir=self.context.project_file_dir,
            tmp_file_dir=self.context.tmp_file_dir,
        )
        self.feature_table = convert_features_to_df(
            features=self.features,
            sample_names=self.metadata.sample_metadata.iloc[:, 0],
            quant_method=self.params.quant_method,
        )
        before_annotation_path = os.path.join(
            self.context.project_file_dir,
            "aligned_feature_table_before_annotation.txt",
        )
        output_feature_table(self.feature_table, before_annotation_path)
        with open(aligned_features_path, "wb") as file:
            pickle.dump(self.features, file)
        self.state.artifacts["aligned_features"] = aligned_features_path
        self.state.processing_metadata[3]["status"] = "completed"
        return True

    def annotate_aligned_features(self):
        """Annotate and group the newly aligned project features."""

        if self.features is None:
            raise ValueError("Aligned feature objects are required for annotation.")
        self.state.start_step("feature_annotation")
        project_ion_mode = self.metadata.ion_mode
        if project_ion_mode == "mixed":
            raise ValueError(
                "Aligned annotation/grouping requires one ion mode per project."
            )

        run_metadata = self.state.processing_metadata
        library_path = self.params.ms2_library_path
        print("\tAnnotating features using the MS2 library...")
        if library_path is not None and str(library_path).strip():
            try:
                self.features = annotate_aligned_features(
                    self.features,
                    self.params,
                    ion_mode=project_ion_mode,
                )
                run_metadata[4]["ms2_annotation_status"] = "completed"
                print("\tMS2 annotation is completed.")
            except MS2LibraryError as error:
                run_metadata[4]["ms2_annotation_status"] = "skipped"
                run_metadata[4]["ms2_annotation_error"] = (
                    f"{type(error).__name__}: {error}"
                )
                _print_ms2_annotation_skip_message(error, library_path)
        else:
            run_metadata[4]["ms2_annotation_status"] = "skipped"
            print("\tNo MS2 library path is provided. MS2 annotation is skipped.")

        mzrt_path = os.path.join(self.context.project_dir, "mzrt_list.csv")
        if os.path.exists(mzrt_path):
            print("\tAnnotating features using the extra mzrt list...")
            self.features = feature_annotation_mzrt(
                self.features,
                mzrt_path,
                self.params.mz_tol_alignment,
                self.params.rt_tol_alignment,
            )
            print("\tmz/rt annotation is completed.")

        print("\tAnnotating feature groups...")
        if self.params.group_features_after_alignment:
            group_features_after_alignment(
                self.features,
                self.params,
                ion_mode=project_ion_mode,
                sample_metadata=self.metadata.sample_metadata,
                project_file_dir=self.context.project_file_dir,
                tmp_file_dir=self.context.tmp_file_dir,
            )
        for feature in self.features:
            feature.isotope_signals = convert_signals_to_string(
                feature.isotope_signals
            )
        run_metadata[4]["status"] = "completed"

        self.feature_table = convert_features_to_df(
            features=self.features,
            sample_names=self.metadata.sample_metadata.iloc[:, 0],
            quant_method=self.params.quant_method,
        )
        aligned_table_path = os.path.join(
            self.context.project_dir, "aligned_feature_table.txt"
        )
        msp_path = os.path.join(self.context.project_file_dir, "features.msp")
        aligned_features_path = os.path.join(
            self.context.project_file_dir, "aligned_features.pkl"
        )
        output_feature_table(self.feature_table, aligned_table_path)
        output_feature_to_msp(self.feature_table, msp_path)
        with open(aligned_features_path, "wb") as file:
            pickle.dump(self.features, file)
        self.state.artifacts.update(
            {
                "aligned_feature_table": aligned_table_path,
                "aligned_features": aligned_features_path,
                "features_msp": msp_path,
            }
        )
        return self

    def normalize(self):
        """Apply enabled signal and sample normalization steps."""

        if self.feature_table is None:
            raise ValueError("A feature table is required for normalization.")
        sample_metadata = self.metadata.sample_metadata
        run_metadata = self.state.processing_metadata

        self.state.start_step("signal_normalization")
        if self.params.signal_normalization:
            plot_dir = (
                self.context.normalization_dir
                if self.params.plot_normalization
                else None
            )
            self.feature_table = signal_normalization(
                self.feature_table,
                sample_metadata,
                self.params.signal_norm_method,
                output_plot_path=plot_dir,
            )
            run_metadata[5]["status"] = "completed"
        else:
            run_metadata[5]["status"] = "skipped"

        self.state.start_step("sample_normalization")
        if self.params.sample_normalization:
            self.feature_table = sample_normalization(
                self.feature_table,
                sample_metadata,
                self.params.sample_norm_method,
            )
            run_metadata[6]["status"] = "completed"
        else:
            run_metadata[6]["status"] = "skipped"

        if self.params.sample_normalization or self.params.signal_normalization:
            output_path = os.path.join(
                self.context.project_dir, "normalized_feature_table.txt"
            )
            output_feature_table(self.feature_table, output_path)
            self.state.artifacts["normalized_feature_table"] = output_path
        return self

    def run_statistics(self):
        """Run enabled project-level statistical analysis."""

        self.state.start_step("statistical_analysis")
        if self.params.run_statistics:
            self.feature_table = full_statistical_analysis(
                self.feature_table,
                self.params,
                sample_metadata=self.metadata.sample_metadata,
                output_dir=self.context.statistics_dir,
            )
            self.state.processing_metadata[7]["status"] = "completed"
        else:
            self.state.processing_metadata[7]["status"] = "skipped"
        return self

    def save(self):
        """Persist run metadata and a lightweight project workflow object."""

        self.state.ended_at = _time_string()
        self.state.processing_metadata[0]["end_time"] = self.state.ended_at
        time_label = time.strftime("%Y%m%d%H%M%S", time.localtime())
        metadata_path = os.path.join(
            self.context.project_file_dir,
            f"data_processing_metadata_{time_label}.pkl",
        )
        project_path = os.path.join(
            self.context.project_file_dir, "project.masscube"
        )
        self.state.artifacts["processing_metadata"] = metadata_path
        self.state.artifacts["project"] = project_path
        with open(metadata_path, "wb") as file:
            pickle.dump(self.state.processing_metadata, file)
        with open(project_path, "wb") as file:
            pickle.dump(self, file)
        return self

    @classmethod
    def load(cls, path):
        """Load a workflow from a project or ``project.masscube`` path."""

        project_path = os.fspath(path)
        if os.path.isdir(project_path):
            project_path = os.path.join(
                project_path, "project_files", "project.masscube"
            )
        class _WorkflowUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if (
                    module == "masscube.project_workflow"
                    and name == "UntargetedMetabolomicsWorkflow"
                ):
                    return cls
                return super().find_class(module, name)

        with open(project_path, "rb") as file:
            workflow = _WorkflowUnpickler(file).load()
        if not isinstance(workflow, cls):
            raise TypeError(
                "The saved project predates the workflow-object format and "
                "cannot be loaded as UntargetedMetabolomicsWorkflow."
            )
        workflow._reload_results()
        return workflow

    def __getstate__(self):
        """Avoid duplicating large result artifacts inside project.masscube."""

        state = self.__dict__.copy()
        state["features"] = None
        state["feature_table"] = None
        return state

    def run(self, only_process_single_files=False):
        """Execute the complete workflow and return this project object."""

        _banner("Welcome to the untargeted metabolomics workflow")
        try:
            print("Step 1: Preparing the workflow...")
            if not self._prepared:
                self.prepare()
            print("\tWorkflow is prepared.")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            print(
                "Step 2: Preparing mcraw caches and processing individual files..."
            )
            self.process_single_files()
            print("\tIndividual file processing is completed.")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if only_process_single_files:
                self.state.complete()
                self.save()
                return self

            created_alignment = self.align_features()
            if created_alignment:
                print("Step 3: Feature alignment is completed.")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("Step 4: Annotating features...")
                self.annotate_aligned_features()
                print("\tFeature annotation and grouping are completed.")
            else:
                print(
                    "Step 3: Feature alignment is skipped. Using the existing "
                    "aligned feature table."
                )
                print(
                    "Step 4: Feature annotation is skipped. Using the existing "
                    "aligned feature table."
                )
                self.state.processing_metadata[4]["status"] = "skipped"
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            print(
                "Step 5: Running signal normalization..."
                if self.params.signal_normalization
                else "Step 5: MS signal drift normalization is skipped."
            )
            print(
                "Step 6: Running sample normalization..."
                if self.params.sample_normalization
                else "Step 6: Sample normalization is skipped."
            )
            self.normalize()
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            print(
                "Step 7: Running statistical analysis..."
                if self.params.run_statistics
                else "Step 7: Statistical analysis is skipped."
            )
            self.run_statistics()
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            if self.params.plot_ms2:
                print("Plotting MS2 matching...")
                plot_ms2_matching_from_feature_table(
                    self.feature_table,
                    output_dir=self.context.ms2_matching_dir,
                )
                print("\tMS2 plotting is completed.")

            self.state.complete()
            self.save()
            print("The workflow is completed.")
            return self
        except Exception as error:
            self.state.record_error(error)
            raise

    def _require_prepared(self):
        if not self._prepared:
            raise RuntimeError("Call prepare() before running workflow steps.")

    def _reload_results(self):
        aligned_table_path = os.path.join(
            self.context.project_dir, "aligned_feature_table.txt"
        )
        aligned_features_path = os.path.join(
            self.context.project_file_dir, "aligned_features.pkl"
        )
        if os.path.exists(aligned_table_path):
            self.feature_table = pd.read_csv(
                aligned_table_path, sep="\t", low_memory=False
            )
        if os.path.exists(aligned_features_path):
            with open(aligned_features_path, "rb") as file:
                self.features = pickle.load(file)
        return self


def run_untargeted_metabolomics_workflow(
    path=None,
    return_results=False,
    only_process_single_files=False,
    return_params_only=False,
):
    """Functional compatibility adapter around the workflow object."""

    workflow = UntargetedMetabolomicsWorkflow(path)
    if return_params_only:
        workflow.prepare()
        return workflow.params
    workflow.run(only_process_single_files=only_process_single_files)
    if only_process_single_files:
        return None
    if return_results:
        if workflow.features is None:
            aligned_features_path = os.path.join(
                workflow.context.project_file_dir, "aligned_features.pkl"
            )
            raise FileNotFoundError(
                "Cannot return aligned features because the saved feature object "
                f"does not exist: {aligned_features_path}"
            )
        return workflow.features, workflow.params
    return None


SINGLE_FILE_STATUSES = frozenset(
    {"pending", "running", "completed", "reused", "skipped", "failed"}
)

SINGLE_FILE_STATUS_COLUMNS = (
    "sample_name",
    "source_path",
    "result_path",
    "status",
    "reason",
    "feature_count",
    "started_at",
    "ended_at",
    "run_id",
)


def _read_single_file_status(path):
    if not os.path.exists(path):
        return None
    text_columns = {
        column: "string"
        for column in SINGLE_FILE_STATUS_COLUMNS
        if column != "feature_count"
    }
    status_table = pd.read_csv(path, dtype=text_columns)
    missing = [
        column
        for column in SINGLE_FILE_STATUS_COLUMNS
        if column not in status_table.columns
    ]
    if missing:
        raise ValueError(
            "Single-file processing status is missing required columns: "
            + ", ".join(missing)
        )
    if status_table["sample_name"].duplicated().any():
        raise ValueError(
            "Single-file processing status contains duplicate sample names."
        )
    if status_table["sample_name"].isna().any():
        raise ValueError(
            "Single-file processing status contains an empty sample name."
        )
    if status_table["status"].isna().any():
        raise ValueError(
            "Single-file processing status contains an empty state."
        )
    invalid_statuses = set(status_table["status"].dropna().astype(str)) - set(
        SINGLE_FILE_STATUSES
    )
    if invalid_statuses:
        raise ValueError(
            "Single-file processing status contains invalid states: "
            + ", ".join(sorted(invalid_statuses))
        )
    return status_table


def _reusable_single_file_results(processed_files, previous_status):
    if previous_status is None:
        # Existing projects created before the status table was introduced.
        return set(processed_files)
    reusable_status = previous_status["status"].isin({"completed", "reused"})
    reusable_names = set(
        previous_status.loc[reusable_status, "sample_name"].astype(str)
    )
    return set(processed_files) & reusable_names


def _initialize_single_file_status(
    sample_metadata,
    single_file_dir,
    cache_was_current,
    reusable_files,
    previous_status,
    run_id,
):
    previous_by_name = {}
    if previous_status is not None:
        previous_by_name = {
            str(row["sample_name"]): row
            for _, row in previous_status.iterrows()
        }

    rows = []
    timestamp = _time_string()
    for _, sample in sample_metadata.iterrows():
        sample_name = str(sample["sample_name"])
        source_path = sample.get("ABSOLUTE_PATH")
        if pd.isna(source_path):
            source_path = ""
        else:
            source_path = os.fspath(source_path)
        result_path = os.path.join(single_file_dir, sample_name + ".txt")
        valid = bool(sample.get("VALID", False))

        if not valid:
            reason = sample.get("INVALID_REASON", "")
            if pd.isna(reason) or not str(reason).strip():
                previous = previous_by_name.get(sample_name)
                reason = "" if previous is None else previous.get("reason", "")
            if pd.isna(reason):
                reason = ""
            status = "skipped"
            feature_count = None
            ended_at = timestamp
        elif (
            sample_name in reusable_files
            and cache_was_current.get(sample_name, False)
        ):
            status = "reused"
            reason = ""
            feature_count = _count_single_file_features(result_path)
            ended_at = timestamp
        else:
            status = "pending"
            reason = ""
            feature_count = None
            ended_at = ""

        rows.append(
            {
                "sample_name": sample_name,
                "source_path": source_path,
                "result_path": result_path,
                "status": status,
                "reason": str(reason) if reason is not None else "",
                "feature_count": feature_count,
                "started_at": "",
                "ended_at": ended_at,
                "run_id": run_id,
            }
        )
    return pd.DataFrame(rows, columns=SINGLE_FILE_STATUS_COLUMNS)


def _update_single_file_status(status_table, outcome):
    sample_name = str(outcome["sample_name"])
    mask = status_table["sample_name"].astype(str) == sample_name
    if not mask.any():
        raise KeyError(f"Unknown sample in single-file status: {sample_name}")
    for column in (
        "status",
        "reason",
        "feature_count",
        "started_at",
        "ended_at",
    ):
        status_table.loc[mask, column] = outcome.get(column)


def _write_single_file_status(status_table, path):
    """Atomically replace the project-level single-file status CSV."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp-{os.getpid()}-{time.time_ns()}"
    output = status_table.loc[:, SINGLE_FILE_STATUS_COLUMNS].copy()
    output["feature_count"] = pd.array(
        pd.to_numeric(output["feature_count"], errors="coerce"),
        dtype="Int64",
    )
    try:
        output.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _count_single_file_features(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            return max(sum(1 for _ in file) - 1, 0)
    except OSError:
        return None


def _pure_file_name(path):
    return os.path.splitext(os.path.basename(os.fspath(path)))[0]


def _new_run_id():
    return (
        time.strftime("%Y%m%dT%H%M%S", time.localtime())
        + f"-{time.time_ns() % 1_000_000_000:09d}"
    )


def _time_string():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())


class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"


def _banner(title):
    print(f"\n{_C.BOLD}{_C.CYAN}{'═' * 60}")
    print(f" {title}")
    print(f"{'═' * 60}{_C.RESET}")

# 3. Evaluate the data quality of the raw files
def run_evaluation(path=None):
    """
    Evaluate the run and report the problematic files.

    Parameters
    ----------
    path : str
        Path to the project directory.
    zscore_threshold : float
        The threshold of z-score for detecting problematic files. Default is -2.
    """

    if path is None:
        path = os.getcwd()

    # check if sample table exists
    if os.path.exists(os.path.join(path, "sample_table.csv")):
        sample_table = pd.read_csv(os.path.join(path, "sample_table.csv"))
        sample_table['is_blank'] = sample_table['is_blank'].apply(lambda x: True if x.lower() == 'yes' else False)
        blank_samples = sample_table[sample_table['is_blank']].iloc[:, 0].values
    else:
        print("Sample table is not found. Problematic files may include blank samples.")
        blank_samples = []

    # get all .txt files
    txt_path = os.path.join(path, "single_files")
    txt_files = [f for f in os.listdir(txt_path) if f.lower().endswith('.txt')]
    txt_files = [f for f in txt_files if not f.startswith(".")]
    txt_files = [f for f in txt_files if f.split(".")[0] not in blank_samples]

    int_array = np.zeros(len(txt_files))
    for i in range(len(txt_files)):
        df = pd.read_csv(os.path.join(txt_path, txt_files[i]), sep="\t", low_memory=False)
        int_array[i] = np.max(df['peak_height'].values)
        
    v = int_array < np.median(int_array) / 3

    problematic_files = [txt_files[i].split(".")[0] for i in range(len(txt_files)) if v[i]]
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


def _print_ms2_annotation_skip_message(error, ms2_library_path, indent="\t"):
    """
    Print a consistent, non-fatal message when MS2 library annotation cannot run.
    """

    print(f"{indent}MS2 annotation is skipped because the MS2 library could not be loaded or used.")
    print(f"{indent}MS2 library path: {ms2_library_path}")
    print(f"{indent}{type(error).__name__}: {error}")


def untargeted_metabolomics_workflow(
    path: str = None,
    return_results: bool = False,
    only_process_single_files: bool = False,
    return_params_only: bool = False,
):
    """Run the object-oriented workflow through the historical function API."""

    return run_untargeted_metabolomics_workflow(
        path=path,
        return_results=return_results,
        only_process_single_files=only_process_single_files,
        return_params_only=return_params_only,
    )
