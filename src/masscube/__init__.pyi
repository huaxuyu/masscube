from typing import Any, List, Optional, Sequence

from .params import Params
from .project import ProjectContext, ProjectMetadata, WorkflowState
from .workflows import SingleFileProcessingSkipped, UntargetedMetabolomicsWorkflow
from .raw_data_utils import MSData

__version__: str
__all__: List[str]

def process_single_file(
    file_name: str,
    params: Optional[Params] = ...,
    segment_feature: Optional[bool] = ...,
    group_features: Optional[bool] = ...,
    annotate_ms2: Optional[bool] = ...,
    ms2_library_path: Optional[str] = ...,
    output_dir: Optional[str] = ...,
    bpc_output_dir: Optional[str] = ...,
    cache_dir: Optional[str] = ...,
    return_data: bool = ...,
) -> Optional[MSData]: ...

def untargeted_metabolomics_workflow(
    path: Optional[str] = ...,
    return_results: bool = ...,
    only_process_single_files: bool = ...,
    return_params_only: bool = ...,
) -> Any: ...

def run_evaluation(path: Optional[str] = ...) -> Any: ...

def batch_file_processing(
    path: Optional[str] = ...,
    segment_feature: bool = ...,
    group_features: bool = ...,
    evaluate_peak_shape: bool = ...,
    annotate_ms2: bool = ...,
    ms2_library_path: Optional[str] = ...,
    cpu_ratio: float = ...,
    batch_size: int = ...,
) -> Any: ...

def read_raw_file_to_obj(
    file_name: str,
    params: Optional[Params] = ...,
    ms1_abs_int_tol: float = ...,
    ms2_abs_int_tol: float = ...,
    *,
    scan_levels: Optional[Sequence[int]] = ...,
    centroid_mz_tol: Optional[float] = ...,
    ms2_rel_int_tol: Optional[float] = ...,
    precursor_mz_offset: Optional[float] = ...,
    ms_info: Optional[tuple] = ...,
    normalize_tims_intensity: bool = ...,
    zstd_library: Optional[str] = ...,
    progress: Any = ...,
    preprocess: bool = ...,
) -> MSData: ...

def generate_sample_table(path: Optional[str] = ..., output: bool = ...) -> Any: ...

def get_timestamps(path: Optional[str] = ..., output: bool = ...) -> Any: ...

def build_classifier(
    path: Optional[str] = ...,
    by_group: Optional[str] = ...,
    feature_num: Optional[int] = ...,
    gaussian_cutoff: float = ...,
    detection_rate_cutoff: float = ...,
    fill_ratio: float = ...,
    cross_validation_k: int = ...,
) -> Any: ...
