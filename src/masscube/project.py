"""Project-level context, metadata, and workflow state models."""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .raw_data_utils import find_raw_data_info, get_raw_data_format
from .utils_functions import label_batch_id


@dataclass
class ProjectContext:
    """Filesystem locations owned by one MassCube project."""

    project_dir: str
    sample_dir: str = field(init=False)
    single_file_dir: str = field(init=False)
    tmp_file_dir: str = field(init=False)
    ms2_matching_dir: str = field(init=False)
    bpc_dir: str = field(init=False)
    project_file_dir: str = field(init=False)
    normalization_dir: str = field(init=False)
    statistics_dir: str = field(init=False)

    def __post_init__(self):
        self.project_dir = os.path.abspath(os.fspath(self.project_dir))
        self.sample_dir = os.path.join(self.project_dir, "data")
        self.single_file_dir = os.path.join(self.project_dir, "single_files")
        self.tmp_file_dir = os.path.join(self.project_dir, "tmp")
        self.ms2_matching_dir = os.path.join(self.project_dir, "ms2_matching")
        self.bpc_dir = os.path.join(self.project_dir, "chromatograms")
        self.project_file_dir = os.path.join(self.project_dir, "project_files")
        self.normalization_dir = os.path.join(
            self.project_dir, "normalization_results"
        )
        self.statistics_dir = os.path.join(
            self.project_dir, "statistical_analysis"
        )

    def create_output_directories(self):
        """Create all workflow-owned output directories."""

        for path in (
            self.single_file_dir,
            self.tmp_file_dir,
            self.ms2_matching_dir,
            self.bpc_dir,
            self.project_file_dir,
            self.normalization_dir,
            self.statistics_dir,
        ):
            os.makedirs(path, exist_ok=True)

    def as_dict(self) -> Dict[str, str]:
        return {
            name: getattr(self, name)
            for name in (
                "project_dir",
                "sample_dir",
                "single_file_dir",
                "tmp_file_dir",
                "ms2_matching_dir",
                "bpc_dir",
                "project_file_dir",
                "normalization_dir",
                "statistics_dir",
            )
        }


@dataclass
class ProjectMetadata:
    """Facts about a project and its source data, separate from parameters."""

    project_name: str
    sample_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    ms_type: Optional[str] = None
    ion_mode: Optional[str] = None
    is_centroid: Optional[bool] = None
    file_format: Optional[str] = None
    acquisition_time: Any = None

    @classmethod
    def from_sources(
        cls,
        context: ProjectContext,
        sample_files: Sequence[str],
        sample_table_path: Optional[str] = None,
    ) -> "ProjectMetadata":
        """Build and validate project metadata from the raw-data sources."""

        source_by_name = _index_project_sources(context, sample_files)
        sample_metadata = _load_sample_table(source_by_name, sample_table_path)
        sample_metadata = _attach_source_metadata(sample_metadata, source_by_name)

        valid = sample_metadata["VALID"].astype(bool)
        if not valid.any():
            reasons = sample_metadata.loc[
                sample_metadata["INVALID_REASON"].notna(), "INVALID_REASON"
            ].astype(str)
            detail = "; ".join(dict.fromkeys(reasons))
            suffix = f" Details: {detail}" if detail else ""
            raise ValueError(
                "No valid centroid raw MS data is found in the project. "
                f"Please check the sample table and raw data files.{suffix}"
            )

        sample_metadata = sample_metadata.sort_values(
            by="time", na_position="last"
        ).reset_index(drop=True)
        sample_metadata["analytical_order"] = np.arange(len(sample_metadata))
        sample_metadata = label_batch_id(sample_metadata).reset_index(drop=True)

        valid_samples = sample_metadata[sample_metadata["VALID"].astype(bool)]
        first_valid = valid_samples.iloc[0]
        return cls(
            project_name=os.path.basename(context.project_dir.rstrip(os.sep)),
            sample_metadata=sample_metadata,
            ms_type=_common_value(valid_samples["MS_TYPE"]),
            ion_mode=_common_value(valid_samples["ION_MODE"]),
            is_centroid=bool(valid_samples["IS_CENTROID"].astype(bool).all()),
            file_format=_common_value(valid_samples["FILE_FORMAT"]),
            acquisition_time=first_valid["time"],
        )

    @property
    def valid_samples(self) -> pd.DataFrame:
        if self.sample_metadata.empty or "VALID" not in self.sample_metadata:
            return self.sample_metadata
        return self.sample_metadata[self.sample_metadata["VALID"].astype(bool)]

    @property
    def sample_names(self) -> List[str]:
        if self.sample_metadata.empty:
            return []
        if "sample_name" not in self.sample_metadata:
            raise ValueError("Project sample metadata has no 'sample_name' column.")
        return self.sample_metadata["sample_name"].astype(str).tolist()

    @property
    def source_paths(self) -> List[str]:
        samples = self.valid_samples
        if samples.empty or "ABSOLUTE_PATH" not in samples:
            return []
        return [os.fspath(path) for path in samples["ABSOLUTE_PATH"]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "ms_type": self.ms_type,
            "ion_mode": self.ion_mode,
            "is_centroid": self.is_centroid,
            "file_format": self.file_format,
            "acquisition_time": self.acquisition_time,
            "samples": self.sample_metadata.to_dict(orient="records"),
        }


def _index_project_sources(
    context: ProjectContext, sample_files: Sequence[str]
) -> Dict[str, str]:
    source_by_name = {}
    for file_name in sample_files:
        source_path = os.path.join(context.sample_dir, os.fspath(file_name))
        pure_name = os.path.splitext(os.path.basename(source_path))[0]
        if pure_name in source_by_name:
            raise ValueError(
                "Multiple raw files share the same file name without extension: "
                f"{pure_name}"
            )
        source_by_name[pure_name] = source_path
    return source_by_name


def _load_sample_table(
    source_by_name: Dict[str, str], sample_table_path: Optional[str]
) -> pd.DataFrame:
    if sample_table_path is not None and os.path.exists(sample_table_path):
        sample_metadata = pd.read_csv(sample_table_path)
        if sample_metadata.empty:
            raise ValueError("sample_table.csv does not contain any samples.")
        sample_metadata = _normalize_sample_table_columns(sample_metadata)
        if "sample_name" not in sample_metadata:
            raise ValueError(
                "sample_table.csv must contain a 'sample_name' column."
            )
    else:
        sample_metadata = pd.DataFrame(
            {
                "sample_name": list(source_by_name),
                "is_qc": False,
                "is_blank": False,
            }
        )

    if sample_metadata["sample_name"].isna().any():
        raise ValueError("sample_table.csv contains an empty sample_name.")
    sample_metadata["sample_name"] = (
        sample_metadata["sample_name"].astype(str).str.strip()
    )
    if sample_metadata["sample_name"].eq("").any():
        raise ValueError("sample_table.csv contains an empty sample_name.")

    # Downstream feature tables use the first metadata column as the sample
    # identifier. Keep that established table contract explicit here.
    sample_metadata = sample_metadata[
        ["sample_name"]
        + [column for column in sample_metadata.columns if column != "sample_name"]
    ]

    duplicates = sample_metadata.loc[
        sample_metadata["sample_name"].duplicated(keep=False), "sample_name"
    ].unique()
    if len(duplicates) > 0:
        raise ValueError(
            "sample_table.csv contains duplicate sample_name values: "
            + ", ".join(map(str, duplicates))
        )

    for column in ("is_qc", "is_blank"):
        if column in sample_metadata:
            sample_metadata[column] = sample_metadata[column].map(
                _coerce_sample_boolean
            )
        else:
            sample_metadata[column] = False
    return sample_metadata


def _normalize_sample_table_columns(sample_metadata: pd.DataFrame) -> pd.DataFrame:
    normalized_columns = []
    for column in sample_metadata.columns:
        stripped = str(column).strip()
        normalized = stripped.lower()
        normalized_columns.append(
            normalized
            if normalized in {"sample_name", "is_qc", "is_blank"}
            else stripped
        )
    if len(normalized_columns) != len(set(normalized_columns)):
        raise ValueError(
            "sample_table.csv contains duplicate columns after normalization."
        )
    result = sample_metadata.copy()
    result.columns = normalized_columns
    return result


def _attach_source_metadata(
    sample_metadata: pd.DataFrame, source_by_name: Dict[str, str]
) -> pd.DataFrame:
    result = sample_metadata.copy()
    result["VALID"] = False
    result["INVALID_REASON"] = None
    result["ABSOLUTE_PATH"] = None
    result["MS_TYPE"] = None
    result["ION_MODE"] = None
    result["IS_CENTROID"] = None
    result["FILE_FORMAT"] = None
    result["time"] = None

    for index, sample_name in result["sample_name"].items():
        source_path = source_by_name.get(sample_name)
        if source_path is None:
            result.loc[index, "INVALID_REASON"] = "raw data file not found"
            continue

        ms_type, ion_mode, is_centroid, acquisition_time = find_raw_data_info(
            source_path
        )
        result.loc[index, "ABSOLUTE_PATH"] = source_path
        result.loc[index, "MS_TYPE"] = ms_type
        result.loc[index, "ION_MODE"] = ion_mode
        result.loc[index, "IS_CENTROID"] = bool(is_centroid)
        result.loc[index, "FILE_FORMAT"] = get_raw_data_format(source_path)
        result.loc[index, "time"] = acquisition_time
        if is_centroid:
            result.loc[index, "VALID"] = True
        else:
            result.loc[index, "INVALID_REASON"] = (
                "profile-mode raw data are not supported"
            )
    return result


def _coerce_sample_boolean(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        if value in {0, 0.0}:
            return False
        if value in {1, 1.0}:
            return True
        raise ValueError(f"Invalid sample metadata boolean value: {value}")
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1", "y"}:
        return True
    if normalized in {"no", "false", "0", "n", ""}:
        return False
    raise ValueError(f"Invalid sample metadata boolean value: {value}")


def _common_value(values: pd.Series):
    unique = [value for value in pd.unique(values) if pd.notna(value)]
    if not unique:
        return None
    return unique[0] if len(unique) == 1 else "mixed"


@dataclass
class WorkflowState:
    """Mutable execution state and artifacts for one workflow run."""

    run_id: Optional[str] = None
    status: str = "not started"
    current_step: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    processing_metadata: List[Dict[str, Any]] = field(default_factory=list)
    problematic_files: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)

    def start_step(self, step: str):
        self.status = "running"
        self.current_step = step

    def record_error(self, error: Exception):
        self.status = "failed"
        self.errors.append(
            {
                "step": self.current_step or "unknown",
                "type": type(error).__name__,
                "message": str(error),
            }
        )

    def complete(self):
        self.status = "completed"
        self.current_step = None
