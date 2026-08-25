"""Raw-format detection, metadata inspection, and public read dispatch."""

from __future__ import annotations

import os
from pathlib import Path

from ..params import Params
from ..utils_functions import find_ms_info
from .core import MSData, UNSUPPORTED_RAW_FORMAT_MESSAGE


_UNSET = object()


def get_raw_data_format(file_path: str | os.PathLike[str]) -> str:
    """Return ``'mzml'``, ``'d'`` or ``'mcraw'`` after path validation."""

    path = Path(file_path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".mzml" and path.is_file():
        return "mzml"
    if suffix == ".mcraw" and path.is_dir():
        from .mcraw import read_mcraw_manifest

        read_mcraw_manifest(path)
        return "mcraw"
    if suffix == ".d" and path.is_dir():
        if not (path / "analysis.tdf").is_file():
            raise FileNotFoundError(path / "analysis.tdf")
        if not (path / "analysis.tdf_bin").is_file():
            raise FileNotFoundError(path / "analysis.tdf_bin")
        return "d"
    if not path.exists():
        raise FileNotFoundError(f"Raw data path not found: {path}")
    raise ValueError(UNSUPPORTED_RAW_FORMAT_MESSAGE)


def is_supported_raw_data_path(file_path: str | os.PathLike[str]) -> bool:
    """Return whether a directory entry looks like a supported raw-data source."""

    path = Path(file_path)
    if path.name.startswith("."):
        return False
    if path.suffix.lower() == ".mzml":
        return path.is_file()
    if path.suffix.lower() == ".mcraw":
        if not path.is_dir():
            return False
        try:
            from .mcraw import read_mcraw_manifest

            read_mcraw_manifest(path)
            return True
        except (FileNotFoundError, OSError, ValueError):
            return False
    if path.suffix.lower() == ".d":
        return (
            path.is_dir()
            and (path / "analysis.tdf").is_file()
            and (path / "analysis.tdf_bin").is_file()
        )
    return False


def find_raw_data_info(file_path: str | os.PathLike[str]) -> tuple:
    """Return ``(ms_type, ion_mode, is_centroid, acquisition_time)``."""

    raw_format = get_raw_data_format(file_path)
    if raw_format == "mcraw":
        from .mcraw import inspect_mcraw

        return inspect_mcraw(file_path)
    if raw_format == "d":
        from .bruker import inspect_bruker_d

        return inspect_bruker_d(file_path)
    return find_ms_info(os.fspath(file_path))


def read_raw_data_into(
    data: MSData,
    file_path: str | os.PathLike[str],
    params: Params,
    *,
    normalize_tims_intensity: bool = True,
    zstd_library: str | os.PathLike[str] | None = None,
    progress=None,
    ms_info=None,
    preprocess: bool = True,
) -> None:
    """Populate an existing ``MSData`` using the reader selected by path type."""

    raw_format = get_raw_data_format(file_path)
    path = Path(file_path).expanduser().absolute()
    if raw_format == "mcraw":
        from .mcraw import load_mcraw

        # Parameter-specific preprocessing materializes compact per-scan
        # arrays. mmap avoids holding the full unfiltered peak buffer in RAM
        # at the same time.
        loaded = load_mcraw(
            path,
            params=params,
            preprocess=preprocess,
            mmap=preprocess,
        )
        data.__dict__.update(loaded.__dict__)
        return
    if ms_info is None:
        ms_info = find_raw_data_info(path)
    ms_type, ion_mode, is_centroid, acquisition_time = ms_info
    data.update_metadata(
        {
            "file_path": os.fspath(path),
            "file_format": raw_format,
            "ms_type": ms_type,
            "ion_mode": ion_mode,
            "is_centroid": is_centroid,
            "acquisition_time": acquisition_time,
        }
    )
    data.params = params
    data.scans = []
    data.ms1_idx_arr = []
    data.ms2_idx_arr = []
    data.ms1_time_arr = []
    data.base_peak_arr = []

    if raw_format == "mzml":
        from .mzml import read_mzml_into_msdata

        read_mzml_into_msdata(data, preprocess=preprocess)
    else:
        from .bruker import read_bruker_into_msdata

        read_bruker_into_msdata(
            data,
            normalize_intensity=normalize_tims_intensity,
            zstd_library=zstd_library,
            progress=progress,
            preprocess=preprocess,
        )


def read_raw_file_to_obj(
    file_name,
    params=None,
    ms1_abs_int_tol=1000,
    ms2_abs_int_tol=0,
    *,
    scan_levels=None,
    centroid_mz_tol=_UNSET,
    ms2_rel_int_tol=None,
    precursor_mz_offset=_UNSET,
    ms_info=None,
    normalize_tims_intensity=True,
    zstd_library=None,
    progress=None,
    preprocess=True,
):
    """Read mzML, Bruker TDF2 ``.d`` or a MassCube ``.mcraw`` cache.

    When ``params`` is supplied it remains authoritative.  Convenience
    thresholds are applied only when this function creates a new ``Params``.
    TIMS intensities are normalized to a 100 ms accumulation by default.
    ``preprocess=False`` retains decoded peaks before MassCube intensity
    filtering and centroid repair and is intended for mcraw creation.
    """

    path = Path(file_name).expanduser().absolute()
    raw_format = get_raw_data_format(path)
    if ms_info is None:
        ms_info = find_raw_data_info(path)
    ms_type, ion_mode, is_centroid, acquisition_time = ms_info

    if params is None:
        params = Params()
        params.ms1_abs_int_tol = ms1_abs_int_tol
        params.ms2_abs_int_tol = ms2_abs_int_tol
        if scan_levels is not None:
            params.scan_levels = list(scan_levels)
        if centroid_mz_tol is not _UNSET:
            params.centroid_mz_tol = centroid_mz_tol
        if ms2_rel_int_tol is not None:
            params.ms2_rel_int_tol = ms2_rel_int_tol
        if precursor_mz_offset is not _UNSET:
            params.precursor_mz_offset = precursor_mz_offset

    data = MSData()
    data.update_metadata(
        {
            "file_path": os.fspath(path),
            "file_format": raw_format,
            "ms_type": ms_type,
            "ion_mode": ion_mode,
            "is_centroid": is_centroid,
            "acquisition_time": acquisition_time,
            "scan_time_unit": "minute",
        }
    )
    read_raw_data_into(
        data,
        path,
        params,
        normalize_tims_intensity=normalize_tims_intensity,
        zstd_library=zstd_library,
        progress=progress,
        ms_info=ms_info,
        preprocess=preprocess,
    )
    return data


__all__ = [
    "find_raw_data_info",
    "get_raw_data_format",
    "is_supported_raw_data_path",
    "read_raw_data_into",
    "read_raw_file_to_obj",
]
