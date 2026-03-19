# Author: Huaxu Yu

"""
mzh5.py - HDF5 cache utilities for MassCube

This module converts mzML/mzXML files into a compact hierarchical HDF5 format
for fast reload in downstream workflows.

Layout
------
/meta
    metadata attributes and conversion parameters
/scans
    scan-wise metadata arrays, including offsets into the flattened peak arrays
/peaks
    flattened peak arrays: mz and intensity
/index
    convenience indexes for MS1/MS2 scans
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pyteomics import mzml, mzxml

from .params import Params, find_ms_info
from .raw_data_utils import _preprocess_signals_to_scan, _safe_float


MZH5_FORMAT = "masscube.mzh5"
MZH5_VERSION = 1
_SUPPORTED_RAW_EXT = (".mzml", ".mzxml")


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "h5py is required for mzh5 conversion. Install it with: pip install h5py"
        ) from exc
    return h5py


def _prepare_params(
    file_name: str,
    params: Optional[Params],
    scan_levels: Sequence[int],
    centroid_mz_tol: Optional[float],
    ms1_abs_int_tol: Optional[float],
    ms2_abs_int_tol: Optional[float],
    ms2_rel_int_tol: float,
    precursor_mz_offset: Optional[float],
) -> Params:
    if params is None:
        params = Params()
        ms_type, ion_mode, centroid = find_ms_info(file_name)
        params.scan_levels = list(scan_levels)
        params.centroid_mz_tol = centroid_mz_tol
        params.ms1_abs_int_tol = ms1_abs_int_tol
        params.ms2_abs_int_tol = ms2_abs_int_tol
        params.ms2_rel_int_tol = ms2_rel_int_tol
        params.precursor_mz_offset = precursor_mz_offset
        params.ms_type = ms_type
        params.ion_mode = ion_mode
        params.is_centroid = centroid

    # Follow existing defaults from raw_data_utils.MSData.read_raw_data
    if params.ms1_abs_int_tol is None:
        if params.ms_type == "orbitrap":
            params.ms1_abs_int_tol = 30000
        else:
            params.ms1_abs_int_tol = 1000
    if params.ms2_abs_int_tol is None:
        if params.ms_type == "orbitrap":
            params.ms2_abs_int_tol = 10000
        else:
            params.ms2_abs_int_tol = 500

    return params


def _to_signals(spec) -> np.ndarray:
    mz_arr = spec.get("m/z array")
    int_arr = spec.get("intensity array")
    if mz_arr is None or int_arr is None:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([mz_arr, int_arr], dtype=np.float32).T


def _unit_info(value, default: str = "minute") -> str:
    if value is None:
        return default
    unit = getattr(value, "unit_info", None)
    if unit is None:
        return default
    return str(unit).lower()


def _mzml_scan_time(spec, time_unit) -> float:
    scan = spec.get("scanList", {}).get("scan", [{}])[0]
    scan_time = scan.get("scan start time", scan.get("scan time", 0.0))
    scan_time = float(scan_time)
    if str(time_unit).lower() == "second":
        scan_time /= 60.0
    return scan_time


def _mzxml_scan_time(spec, time_unit) -> float:
    scan_time = float(spec.get("retentionTime", 0.0))
    if str(time_unit).lower() == "second":
        scan_time /= 60.0
    return scan_time


def _get_mzml_ms2_fields(spec) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    precursor_mz = None
    isolation_window = [1.5, 1.5]

    precursor_list = spec.get("precursorList", {}).get("precursor", [])
    if len(precursor_list) == 0:
        return None, None

    precursor = precursor_list[0]
    selected = precursor.get("selectedIonList", {}).get("selectedIon", [])
    if len(selected) > 0:
        precursor_mz = _safe_float(selected[0].get("selected ion m/z"))

    if "isolationWindow" in precursor:
        iw = precursor["isolationWindow"]
        if "isolation window lower offset" in iw and "isolation window upper offset" in iw:
            isolation_window = [
                _safe_float(iw["isolation window lower offset"], default=1.5),
                _safe_float(iw["isolation window upper offset"], default=1.5),
            ]

    return precursor_mz, (float(isolation_window[0]), float(isolation_window[1]))


def _get_mzxml_ms2_fields(spec) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    precursor_mz = None
    isolation_window = [1.5, 1.5]

    precursor = spec.get("precursorMz")
    if precursor is None:
        return None, None

    if isinstance(precursor, list) and len(precursor) > 0:
        precursor_info = precursor[0]
    else:
        precursor_info = precursor

    precursor_mz = _safe_float(getattr(precursor_info, "get", lambda *_: None)("precursorMz"))
    if getattr(precursor_info, "get", None) is not None and "windowWideness" in precursor_info:
        wideness = _safe_float(precursor_info["windowWideness"], default=3.0)
        isolation_window = [wideness / 2.0, wideness / 2.0]

    return precursor_mz, (float(isolation_window[0]), float(isolation_window[1]))


def _should_process_scan(level: int, scan_time: float, params: Params) -> bool:
    return (
        level in params.scan_levels
        and params.rt_lower_limit <= scan_time <= params.rt_upper_limit
    )


def _default_output_path(file_name: str) -> str:
    root, _ = os.path.splitext(file_name)
    return root + ".mzh5"


def _write_hdf5(
    output_path: str,
    source_file: str,
    source_format: str,
    params: Params,
    scan_level_arr: np.ndarray,
    scan_time_arr: np.ndarray,
    precursor_arr: np.ndarray,
    isolation_arr: np.ndarray,
    peak_start_arr: np.ndarray,
    peak_end_arr: np.ndarray,
    ms1_scan_idx_arr: np.ndarray,
    ms2_scan_idx_arr: np.ndarray,
    ms1_time_arr: np.ndarray,
    peak_mz_arr: np.ndarray,
    peak_int_arr: np.ndarray,
    compression: Optional[str],
    compression_opts: Optional[int],
) -> None:
    h5py = _require_h5py()

    peak_ds_kwargs = {}
    if compression is not None:
        peak_ds_kwargs["compression"] = compression
        if compression_opts is not None:
            peak_ds_kwargs["compression_opts"] = compression_opts
        peak_ds_kwargs["shuffle"] = True

    with h5py.File(output_path, "w") as h5:
        h5.attrs["format"] = MZH5_FORMAT
        h5.attrs["version"] = MZH5_VERSION
        h5.attrs["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        g_meta = h5.create_group("meta")
        g_meta.attrs["source_file"] = os.path.abspath(source_file)
        g_meta.attrs["source_format"] = source_format
        g_meta.attrs["scan_levels"] = np.array(params.scan_levels, dtype=np.int8)
        g_meta.attrs["ion_mode"] = str(params.ion_mode)
        g_meta.attrs["ms_type"] = str(params.ms_type)
        g_meta.attrs["is_centroid"] = bool(params.is_centroid)
        g_meta.attrs["centroid_mz_tol"] = (
            np.nan if params.centroid_mz_tol is None else float(params.centroid_mz_tol)
        )
        g_meta.attrs["ms1_abs_int_tol"] = float(params.ms1_abs_int_tol)
        g_meta.attrs["ms2_abs_int_tol"] = float(params.ms2_abs_int_tol)
        g_meta.attrs["ms2_rel_int_tol"] = float(params.ms2_rel_int_tol)
        g_meta.attrs["precursor_mz_offset"] = (
            np.nan if params.precursor_mz_offset is None else float(params.precursor_mz_offset)
        )
        g_meta.attrs["rt_lower_limit"] = float(params.rt_lower_limit)
        g_meta.attrs["rt_upper_limit"] = float(params.rt_upper_limit)

        g_scans = h5.create_group("scans")
        g_scans.create_dataset("id", data=np.arange(scan_level_arr.size, dtype=np.int32))
        g_scans.create_dataset("level", data=scan_level_arr)
        g_scans.create_dataset("time", data=scan_time_arr)
        g_scans.create_dataset("precursor_mz", data=precursor_arr)
        g_scans.create_dataset("isolation_window", data=isolation_arr)
        g_scans.create_dataset("peak_start", data=peak_start_arr)
        g_scans.create_dataset("peak_end", data=peak_end_arr)

        g_peaks = h5.create_group("peaks")
        g_peaks.create_dataset("mz", data=peak_mz_arr, **peak_ds_kwargs)
        g_peaks.create_dataset("intensity", data=peak_int_arr, **peak_ds_kwargs)

        g_index = h5.create_group("index")
        g_index.create_dataset("ms1_scan_idx", data=ms1_scan_idx_arr)
        g_index.create_dataset("ms2_scan_idx", data=ms2_scan_idx_arr)
        g_index.create_dataset("ms1_time", data=ms1_time_arr)


def convert_raw_to_mzh5(
    file_name: str,
    output_path: Optional[str] = None,
    params: Optional[Params] = None,
    scan_levels: Sequence[int] = (1, 2),
    centroid_mz_tol: Optional[float] = 0.005,
    ms1_abs_int_tol: Optional[float] = None,
    ms2_abs_int_tol: Optional[float] = None,
    ms2_rel_int_tol: float = 0.01,
    precursor_mz_offset: Optional[float] = 2.0,
    compression: Optional[str] = "lzf",
    compression_opts: Optional[int] = None,
) -> str:
    """
    Convert a mzML/mzXML file to MassCube HDF5 cache format (mzh5).

    Parameters
    ----------
    file_name : str
        Input mzML or mzXML file path.
    output_path : str, optional
        Output mzh5 path. If None, uses `<input_basename>.mzh5`.
    params : Params, optional
        MassCube parameter object. If None, defaults are inferred from file metadata.
    scan_levels : sequence of int
        Scan levels to keep (default: (1, 2)).
    centroid_mz_tol : float or None
        m/z tolerance for centroiding. Use None to disable.
    ms1_abs_int_tol : float, optional
        MS1 absolute intensity threshold.
    ms2_abs_int_tol : float, optional
        MS2 absolute intensity threshold.
    ms2_rel_int_tol : float
        MS2 relative intensity threshold.
    precursor_mz_offset : float or None
        Remove precursor region in MS2 using `precursor_mz - offset`.
    compression : str or None
        Compression for peak arrays. Typical values: "lzf", "gzip", or None.
    compression_opts : int, optional
        Compression level for codecs that support it.

    Returns
    -------
    str
        Output mzh5 file path.
    """

    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"Raw file does not exist: {file_name}")

    ext = os.path.splitext(file_name)[1].lower()
    if ext not in _SUPPORTED_RAW_EXT:
        raise ValueError("Unsupported raw data format. Expected mzML or mzXML.")

    params = _prepare_params(
        file_name=file_name,
        params=params,
        scan_levels=scan_levels,
        centroid_mz_tol=centroid_mz_tol,
        ms1_abs_int_tol=ms1_abs_int_tol,
        ms2_abs_int_tol=ms2_abs_int_tol,
        ms2_rel_int_tol=ms2_rel_int_tol,
        precursor_mz_offset=precursor_mz_offset,
    )

    if output_path is None:
        output_path = _default_output_path(file_name)

    scan_level: List[int] = []
    scan_time: List[float] = []
    precursor_mz: List[float] = []
    isolation_window: List[Tuple[float, float]] = []
    peak_start: List[int] = []
    peak_end: List[int] = []

    ms1_scan_idx: List[int] = []
    ms2_scan_idx: List[int] = []
    ms1_time: List[float] = []

    peak_mz_chunks: List[np.ndarray] = []
    peak_int_chunks: List[np.ndarray] = []
    peak_cursor = 0

    if ext == ".mzml":
        with mzml.MzML(file_name) as reader:
            scan0 = reader[0]
            scan0_meta = scan0.get("scanList", {}).get("scan", [{}])[0]
            time_unit = _unit_info(
                scan0_meta.get("scan start time", scan0_meta.get("scan time")),
                default="minute",
            )

            for idx, spec in enumerate(reader):
                level = int(spec.get("ms level", 0))
                time_min = _mzml_scan_time(spec, time_unit)
                precursor, iso = (None, None)
                if level == 2:
                    precursor, iso = _get_mzml_ms2_fields(spec)
                signals = _to_signals(spec)

                if _should_process_scan(level, time_min, params):
                    s = _preprocess_signals_to_scan(
                        level=level,
                        id=idx,
                        scan_time=time_min,
                        signals=signals,
                        params=params,
                        precursor_mz=precursor,
                        isolation_window=None if iso is None else [iso[0], iso[1]],
                    )
                    cleaned = s.signals
                else:
                    cleaned = np.empty((0, 2), dtype=np.float32)

                n = 0 if cleaned is None else int(cleaned.shape[0])
                peak_start.append(peak_cursor)
                peak_cursor += n
                peak_end.append(peak_cursor)

                scan_level.append(level)
                scan_time.append(float(time_min))
                precursor_mz.append(np.nan if precursor is None else float(precursor))
                if iso is None:
                    isolation_window.append((np.nan, np.nan))
                else:
                    isolation_window.append((float(iso[0]), float(iso[1])))

                if n > 0:
                    peak_mz_chunks.append(cleaned[:, 0].astype(np.float32, copy=False))
                    peak_int_chunks.append(cleaned[:, 1].astype(np.float32, copy=False))
                    if level == 1:
                        ms1_scan_idx.append(idx)
                        ms1_time.append(float(time_min))
                    elif level == 2:
                        ms2_scan_idx.append(idx)

    elif ext == ".mzxml":
        with mzxml.MzXML(file_name) as reader:
            scan0 = reader[0]
            time_unit = _unit_info(scan0.get("retentionTime"), default="minute")

            for idx, spec in enumerate(reader):
                level = int(spec.get("msLevel", 0))
                time_min = _mzxml_scan_time(spec, time_unit)
                precursor, iso = (None, None)
                if level == 2:
                    precursor, iso = _get_mzxml_ms2_fields(spec)
                signals = _to_signals(spec)

                if _should_process_scan(level, time_min, params):
                    s = _preprocess_signals_to_scan(
                        level=level,
                        id=idx,
                        scan_time=time_min,
                        signals=signals,
                        params=params,
                        precursor_mz=precursor,
                        isolation_window=None if iso is None else [iso[0], iso[1]],
                    )
                    cleaned = s.signals
                else:
                    cleaned = np.empty((0, 2), dtype=np.float32)

                n = 0 if cleaned is None else int(cleaned.shape[0])
                peak_start.append(peak_cursor)
                peak_cursor += n
                peak_end.append(peak_cursor)

                scan_level.append(level)
                scan_time.append(float(time_min))
                precursor_mz.append(np.nan if precursor is None else float(precursor))
                if iso is None:
                    isolation_window.append((np.nan, np.nan))
                else:
                    isolation_window.append((float(iso[0]), float(iso[1])))

                if n > 0:
                    peak_mz_chunks.append(cleaned[:, 0].astype(np.float32, copy=False))
                    peak_int_chunks.append(cleaned[:, 1].astype(np.float32, copy=False))
                    if level == 1:
                        ms1_scan_idx.append(idx)
                        ms1_time.append(float(time_min))
                    elif level == 2:
                        ms2_scan_idx.append(idx)

    scan_level_arr = np.asarray(scan_level, dtype=np.int8)
    scan_time_arr = np.asarray(scan_time, dtype=np.float32)
    precursor_arr = np.asarray(precursor_mz, dtype=np.float32)
    isolation_arr = np.asarray(isolation_window, dtype=np.float32)
    peak_start_arr = np.asarray(peak_start, dtype=np.int64)
    peak_end_arr = np.asarray(peak_end, dtype=np.int64)
    ms1_scan_idx_arr = np.asarray(ms1_scan_idx, dtype=np.int32)
    ms2_scan_idx_arr = np.asarray(ms2_scan_idx, dtype=np.int32)
    ms1_time_arr = np.asarray(ms1_time, dtype=np.float32)

    if len(peak_mz_chunks) > 0:
        peak_mz_arr = np.concatenate(peak_mz_chunks, dtype=np.float32)
        peak_int_arr = np.concatenate(peak_int_chunks, dtype=np.float32)
    else:
        peak_mz_arr = np.empty(0, dtype=np.float32)
        peak_int_arr = np.empty(0, dtype=np.float32)

    _write_hdf5(
        output_path=output_path,
        source_file=file_name,
        source_format=ext[1:],
        params=params,
        scan_level_arr=scan_level_arr,
        scan_time_arr=scan_time_arr,
        precursor_arr=precursor_arr,
        isolation_arr=isolation_arr,
        peak_start_arr=peak_start_arr,
        peak_end_arr=peak_end_arr,
        ms1_scan_idx_arr=ms1_scan_idx_arr,
        ms2_scan_idx_arr=ms2_scan_idx_arr,
        ms1_time_arr=ms1_time_arr,
        peak_mz_arr=peak_mz_arr,
        peak_int_arr=peak_int_arr,
        compression=compression,
        compression_opts=compression_opts,
    )

    return output_path


def batch_convert_raw_to_mzh5(
    file_names: Sequence[str],
    output_dir: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """
    Convert multiple mzML/mzXML files to mzh5.

    Parameters
    ----------
    file_names : sequence of str
        Input raw file paths.
    output_dir : str, optional
        Output directory for mzh5 files. If None, outputs next to each raw file.
    **kwargs
        Extra parameters forwarded to `convert_raw_to_mzh5`.

    Returns
    -------
    list of str
        Output mzh5 paths.
    """

    outputs: List[str] = []
    for fn in file_names:
        if output_dir is None:
            out = None
        else:
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(fn))[0]
            out = os.path.join(output_dir, base + ".mzh5")
        outputs.append(convert_raw_to_mzh5(file_name=fn, output_path=out, **kwargs))
    return outputs
