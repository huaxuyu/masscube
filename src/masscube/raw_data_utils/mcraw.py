"""Versioned NumPy-backed raw cache for MassCube workflows.

``.mcraw`` is an analysis cache, not a replacement for the original vendor
archive.  It stores all decoded centroid signals before MassCube's intensity
filtering and centroid repair so feature detection parameters can be changed
without reparsing mzML/TDF2 and weak signals remain available for gap filling.

Metadata shared by all scans, or by all scans at one MS level, is written once
to ``manifest.json``.  Only genuinely scan-varying values (time, precursor m/z,
etc.) are stored as compact typed columns; no per-scan JSON objects are used.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable
from uuid import uuid4

import numpy as np

from ..params import Params
from .core import MSData, Scan, finalize_scan_indexes, preprocess_msdata


FORMAT_NAME = "MassCube mcraw"
FORMAT_VERSION = 1

_FLOAT_SCAN_FIELDS = (
    "precursor_mz",
    "precursor_ion_fraction",
    "tims_pressure",
    "collision_energy",
)
_INT_SCAN_FIELDS = (
    "bruker_frame_id",
    "bruker_precursor_id",
    "bruker_parent_frame_id",
    "bruker_scan_begin",
    "bruker_scan_end",
    "precursor_charge",
)
_BOOL_SCAN_FIELDS = ("mobility_pressure_compensated",)
_PAIR_SCAN_FIELDS = ("isolation_window",)
_PARAM_SNAPSHOT_FIELDS = (
    "scan_levels",
    "mz_lower_limit",
    "mz_upper_limit",
    "rt_lower_limit",
    "rt_upper_limit",
    "centroid_mz_tol",
    "ms1_abs_int_tol",
    "ms2_abs_int_tol",
    "ms2_rel_int_tol",
    "precursor_mz_offset",
    "mz_tol_ms1",
    "mz_tol_ms2",
    "feature_gap_tol",
)
_EXTRA_MSDATA_FIELDS = (
    "tims_metadata",
    "tims_pressure_compensated",
)


def _encode_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return _encode_json(value.item())
    if isinstance(value, datetime):
        return {"__mcraw_type__": "datetime", "value": value.isoformat()}
    if isinstance(value, os.PathLike):
        return {"__mcraw_type__": "path", "value": os.fspath(value)}
    if isinstance(value, tuple):
        return {
            "__mcraw_type__": "tuple",
            "items": [_encode_json(item) for item in value],
        }
    if isinstance(value, list):
        return [_encode_json(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("mcraw JSON dictionaries require string keys")
        return {key: _encode_json(item) for key, item in value.items()}
    raise TypeError(f"unsupported mcraw JSON value: {type(value).__name__}")


def _decode_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_json(item) for item in value]
    if not isinstance(value, dict):
        return value
    marker = value.get("__mcraw_type__")
    if marker == "datetime":
        return datetime.fromisoformat(value["value"])
    if marker == "path":
        return Path(value["value"])
    if marker == "tuple":
        return tuple(_decode_json(item) for item in value["items"])
    return {key: _decode_json(item) for key, item in value.items()}


def _save_array(directory: Path, name: str, value: np.ndarray) -> None:
    with (directory / name).open("wb") as stream:
        np.save(stream, value, allow_pickle=False)


def _load_array(directory: Path, name: str, mmap: bool) -> np.ndarray:
    return np.load(
        directory / name,
        mmap_mode="r" if mmap else None,
        allow_pickle=False,
    )


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _source_fingerprint(source: str | os.PathLike[str] | None) -> dict | None:
    if source is None:
        return None
    path = Path(source).expanduser().absolute()
    if not path.exists():
        return None

    def item_state(item: Path) -> dict:
        stat = item.stat()
        return {
            "name": item.name,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    if path.is_dir() and path.suffix.lower() == ".d":
        members = [path / "analysis.tdf", path / "analysis.tdf_bin"]
        if not all(member.is_file() for member in members):
            return None
        return {
            "kind": "bruker_d",
            "path": os.fspath(path),
            "members": [item_state(member) for member in members],
        }
    if path.is_file():
        return {"kind": "file", "path": os.fspath(path), "members": [item_state(path)]}
    return None


def _same_value(first: Any, second: Any) -> bool:
    if first is None or second is None:
        return first is second
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        try:
            return bool(np.array_equal(np.asarray(first), np.asarray(second), equal_nan=True))
        except (TypeError, ValueError):
            return first == second
    if isinstance(first, float) or isinstance(second, float):
        try:
            if math.isnan(float(first)) and math.isnan(float(second)):
                return True
        except (TypeError, ValueError):
            pass
    return first == second


def _all_same(values: list[Any]) -> bool:
    return not values or all(_same_value(values[0], value) for value in values[1:])


def _encode_scan_column(values: list[Any], kind: str) -> np.ndarray:
    if kind == "float":
        return np.asarray(
            [np.nan if value is None else float(value) for value in values],
            dtype=np.float64,
        )
    if kind == "int":
        missing = np.iinfo(np.int64).min
        return np.asarray(
            [missing if value is None else int(value) for value in values],
            dtype=np.int64,
        )
    if kind == "bool":
        return np.asarray(
            [-1 if value is None else int(bool(value)) for value in values],
            dtype=np.int8,
        )
    if kind == "pair":
        result = np.full((len(values), 2), np.nan, dtype=np.float64)
        for index, value in enumerate(values):
            if value is None:
                continue
            if len(value) != 2:
                raise ValueError("scan pair metadata must contain exactly two values")
            result[index] = (float(value[0]), float(value[1]))
        return result
    raise ValueError(f"unknown mcraw scan-column kind: {kind}")


def _decode_scan_column(value: Any, kind: str) -> Any:
    if kind == "float":
        value = float(value)
        return None if math.isnan(value) else value
    if kind == "int":
        value = int(value)
        return None if value == np.iinfo(np.int64).min else value
    if kind == "bool":
        value = int(value)
        return None if value < 0 else bool(value)
    if kind == "pair":
        first, second = float(value[0]), float(value[1])
        return None if math.isnan(first) or math.isnan(second) else [first, second]
    raise ValueError(f"unknown mcraw scan-column kind: {kind}")


def _uniform_array_dtype(arrays: Iterable[np.ndarray], label: str) -> np.dtype:
    dtypes = {np.asarray(array).dtype.str for array in arrays}
    if not dtypes:
        return np.dtype("<f4")
    if len(dtypes) != 1:
        raise TypeError(f"{label} arrays have mixed dtypes: {sorted(dtypes)}")
    return np.dtype(next(iter(dtypes)))


def _params_snapshot(params: Params | None) -> dict | None:
    if params is None:
        return None
    return {
        name: _encode_json(getattr(params, name))
        for name in _PARAM_SNAPSHOT_FIELDS
        if hasattr(params, name)
    }


def _write_level_metadata(
    directory: Path,
    scans: list[Scan],
    scan_levels: np.ndarray,
) -> dict:
    field_kinds = {
        **{name: "float" for name in _FLOAT_SCAN_FIELDS},
        **{name: "int" for name in _INT_SCAN_FIELDS},
        **{name: "bool" for name in _BOOL_SCAN_FIELDS},
        **{name: "pair" for name in _PAIR_SCAN_FIELDS},
    }
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for level in sorted(set(int(value) for value in scan_levels)):
        positions = np.flatnonzero(scan_levels == level)
        level_result: dict[str, dict[str, Any]] = {}
        for field, kind in field_kinds.items():
            values = [getattr(scans[int(index)], field, None) for index in positions]
            if _all_same(values):
                level_result[field] = {
                    "storage": "constant",
                    "kind": kind,
                    "value": _encode_json(values[0] if values else None),
                }
                continue
            file_name = f"level_{level}_{field}.npy"
            _save_array(directory, file_name, _encode_scan_column(values, kind))
            level_result[field] = {
                "storage": "array",
                "kind": kind,
                "file": file_name,
            }
        result[str(level)] = level_result
    return result


def _write_mcraw_directory(data: MSData, directory: Path) -> None:
    if data.features or data.feature_mz_arr is not None or data.feature_rt_arr is not None:
        raise ValueError("mcraw stores pre-feature-detection MSData only")
    directory.mkdir(parents=True, exist_ok=False)

    scans = list(data.scans)
    scan_count = len(scans)
    scan_levels = np.asarray([int(scan.level) for scan in scans], dtype=np.uint8)
    scan_times = np.asarray(
        [
            float(scan.raw_time if scan.raw_time is not None else scan.time)
            for scan in scans
        ],
        dtype=np.float64,
    )
    raw_file_ids = np.asarray(
        [index if scan.raw_file_id is None else int(scan.raw_file_id) for index, scan in enumerate(scans)],
        dtype=np.int64,
    )

    signal_inputs = [
        np.empty((0, 2), dtype=np.float32)
        if scan.signals is None
        else np.asarray(scan.signals)
        for scan in scans
    ]
    for signals in signal_inputs:
        if signals.ndim != 2 or signals.shape[1] != 2:
            raise ValueError(f"signals must have shape (peaks, 2), got {signals.shape}")
    # Empty scans do not constrain the stored dtype. This keeps an otherwise
    # float64 MSData lossless even when one scan has no peaks.
    dtype_inputs = [signals for signals in signal_inputs if len(signals) > 0]
    signal_dtype = _uniform_array_dtype(dtype_inputs or signal_inputs, "signals")
    offsets = np.empty(scan_count + 1, dtype=np.int64)
    offsets[0] = 0
    for index, signals in enumerate(signal_inputs):
        offsets[index + 1] = offsets[index] + len(signals)
    signals = np.empty((int(offsets[-1]), 2), dtype=signal_dtype)
    for index, source in enumerate(signal_inputs):
        if len(source) > 0 and source.dtype != signal_dtype:
            raise TypeError("signal dtype changed while writing mcraw")
        signals[int(offsets[index]) : int(offsets[index + 1])] = source

    _save_array(directory, "offsets.npy", offsets)
    _save_array(directory, "signals.npy", signals)
    _save_array(directory, "scan_level.npy", scan_levels)
    _save_array(directory, "scan_time.npy", scan_times)
    _save_array(directory, "raw_file_id.npy", raw_file_ids)

    scan_file_names = [scan.file_name for scan in scans]
    if not _all_same(scan_file_names):
        raise ValueError("all scans in one mcraw must share the same file_name")
    mobility_units = [
        scan.inv_mobility_unit
        for scan in scans
        if scan.inv_mobility is not None or scan.inv_mobility_range is not None
    ]
    if not _all_same(mobility_units):
        raise ValueError("all ion-mobility arrays in one mcraw must share one unit")
    common_scan_metadata = {
        "file_name": (
            scan_file_names[0] if scan_file_names else data.metadata.file_name
        ),
        "inv_mobility_unit": (
            mobility_units[0] if mobility_units else data.metadata.mobility_unit
        ),
    }
    level_metadata = _write_level_metadata(directory, scans, scan_levels)

    mobility_inputs = [
        np.asarray(scan.inv_mobility)
        for scan in scans
        if getattr(scan, "inv_mobility", None) is not None
    ]
    mobility_range_inputs = [
        np.asarray(scan.inv_mobility_range)
        for scan in scans
        if getattr(scan, "inv_mobility_range", None) is not None
    ]
    has_mobility = bool(mobility_inputs)
    has_mobility_range = bool(mobility_range_inputs)
    mobility_presence = []
    mobility_range_presence = []
    if has_mobility:
        dtype = _uniform_array_dtype(mobility_inputs, "inv_mobility")
        mobility = np.full(int(offsets[-1]), np.nan, dtype=dtype)
    if has_mobility_range:
        dtype = _uniform_array_dtype(mobility_range_inputs, "inv_mobility_range")
        mobility_range = np.full((int(offsets[-1]), 2), np.nan, dtype=dtype)

    for index, scan in enumerate(scans):
        start, end = int(offsets[index]), int(offsets[index + 1])
        current = getattr(scan, "inv_mobility", None)
        mobility_presence.append(current is not None)
        if current is not None:
            current = np.asarray(current)
            if current.shape != (end - start,):
                raise ValueError("inv_mobility must align with scan.signals")
            mobility[start:end] = current
        current_range = getattr(scan, "inv_mobility_range", None)
        mobility_range_presence.append(current_range is not None)
        if current_range is not None:
            current_range = np.asarray(current_range)
            if current_range.shape != (end - start, 2):
                raise ValueError("inv_mobility_range must align with scan.signals")
            mobility_range[start:end] = current_range

    if has_mobility:
        _save_array(directory, "inv_mobility.npy", mobility)
        _save_array(directory, "inv_mobility_present.npy", np.asarray(mobility_presence, dtype=bool))
    if has_mobility_range:
        _save_array(directory, "inv_mobility_range.npy", mobility_range)
        _save_array(
            directory,
            "inv_mobility_range_present.npy",
            np.asarray(mobility_range_presence, dtype=bool),
        )

    extra_state = {
        name: _encode_json(getattr(data, name))
        for name in _EXTRA_MSDATA_FIELDS
        if hasattr(data, name)
    }
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "complete": True,
        "scan_count": scan_count,
        "peak_count": int(offsets[-1]),
        "signal_dtype": signal_dtype.str,
        "source_fingerprint": _source_fingerprint(data.metadata.file_path),
        "metadata": _encode_json(vars(data.metadata)),
        "common_scan_metadata": _encode_json(common_scan_metadata),
        "level_metadata": level_metadata,
        "has_inv_mobility": has_mobility,
        "has_inv_mobility_range": has_mobility_range,
        "params_snapshot": _params_snapshot(data.params),
        "processing_status": _encode_json(data.processing_status),
        "ms1_mz_calibration_offset": float(data.ms1_mz_calibration_offset),
        "ms2_mz_calibration_offset": float(data.ms2_mz_calibration_offset),
        "extra_msdata_state": extra_state,
    }
    with (directory / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=True)
        stream.write("\n")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def save_mcraw(
    data: MSData,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> int:
    """Atomically save pre-feature-detection ``MSData`` and return its size."""

    target = Path(path)
    if target.suffix.lower() != ".mcraw":
        raise ValueError("mcraw path must end with .mcraw")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(target)

    temporary = target.parent / f".{target.name}.partial-{uuid4().hex}"
    backup = target.parent / f".{target.name}.backup-{uuid4().hex}"
    try:
        _write_mcraw_directory(data, temporary)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        _remove_path(backup)
    finally:
        _remove_path(temporary)
    return _directory_size(target)


def read_mcraw_manifest(path: str | os.PathLike[str]) -> dict:
    directory = Path(path)
    with (directory / "manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format") != FORMAT_NAME:
        raise ValueError(f"not a MassCube mcraw directory: {directory}")
    if manifest.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported mcraw version {manifest.get('version')}; expected {FORMAT_VERSION}"
        )
    if manifest.get("complete") is not True:
        raise ValueError(f"incomplete mcraw directory: {directory}")
    return manifest


def inspect_mcraw(path: str | os.PathLike[str]) -> tuple:
    """Return ``(ms_type, ion_mode, is_centroid, acquisition_time)``."""

    manifest = read_mcraw_manifest(path)
    metadata = _decode_json(manifest["metadata"])
    return (
        metadata.get("ms_type"),
        metadata.get("ion_mode"),
        metadata.get("is_centroid"),
        metadata.get("acquisition_time"),
    )


def mcraw_matches_source(
    path: str | os.PathLike[str], source: str | os.PathLike[str]
) -> bool:
    try:
        manifest = read_mcraw_manifest(path)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return False
    return manifest.get("source_fingerprint") == _source_fingerprint(source)


def _restore_level_metadata(
    directory: Path,
    manifest: dict,
    scans: list[Scan],
    scan_levels: np.ndarray,
    mmap: bool,
    arrays: dict[str, np.ndarray],
) -> None:
    for level_text, fields in manifest["level_metadata"].items():
        level = int(level_text)
        positions = np.flatnonzero(scan_levels == level)
        for field, descriptor in fields.items():
            kind = descriptor["kind"]
            if descriptor["storage"] == "constant":
                value = _decode_json(descriptor["value"])
                for position in positions:
                    setattr(scans[int(position)], field, value.copy() if isinstance(value, list) else value)
                continue
            file_name = descriptor["file"]
            column = _load_array(directory, file_name, mmap)
            arrays[file_name] = column
            if len(column) != len(positions):
                raise ValueError(f"mcraw {file_name} length does not match MS level {level}")
            for local_index, position in enumerate(positions):
                setattr(
                    scans[int(position)],
                    field,
                    _decode_scan_column(column[local_index], kind),
                )


def _params_from_snapshot(snapshot: dict | None) -> Params:
    params = Params()
    if snapshot is not None:
        for name, value in snapshot.items():
            if name in _PARAM_SNAPSHOT_FIELDS:
                setattr(params, name, _decode_json(value))
    return params


def load_mcraw(
    path: str | os.PathLike[str],
    *,
    params: Params | None = None,
    preprocess: bool = False,
    mmap: bool = False,
) -> MSData:
    """Restore ``MSData``; optionally build a parameter-specific analysis view."""

    directory = Path(path)
    manifest = read_mcraw_manifest(directory)
    arrays = {
        "offsets": _load_array(directory, "offsets.npy", mmap),
        "signals": _load_array(directory, "signals.npy", mmap),
        "scan_level": _load_array(directory, "scan_level.npy", mmap),
        "scan_time": _load_array(directory, "scan_time.npy", mmap),
        "raw_file_id": _load_array(directory, "raw_file_id.npy", mmap),
    }
    offsets = arrays["offsets"]
    signals = arrays["signals"]
    scan_levels = arrays["scan_level"]
    scan_times = arrays["scan_time"]
    raw_file_ids = arrays["raw_file_id"]
    scan_count = int(manifest["scan_count"])
    if not (
        len(offsets) == scan_count + 1
        and len(scan_levels) == scan_count
        and len(scan_times) == scan_count
        and len(raw_file_ids) == scan_count
    ):
        raise ValueError("mcraw core scan columns have inconsistent lengths")
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(signals) or np.any(np.diff(offsets) < 0):
        raise ValueError("mcraw peak offsets are invalid")
    if len(signals) != int(manifest["peak_count"]):
        raise ValueError("mcraw peak count does not match manifest")

    data = MSData()
    data.update_metadata(_decode_json(manifest["metadata"]))
    common = _decode_json(manifest["common_scan_metadata"])
    scans: list[Scan] = []
    for index in range(scan_count):
        start, end = int(offsets[index]), int(offsets[index + 1])
        scan = Scan(
            file_name=common.get("file_name"),
            raw_file_id=int(raw_file_ids[index]),
            level=int(scan_levels[index]),
            scan_time=float(scan_times[index]),
            signals=signals[start:end],
        )
        scan.sum_intensity = float(np.sum(scan.signals[:, 1], dtype=np.float64))
        scans.append(scan)

    _restore_level_metadata(
        directory, manifest, scans, scan_levels, mmap, arrays
    )
    if manifest["has_inv_mobility"]:
        mobility = _load_array(directory, "inv_mobility.npy", mmap)
        present = _load_array(directory, "inv_mobility_present.npy", mmap)
        arrays["inv_mobility"] = mobility
        arrays["inv_mobility_present"] = present
        if len(mobility) != len(signals) or len(present) != scan_count:
            raise ValueError("mcraw inverse-mobility arrays are inconsistent")
        for index, scan in enumerate(scans):
            if present[index]:
                scan.inv_mobility = mobility[int(offsets[index]) : int(offsets[index + 1])]
                scan.inv_mobility_unit = common.get("inv_mobility_unit")
    if manifest["has_inv_mobility_range"]:
        mobility_range = _load_array(directory, "inv_mobility_range.npy", mmap)
        present = _load_array(directory, "inv_mobility_range_present.npy", mmap)
        arrays["inv_mobility_range"] = mobility_range
        arrays["inv_mobility_range_present"] = present
        if len(mobility_range) != len(signals) or len(present) != scan_count:
            raise ValueError("mcraw mobility-range arrays are inconsistent")
        for index, scan in enumerate(scans):
            if present[index]:
                scan.inv_mobility_range = mobility_range[
                    int(offsets[index]) : int(offsets[index + 1])
                ]
                scan.inv_mobility_unit = common.get("inv_mobility_unit")

    data.scans = scans
    data.params = params or _params_from_snapshot(manifest.get("params_snapshot"))
    data.processing_status = _decode_json(manifest["processing_status"])
    data.ms1_mz_calibration_offset = float(manifest["ms1_mz_calibration_offset"])
    data.ms2_mz_calibration_offset = float(manifest["ms2_mz_calibration_offset"])
    for name, value in manifest.get("extra_msdata_state", {}).items():
        setattr(data, name, _decode_json(value))
    data.features = []
    data.feature_mz_arr = None
    data.feature_rt_arr = None
    data._mcraw_path = os.fspath(directory.absolute())
    data._mcraw_backing_arrays = arrays
    data._mcraw_mmap = bool(mmap)
    finalize_scan_indexes(data)
    if preprocess:
        preprocess_msdata(data, data.params)
        # Every selected MS1/MS2 signal array has now been materialized by
        # filtering/centroid repair. Drop the full raw backing buffers so a
        # normal analysis object does not retain both representations.
        data._mcraw_backing_arrays = {}
        data._mcraw_mmap = False
    return data


def ensure_mcraw(
    raw_path: str | os.PathLike[str],
    mcraw_path: str | os.PathLike[str],
    params: Params,
    *,
    ms_info=None,
    normalize_tims_intensity: bool = True,
    zstd_library: str | os.PathLike[str] | None = None,
    progress=None,
) -> Path:
    """Create or refresh an unfiltered mcraw cache for one raw data source."""

    raw_path = Path(raw_path).expanduser().absolute()
    target = Path(mcraw_path).expanduser().absolute()
    if target.exists() and mcraw_matches_source(target, raw_path):
        return target

    # Cache all MS1/MS2 scans and peaks.  These are reader-selection settings,
    # not feature-detection settings; the caller's Params remains untouched.
    cache_params = deepcopy(params)
    cache_params.scan_levels = [1, 2]
    cache_params.rt_lower_limit = 0.0
    cache_params.rt_upper_limit = 1.0e12
    cache_params.mz_lower_limit = 0.0
    cache_params.mz_upper_limit = 1.0e12
    cache_params.centroid_mz_tol = None

    from .io import read_raw_file_to_obj

    data = read_raw_file_to_obj(
        raw_path,
        params=cache_params,
        ms_info=ms_info,
        normalize_tims_intensity=normalize_tims_intensity,
        zstd_library=zstd_library,
        progress=progress,
        preprocess=False,
    )
    save_mcraw(data, target, overwrite=target.exists())
    return target


class McrawStore:
    """Lightweight zero-object random-access view over the peak arrays."""

    def __init__(self, path: str | os.PathLike[str], *, mmap: bool = True):
        self.path = Path(path)
        read_mcraw_manifest(self.path)
        self.offsets = _load_array(self.path, "offsets.npy", mmap)
        self.signals = _load_array(self.path, "signals.npy", mmap)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def spectrum(self, index: int) -> np.ndarray:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.signals[start:end]

    def __getitem__(self, index: int) -> np.ndarray:
        return self.spectrum(index)


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "McrawStore",
    "ensure_mcraw",
    "inspect_mcraw",
    "load_mcraw",
    "mcraw_matches_source",
    "read_mcraw_manifest",
    "save_mcraw",
]
