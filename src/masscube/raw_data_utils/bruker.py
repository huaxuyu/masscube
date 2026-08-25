"""Adapt independent TDF2 reading results to MassCube's ``MSData`` model.

This module adds no dependency beyond MassCube itself.  MassCube already
depends on NumPy, and the binary reader in :mod:`._tdf` uses only the
Python standard library plus the system ``libzstd`` runtime.

Mapping from timsTOF's four-dimensional data to MassCube's two-dimensional
``Scan.signals`` representation:

* one precursor/MS1 TIMS frame -> one MassCube level-1 ``Scan``;
* one DDA-PASEF isolation window -> one MassCube level-2 ``Scan``;
* detector events are collapsed by exact TOF index and their intensities are
  summed across the selected mobility scans;
* intensity-weighted mean 1/K0 and observed 1/K0 range remain attached to each
  ``Scan`` as ``inv_mobility`` and ``inv_mobility_range`` arrays aligned with
  ``Scan.signals``.

MassCube's current core algorithms use only ``signals[:, 0:2]`` (m/z and
intensity).  The additional mobility arrays are therefore preserved without
changing existing feature detection behavior.
"""

from __future__ import annotations

from datetime import datetime
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import sqlite3
from typing import Callable, Iterable

import numpy as np

from ..params import Params
from .core import MSData, Scan, finalize_scan_indexes
from ._tdf import (
    PasefWindow,
    RawTimsFrame,
    TDFReader,
    UnsupportedTDFError,
)


ProgressCallback = Callable[[int, int, int], None]


def _array_view(values) -> np.ndarray:
    """Zero-copy NumPy view over a native ``array('I')``."""
    return np.frombuffer(values, dtype=np.uint32)


def _collapse_by_tof(
    reader: TDFReader,
    raw_frame: RawTimsFrame,
    *,
    scan_begin: int = 0,
    scan_end: int | None = None,
    normalize_intensity: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse detector events across mobility scans by exact TOF index."""
    if scan_end is None:
        scan_end = raw_frame.metadata.num_scans

    scans = _array_view(raw_frame.scans)
    tofs = _array_view(raw_frame.tof_indices)
    intensities = _array_view(raw_frame.intensities)
    selected = (scans >= scan_begin) & (scans < scan_end)
    if not np.any(selected):
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy(), np.empty((0, 2), dtype=np.float64)

    selected_scans = scans[selected]
    selected_tofs = tofs[selected]
    selected_intensities = intensities[selected].astype(np.float64)
    if normalize_intensity:
        # Normalize every detector event to a 100 ms accumulation before
        # mobility collapse.  The +0.5 followed by floor reproduces the
        # integer rounding convention used by common TDF readers.
        selected_intensities = np.floor(
            selected_intensities * raw_frame.metadata.intensity_correction + 0.5
        )

    mobility_axis = np.frombuffer(
        reader.scan_to_inv_mobility(
            raw_frame.metadata.frame_id, range(raw_frame.metadata.num_scans)
        ),
        dtype=np.float64,
    )
    selected_mobility = mobility_axis[selected_scans]

    order = np.argsort(selected_tofs, kind="stable")
    sorted_tofs = selected_tofs[order]
    sorted_intensities = selected_intensities[order]
    sorted_mobility = selected_mobility[order]
    unique_tofs, starts = np.unique(sorted_tofs, return_index=True)

    summed_intensity = np.add.reduceat(sorted_intensities, starts)
    weighted_mobility = (
        np.add.reduceat(sorted_mobility * sorted_intensities, starts) / summed_intensity
    )
    mobility_low = np.minimum.reduceat(sorted_mobility, starts)
    mobility_high = np.maximum.reduceat(sorted_mobility, starts)
    mobility_range = np.column_stack((mobility_low, mobility_high))
    mz = np.frombuffer(
        reader.tof_to_mz(raw_frame.metadata.frame_id, unique_tofs), dtype=np.float64
    ).copy()
    return mz, summed_intensity, weighted_mobility, mobility_range


def _filter_and_centroid(
    mz: np.ndarray,
    intensity: np.ndarray,
    mobility: np.ndarray,
    mobility_range: np.ndarray,
    *,
    mz_lower: float,
    mz_upper: float,
    intensity_lower: float,
    centroid_mz_tolerance: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply MassCube-compatible filtering/centroiding while retaining mobility."""
    selected = (
        (mz > mz_lower)
        & (mz < mz_upper)
        & (intensity > intensity_lower)
        & np.isfinite(mz)
        & np.isfinite(intensity)
        & np.isfinite(mobility)
    )
    mz = mz[selected]
    intensity = intensity[selected]
    mobility = mobility[selected]
    mobility_range = mobility_range[selected]
    if mz.size == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )

    order = np.argsort(mz, kind="stable")
    mz = mz[order]
    intensity = intensity[order]
    mobility = mobility[order]
    mobility_range = mobility_range[order]

    if centroid_mz_tolerance is not None and mz.size > 1:
        starts = np.r_[0, np.flatnonzero(np.diff(mz) >= centroid_mz_tolerance) + 1]
        summed_intensity = np.add.reduceat(intensity, starts)
        mz = np.add.reduceat(mz * intensity, starts) / summed_intensity
        mobility = np.add.reduceat(mobility * intensity, starts) / summed_intensity
        mobility_range = np.column_stack(
            (
                np.minimum.reduceat(mobility_range[:, 0], starts),
                np.maximum.reduceat(mobility_range[:, 1], starts),
            )
        )
        intensity = summed_intensity

    signals = np.column_stack((mz, intensity)).astype(np.float32, copy=False)
    return (
        signals,
        mobility.astype(np.float32, copy=False),
        mobility_range.astype(np.float32, copy=False),
    )


def _prepare_masscube_arrays(
    reader: TDFReader,
    raw_frame: RawTimsFrame,
    params: Params,
    *,
    level: int,
    scan_begin: int = 0,
    scan_end: int | None = None,
    precursor_mz: float | None = None,
    normalize_intensity: bool = True,
    preprocess: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mz, intensity, mobility, mobility_range = _collapse_by_tof(
        reader,
        raw_frame,
        scan_begin=scan_begin,
        scan_end=scan_end,
        normalize_intensity=normalize_intensity,
    )
    if not preprocess:
        selected = (
            np.isfinite(mz)
            & np.isfinite(intensity)
            & np.isfinite(mobility)
        )
        mz = mz[selected]
        intensity = intensity[selected]
        mobility = mobility[selected]
        mobility_range = mobility_range[selected]
        order = np.argsort(mz, kind="stable")
        return (
            np.column_stack((mz[order], intensity[order])).astype(
                np.float32, copy=False
            ),
            mobility[order].astype(np.float32, copy=False),
            mobility_range[order].astype(np.float32, copy=False),
        )

    if level == 1:
        mz_upper = float(params.mz_upper_limit)
        intensity_lower = float(params.ms1_abs_int_tol)
    else:
        if intensity.size:
            intensity_lower = max(
                float(params.ms2_abs_int_tol),
                float(np.max(intensity)) * float(params.ms2_rel_int_tol),
            )
        else:
            intensity_lower = float(params.ms2_abs_int_tol)
        if params.precursor_mz_offset is None or precursor_mz is None:
            mz_upper = float(params.mz_upper_limit)
        else:
            mz_upper = min(
                float(params.mz_upper_limit),
                float(precursor_mz) - float(params.precursor_mz_offset),
            )

    return _filter_and_centroid(
        mz,
        intensity,
        mobility,
        mobility_range,
        mz_lower=float(params.mz_lower_limit),
        mz_upper=mz_upper,
        intensity_lower=intensity_lower,
        centroid_mz_tolerance=params.centroid_mz_tol,
    )


def _attach_tims_attributes(
    scan: Scan,
    *,
    frame_id: int,
    mobility: np.ndarray,
    mobility_range: np.ndarray,
    pressure: float | None,
    pasef_window: PasefWindow | None = None,
) -> None:
    """Attach TIMS-specific fields without changing MassCube's base classes."""
    scan.bruker_frame_id = int(frame_id)
    scan.inv_mobility = mobility
    scan.inv_mobility_range = mobility_range
    scan.inv_mobility_unit = "1/K0"
    scan.mobility_pressure_compensated = False
    scan.tims_pressure = pressure
    if pasef_window is not None:
        scan.bruker_precursor_id = pasef_window.precursor_id
        scan.bruker_parent_frame_id = pasef_window.parent_frame_id
        scan.bruker_scan_begin = pasef_window.scan_begin
        scan.bruker_scan_end = pasef_window.scan_end
        scan.collision_energy = pasef_window.collision_energy
        scan.precursor_charge = pasef_window.charge


def _new_masscube_scan(
    *,
    file_name: str,
    raw_file_id: int,
    level: int,
    time_min: float,
    signals: np.ndarray,
    mobility: np.ndarray,
    mobility_range: np.ndarray,
    raw_frame: RawTimsFrame,
    pasef_window: PasefWindow | None = None,
) -> Scan:
    precursor_mz = None if pasef_window is None else pasef_window.precursor_mz
    isolation_window = None
    if pasef_window is not None:
        half_width = pasef_window.isolation_width / 2.0
        isolation_window = [half_width, half_width]

    scan = Scan(
        file_name=file_name,
        raw_file_id=raw_file_id,
        level=level,
        scan_time=time_min,
        signals=signals,
        precursor_mz=precursor_mz,
        isolation_window=isolation_window,
    )
    scan.sum_intensity = float(np.sum(signals[:, 1], dtype=np.float64))
    _attach_tims_attributes(
        scan,
        frame_id=raw_frame.metadata.frame_id,
        mobility=mobility,
        mobility_range=mobility_range,
        pressure=raw_frame.metadata.pressure,
        pasef_window=pasef_window,
    )
    return scan


def _ion_mode(frames: Iterable) -> str:
    polarities = {frame.polarity for frame in frames}
    if polarities == {"+"}:
        return "positive"
    if polarities == {"-"}:
        return "negative"
    return "unknown"


def _acquisition_time(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value


def _initialize_params(params: Params | None, *, ion_mode: str, data_dir: Path) -> Params:
    if params is None:
        params = Params()
        params.set_default("qtof", ion_mode)
    return params


def _finalize_msdata_indexes(data: MSData) -> None:
    finalize_scan_indexes(data)


def inspect_bruker_d(data_directory: str | os.PathLike[str]) -> tuple:
    """Return MassCube metadata without opening or decoding ``analysis.tdf_bin``."""

    data_dir = Path(data_directory).expanduser().absolute()
    tdf_path = data_dir / "analysis.tdf"
    bin_path = data_dir / "analysis.tdf_bin"
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if not tdf_path.is_file():
        raise FileNotFoundError(tdf_path)
    if not bin_path.is_file():
        raise FileNotFoundError(bin_path)

    uri = f"{tdf_path.as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        metadata = dict(connection.execute("SELECT Key, Value FROM GlobalMetadata"))
        polarities = {
            str(row[0]) for row in connection.execute("SELECT DISTINCT Polarity FROM Frames")
        }
    finally:
        connection.close()
    if polarities == {"+"}:
        ion_mode = "positive"
    elif polarities == {"-"}:
        ion_mode = "negative"
    else:
        ion_mode = "unknown"
    return "qtof", ion_mode, True, _acquisition_time(metadata.get("AcquisitionDateTime"))


def read_tims_d_to_msdata(
    data_directory: str | os.PathLike[str],
    params: Params | None = None,
    *,
    include_ms1: bool = True,
    include_ms2: bool = True,
    normalize_intensity: bool = True,
    zstd_library: str | os.PathLike[str] | None = None,
    progress: ProgressCallback | None = None,
    preprocess: bool = True,
) -> MSData:
    """Read a DDA-PASEF ``.d`` directory into a MassCube ``MSData`` object.

    Retention times are converted from TDF seconds to MassCube minutes.  MS1
    frames are mobility-collapsed into one spectrum per frame.  Each
    ``PasefFrameMsMsInfo`` row becomes a separate MS2 spectrum restricted to
    its half-open ``[ScanNumBegin, ScanNumEnd)`` mobility scan interval.
    ``normalize_intensity=True`` applies the conventional per-event
    ``100 / AccumulationTime`` correction before mobility collapse; set it to
    ``False`` to retain the exact integers stored in ``analysis.tdf_bin``.
    """
    data_dir = Path(data_directory).expanduser().absolute()
    with TDFReader(data_dir, zstd_library=zstd_library) as reader:
        ion_mode = _ion_mode(reader.frames.values())
        params = _initialize_params(params, ion_mode=ion_mode, data_dir=data_dir)
        unsupported_msms_types = {
            frame.msms_type for frame in reader.frames.values()
        }.difference({0, 8})
        if include_ms2 and unsupported_msms_types:
            raise UnsupportedTDFError(
                "MassCube adapter currently supports DDA-PASEF MsMsType 8 only; "
                f"found additional types {sorted(unsupported_msms_types)}"
            )

        data = MSData()
        try:
            masscube_version = importlib_metadata.version("masscube")
        except importlib_metadata.PackageNotFoundError:
            masscube_version = None
        data.update_metadata(
            {
                "file_path": os.fspath(data_dir),
                "file_name": data_dir.stem,
                "ms_type": "qtof",
                "ion_mode": ion_mode,
                "is_centroid": True,
                "acquisition_time": _acquisition_time(
                    reader.global_metadata.get("AcquisitionDateTime")
                ),
                "file_format": "d",
                "instrument_name": reader.global_metadata.get("InstrumentName"),
                "has_ion_mobility": True,
                "mobility_unit": "1/K0",
                "mz_calibration_applied": True,
                "mobility_calibration_applied": True,
                "mobility_pressure_compensated": False,
                "masscube_version": masscube_version,
            }
        )
        data.params = params
        data.tims_metadata = dict(reader.global_metadata)
        data.tims_pressure_compensated = False
        data.tims_zstd_library = reader.zstd_library_path

        frame_ids = list(reader.frames)
        total_frames = len(frame_ids)
        for position, frame_id in enumerate(frame_ids, start=1):
            frame_meta = reader.frames[frame_id]
            level = 1 if frame_meta.msms_type == 0 else 2
            if level not in params.scan_levels:
                continue
            if level == 1 and not include_ms1:
                continue
            if level == 2 and not include_ms2:
                continue
            time_min = frame_meta.time_s / 60.0
            if time_min < params.rt_lower_limit or time_min > params.rt_upper_limit:
                continue

            raw_frame = reader.read_raw_frame(frame_id)
            if level == 1:
                signals, mobility, mobility_range = _prepare_masscube_arrays(
                    reader,
                    raw_frame,
                    params,
                    level=1,
                    normalize_intensity=normalize_intensity,
                    preprocess=preprocess,
                )
                data.scans.append(
                    _new_masscube_scan(
                        file_name=data.metadata.file_name,
                        raw_file_id=len(data.scans),
                        level=1,
                        time_min=time_min,
                        signals=signals,
                        mobility=mobility,
                        mobility_range=mobility_range,
                        raw_frame=raw_frame,
                    )
                )
            else:
                windows = reader.get_pasef_windows(frame_id)
                if not windows:
                    raise UnsupportedTDFError(
                        f"MS/MS frame {frame_id} has no PasefFrameMsMsInfo rows"
                    )
                for window in windows:
                    signals, mobility, mobility_range = _prepare_masscube_arrays(
                        reader,
                        raw_frame,
                        params,
                        level=2,
                        scan_begin=window.scan_begin,
                        scan_end=window.scan_end,
                        precursor_mz=window.precursor_mz,
                        normalize_intensity=normalize_intensity,
                        preprocess=preprocess,
                    )
                    data.scans.append(
                        _new_masscube_scan(
                            file_name=data.metadata.file_name,
                            raw_file_id=len(data.scans),
                            level=2,
                            time_min=time_min,
                            signals=signals,
                            mobility=mobility,
                            mobility_range=mobility_range,
                            raw_frame=raw_frame,
                            pasef_window=window,
                        )
                    )

            if progress is not None:
                progress(position, total_frames, frame_id)

        _finalize_msdata_indexes(data)
        return data


def read_bruker_into_msdata(
    data: MSData,
    *,
    normalize_intensity: bool = True,
    zstd_library: str | os.PathLike[str] | None = None,
    progress: ProgressCallback | None = None,
    preprocess: bool = True,
) -> None:
    """Populate an existing ``MSData`` from its configured Bruker ``.d`` path."""

    loaded = read_tims_d_to_msdata(
        data.metadata.file_path,
        params=data.params,
        normalize_intensity=normalize_intensity,
        zstd_library=zstd_library,
        progress=progress,
        preprocess=preprocess,
    )
    data.__dict__.update(loaded.__dict__)


__all__ = ["inspect_bruker_d", "read_bruker_into_msdata", "read_tims_d_to_msdata"]
