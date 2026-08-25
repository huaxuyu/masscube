"""Centroid mzML input for :mod:`masscube.raw_data_utils`."""

from __future__ import annotations

import numpy as np

from .core import MSData, Scan, finalize_scan_indexes


def read_mzml_into_msdata(data: MSData, *, preprocess: bool = True) -> None:
    """Read ``data.metadata.file_path`` and populate an existing ``MSData``."""

    from pyteomics import mzml

    with mzml.MzML(data.metadata.file_path) as reader:
        extract_scans_mzml(data, reader, preprocess=preprocess)


def _scan_time_minutes(scan_info: dict) -> float:
    if "scan start time" in scan_info:
        value = scan_info["scan start time"]
    elif "scan time" in scan_info:
        value = scan_info["scan time"]
    else:
        raise ValueError("mzML spectrum has no scan start time")

    scan_time = float(value)
    unit = str(getattr(value, "unit_info", "") or "").lower()
    if unit in {"second", "seconds", "s"}:
        scan_time /= 60.0
    return scan_time


def _precursor_information(spec: dict) -> tuple[float | None, list[float] | None]:
    precursor_list = spec.get("precursorList", {}).get("precursor", [])
    if not precursor_list:
        return None, None

    precursor = precursor_list[0]
    selected_ions = precursor.get("selectedIonList", {}).get("selectedIon", [])
    precursor_mz = None
    if selected_ions:
        value = selected_ions[0].get("selected ion m/z")
        if value is not None:
            precursor_mz = float(value)

    isolation_window = None
    window = precursor.get("isolationWindow", {})
    lower = window.get("isolation window lower offset")
    upper = window.get("isolation window upper offset")
    if lower is not None and upper is not None:
        isolation_window = [float(lower), float(upper)]
    return precursor_mz, isolation_window


def extract_scans_mzml(data: MSData, scans, *, preprocess: bool = True) -> None:
    """Convert an iterable of pyteomics mzML spectra into MassCube scans."""

    for raw_index, spec in enumerate(scans):
        scan_list = spec.get("scanList", {}).get("scan", [])
        if not scan_list:
            continue
        scan_time = _scan_time_minutes(scan_list[0])
        level = int(spec["ms level"])
        if level not in data.params.scan_levels:
            continue
        if not data.params.rt_lower_limit <= scan_time <= data.params.rt_upper_limit:
            continue

        if "m/z array" in spec and "intensity array" in spec:
            signals = np.column_stack((spec["m/z array"], spec["intensity array"])).astype(
                np.float32, copy=False
            )
        else:
            signals = np.empty((0, 2), dtype=np.float32)

        precursor_mz = None
        isolation_window = None
        if level == 2:
            precursor_mz, isolation_window = _precursor_information(spec)

        scan = Scan(
            file_name=data.metadata.file_name,
            raw_file_id=raw_index,
            level=level,
            scan_time=scan_time,
            signals=signals,
            precursor_mz=precursor_mz,
            isolation_window=isolation_window,
        )
        if preprocess:
            scan.preprocess_signals(data.params)
        scan.sum_intensity = float(np.sum(scan.signals[:, 1], dtype=np.float64))
        data.scans.append(scan)

    finalize_scan_indexes(data)


__all__ = ["extract_scans_mzml", "read_mzml_into_msdata"]
