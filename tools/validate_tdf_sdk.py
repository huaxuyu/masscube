#!/usr/bin/env python3
"""Compare MassCube's independent TDF2 reader with Bruker's official TDF-SDK.

Run this script on Windows or Linux with the official SDK dynamic library. It
uses no SDK code at import time and does not make TDF-SDK a MassCube runtime
dependency. The comparison is performed on exact detector coordinates before
MassCube mobility collapse/centroiding.
"""

from __future__ import annotations

from array import array
import argparse
import ctypes
import json
import math
from pathlib import Path
import sys


try:
    from masscube.raw_data_utils._tdf import TDFReader
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from masscube.raw_data_utils._tdf import TDFReader


PRESSURE_STRATEGIES = {
    0: "no_pressure_compensation",
    1: "analysis_global_pressure_compensation",
    2: "per_frame_pressure_compensation",
}


class OfficialTDFSDK:
    """Minimal ctypes binding to the official functions needed for validation."""

    def __init__(
        self,
        data_directory: str | Path,
        library_path: str | Path,
        *,
        use_recalibrated_state: bool = False,
        pressure_strategy: int = 0,
    ):
        self.data_directory = Path(data_directory).expanduser().absolute()
        self.library_path = Path(library_path).expanduser().absolute()
        self._lib = ctypes.CDLL(str(self.library_path))
        self._configure_api()
        self.handle = int(
            self._lib.tims_open_v2(
                str(self.data_directory).encode("utf-8"),
                int(use_recalibrated_state),
                int(pressure_strategy),
            )
        )
        if self.handle == 0:
            self._raise_last_error()

    def _configure_api(self) -> None:
        lib = self._lib
        lib.tims_open_v2.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32]
        lib.tims_open_v2.restype = ctypes.c_uint64
        lib.tims_close.argtypes = [ctypes.c_uint64]
        lib.tims_close.restype = None
        lib.tims_get_last_error_string.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
        lib.tims_get_last_error_string.restype = ctypes.c_uint32
        lib.tims_read_scans_v2.argtypes = [
            ctypes.c_uint64,
            ctypes.c_int64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        lib.tims_read_scans_v2.restype = ctypes.c_uint32
        conversion_args = [
            ctypes.c_uint64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_uint32,
        ]
        lib.tims_index_to_mz.argtypes = conversion_args
        lib.tims_index_to_mz.restype = ctypes.c_uint32
        lib.tims_scannum_to_oneoverk0.argtypes = conversion_args
        lib.tims_scannum_to_oneoverk0.restype = ctypes.c_uint32

    def _last_error(self) -> str:
        length = int(self._lib.tims_get_last_error_string(None, 0))
        if length <= 0:
            return "unknown TDF-SDK error"
        buffer = ctypes.create_string_buffer(length)
        self._lib.tims_get_last_error_string(buffer, length)
        return buffer.value.decode("utf-8", errors="replace")

    def _raise_last_error(self) -> None:
        raise RuntimeError(self._last_error())

    def read_frame(self, frame_id: int, num_scans: int) -> tuple[array, array, array]:
        capacity_words = max(128, num_scans)
        while True:
            buffer = (ctypes.c_uint32 * capacity_words)()
            required_bytes = int(
                self._lib.tims_read_scans_v2(
                    self.handle,
                    int(frame_id),
                    0,
                    int(num_scans),
                    ctypes.cast(buffer, ctypes.c_void_p),
                    capacity_words * 4,
                )
            )
            if required_bytes == 0:
                self._raise_last_error()
            if required_bytes <= capacity_words * 4:
                break
            capacity_words = required_bytes // 4 + 1

        counts = buffer[:num_scans]
        cursor = num_scans
        scans = array("I")
        tof_indices = array("I")
        intensities = array("I")
        for scan_number, count in enumerate(counts):
            count = int(count)
            if count:
                scans.extend(array("I", [scan_number]) * count)
                tof_indices.extend(buffer[cursor : cursor + count])
                cursor += count
                intensities.extend(buffer[cursor : cursor + count])
                cursor += count
        return scans, tof_indices, intensities

    def _convert(self, frame_id: int, values, function) -> array:
        values = tuple(float(value) for value in values)
        count = len(values)
        if count == 0:
            return array("d")
        inputs = (ctypes.c_double * count)(*values)
        outputs = (ctypes.c_double * count)()
        success = int(function(self.handle, int(frame_id), inputs, outputs, count))
        if success == 0:
            self._raise_last_error()
        return array("d", outputs)

    def index_to_mz(self, frame_id: int, tof_indices) -> array:
        return self._convert(frame_id, tof_indices, self._lib.tims_index_to_mz)

    def scan_to_inv_mobility(self, frame_id: int, scans) -> array:
        return self._convert(
            frame_id, scans, self._lib.tims_scannum_to_oneoverk0
        )

    def close(self) -> None:
        if getattr(self, "handle", 0):
            self._lib.tims_close(self.handle)
            self.handle = 0

    def __enter__(self) -> "OfficialTDFSDK":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _quantile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _error_statistics(reference, observed, *, relative_ppm: bool = False) -> dict:
    if len(reference) != len(observed):
        raise ValueError(f"length mismatch: {len(reference)} != {len(observed)}")
    absolute_errors = [abs(float(a) - float(b)) for a, b in zip(reference, observed)]
    absolute_errors.sort()
    result = {
        "count": len(absolute_errors),
        "mean_abs": None,
        "rmse": None,
        "p50_abs": _quantile(absolute_errors, 0.50),
        "p95_abs": _quantile(absolute_errors, 0.95),
        "p99_abs": _quantile(absolute_errors, 0.99),
        "max_abs": None if not absolute_errors else absolute_errors[-1],
    }
    if absolute_errors:
        result["mean_abs"] = sum(absolute_errors) / len(absolute_errors)
        result["rmse"] = math.sqrt(
            sum(value * value for value in absolute_errors) / len(absolute_errors)
        )
    if relative_ppm:
        ppm_errors = [
            abs(float(a) - float(b)) / abs(float(a)) * 1.0e6
            for a, b in zip(reference, observed)
            if float(a) != 0.0
        ]
        ppm_errors.sort()
        result.update(
            {
                "p50_abs_ppm": _quantile(ppm_errors, 0.50),
                "p95_abs_ppm": _quantile(ppm_errors, 0.95),
                "p99_abs_ppm": _quantile(ppm_errors, 0.99),
                "max_abs_ppm": None if not ppm_errors else ppm_errors[-1],
            }
        )
    return result


def _default_frame_ids(reader: TDFReader, count: int = 5) -> list[int]:
    frame_ids = list(reader.frames)
    if len(frame_ids) <= count:
        return frame_ids
    positions = [round(index * (len(frame_ids) - 1) / (count - 1)) for index in range(count)]
    return [frame_ids[position] for position in positions]


def validate(
    data_directory: str | Path,
    sdk_library: str | Path,
    *,
    frame_ids: list[int] | None = None,
    pressure_strategies: tuple[int, ...] = (0, 1, 2),
    use_recalibrated_state: bool = False,
) -> dict:
    """Return a JSON-serializable official-SDK comparison report."""

    report = {
        "data_directory": str(Path(data_directory).expanduser().absolute()),
        "sdk_library": str(Path(sdk_library).expanduser().absolute()),
        "use_recalibrated_state": bool(use_recalibrated_state),
        "pressure_strategies": {
            str(value): PRESSURE_STRATEGIES[value] for value in pressure_strategies
        },
        "frames": [],
        "aggregate": {},
    }
    aggregate_mz_reference = array("d")
    aggregate_mz_observed = array("d")
    aggregate_mobility: dict[int, tuple[array, array]] = {
        strategy: (array("d"), array("d")) for strategy in pressure_strategies
    }
    raw_scan_exact = True
    raw_tof_exact = True
    intensity_exact_vs_raw = True
    intensity_exact_vs_100ms = True

    with TDFReader(data_directory) as independent:
        if frame_ids is None:
            frame_ids = _default_frame_ids(independent)
        missing = [
            frame_id for frame_id in frame_ids if frame_id not in independent.frames
        ]
        if missing:
            raise KeyError(f"unknown frame IDs: {missing}")

        sdk_handles = {
            strategy: OfficialTDFSDK(
                data_directory,
                sdk_library,
                use_recalibrated_state=use_recalibrated_state,
                pressure_strategy=strategy,
            )
            for strategy in pressure_strategies
        }
        try:
            raw_sdk = sdk_handles[pressure_strategies[0]]
            for frame_id in frame_ids:
                metadata = independent.frames[frame_id]
                raw = independent.read_raw_frame(frame_id)
                sdk_scans, sdk_tofs, sdk_intensities = raw_sdk.read_frame(
                    frame_id, metadata.num_scans
                )
                corrected = independent.correct_intensities(frame_id, raw.intensities)
                frame_scan_exact = raw.scans == sdk_scans
                frame_tof_exact = raw.tof_indices == sdk_tofs
                frame_intensity_raw_exact = raw.intensities == sdk_intensities
                frame_intensity_corrected_exact = corrected == sdk_intensities
                raw_scan_exact &= frame_scan_exact
                raw_tof_exact &= frame_tof_exact
                intensity_exact_vs_raw &= frame_intensity_raw_exact
                intensity_exact_vs_100ms &= frame_intensity_corrected_exact

                sdk_mz = raw_sdk.index_to_mz(frame_id, sdk_tofs)
                independent_mz = independent.tof_to_mz(frame_id, raw.tof_indices)
                aggregate_mz_reference.extend(sdk_mz)
                aggregate_mz_observed.extend(independent_mz)

                mobility_comparisons = {}
                scan_axis = range(metadata.num_scans)
                independent_mobility = independent.scan_to_inv_mobility(
                    frame_id, scan_axis
                )
                for strategy, sdk in sdk_handles.items():
                    sdk_mobility = sdk.scan_to_inv_mobility(frame_id, scan_axis)
                    aggregate_mobility[strategy][0].extend(sdk_mobility)
                    aggregate_mobility[strategy][1].extend(independent_mobility)
                    mobility_comparisons[PRESSURE_STRATEGIES[strategy]] = (
                        _error_statistics(
                            sdk_mobility, independent_mobility, relative_ppm=True
                        )
                    )

                report["frames"].append(
                    {
                        "frame_id": frame_id,
                        "msms_type": metadata.msms_type,
                        "time_s": metadata.time_s,
                        "num_scans": metadata.num_scans,
                        "num_peaks": metadata.num_peaks,
                        "raw": {
                            "scan_exact": frame_scan_exact,
                            "tof_exact": frame_tof_exact,
                            "intensity_exact_vs_raw": frame_intensity_raw_exact,
                            "intensity_exact_vs_100ms": frame_intensity_corrected_exact,
                        },
                        "mz": _error_statistics(
                            sdk_mz, independent_mz, relative_ppm=True
                        ),
                        "inv_mobility": mobility_comparisons,
                    }
                )
        finally:
            for sdk in sdk_handles.values():
                sdk.close()

    report["aggregate"] = {
        "raw": {
            "scan_exact": raw_scan_exact,
            "tof_exact": raw_tof_exact,
            "intensity_exact_vs_raw": intensity_exact_vs_raw,
            "intensity_exact_vs_100ms": intensity_exact_vs_100ms,
        },
        "mz": _error_statistics(
            aggregate_mz_reference, aggregate_mz_observed, relative_ppm=True
        ),
        "inv_mobility": {
            PRESSURE_STRATEGIES[strategy]: _error_statistics(
                reference, observed, relative_ppm=True
            )
            for strategy, (reference, observed) in aggregate_mobility.items()
        },
    }
    return report


def _parse_csv_integers(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", help="Bruker .d directory")
    parser.add_argument("sdk_library", help="Official timsdata.dll or libtimsdata.so")
    parser.add_argument(
        "--frames",
        help="Comma-separated frame IDs; default is five evenly spaced frames",
    )
    parser.add_argument(
        "--pressure-strategies",
        default="0,1,2",
        help="Comma-separated SDK strategies: 0=no, 1=global, 2=per-frame",
    )
    parser.add_argument(
        "--recalibrated",
        action="store_true",
        help="Ask the SDK for the newest recalibrated state (independent reader remains raw)",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    frame_ids = (
        None if args.frames is None else list(_parse_csv_integers(args.frames))
    )
    strategies = _parse_csv_integers(args.pressure_strategies)
    invalid = set(strategies).difference(PRESSURE_STRATEGIES)
    if invalid:
        parser.error(f"invalid pressure strategies: {sorted(invalid)}")
    report = validate(
        args.data_directory,
        args.sdk_library,
        frame_ids=frame_ids,
        pressure_strategies=strategies,
        use_recalibrated_state=args.recalibrated,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
