"""Independent Bruker TDF2 reader for macOS/Linux/Windows.

The module reads ``analysis.tdf`` with Python's standard-library SQLite
driver and decodes ``analysis.tdf_bin`` itself.  It does not import OpenTIMS,
AlphaTims, NumPy, pandas, or Bruker's TDF-SDK.

Python 3.11 does not include a ZSTD decoder, so compressed frame payloads are
decoded through the public C API of ``libzstd`` using ``ctypes``.  A caller can
pass an explicit library path or set ``MASSCUBE_ZSTD_LIBRARY``.  Homebrew's
usual macOS paths and common Linux/Windows names are searched automatically.

Supported acquisition/calibration variants:

* TimsCompressionType 2 (TDF2/ZSTD)
* MzCalibration ModelType 1 with dC2 == 0 and C3/C4 == 0
* TimsCalibration ModelType 2 (static physical 1/K0 calibration)

The stored static mobility calibration is applied exactly as represented by
the calibration coefficients.  Proprietary per-frame pressure compensation is
not applied; raw scan numbers are always retained so a different calibration
can be applied later without rereading the binary data.
"""

from __future__ import annotations

from array import array
import ctypes
import ctypes.util
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sqlite3
import struct
import sys
from typing import Iterable, Iterator, Sequence


class TDFError(RuntimeError):
    """Base class for TDF reader errors."""


class UnsupportedTDFError(TDFError):
    """Raised when a file uses an unsupported compression/calibration model."""


class CorruptTDFError(TDFError):
    """Raised when metadata and binary frame contents are inconsistent."""


class ZstdUnavailableError(TDFError):
    """Raised when no usable system libzstd can be loaded."""


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Metadata needed to locate and interpret one TIMS frame."""

    frame_id: int
    time_s: float
    polarity: str
    scan_mode: int
    msms_type: int
    binary_offset: int | None
    num_scans: int
    num_peaks: int
    mz_calibration_id: int
    tims_calibration_id: int
    t1: float
    t2: float
    accumulation_time_ms: float
    pressure: float | None

    @property
    def intensity_correction(self) -> float:
        """Bruker's conventional normalization to a 100 ms accumulation."""
        if self.accumulation_time_ms <= 0.0:
            raise CorruptTDFError(
                f"frame {self.frame_id} has invalid AccumulationTime={self.accumulation_time_ms}"
            )
        return 100.0 / self.accumulation_time_ms


@dataclass(frozen=True, slots=True)
class PasefWindow:
    """One DDA-PASEF isolation window inside an MS/MS frame."""

    frame_id: int
    scan_begin: int
    scan_end: int
    isolation_mz: float
    isolation_width: float
    collision_energy: float
    precursor_id: int | None
    largest_peak_mz: float | None
    average_mz: float | None
    monoisotopic_mz: float | None
    charge: int | None
    precursor_scan_number: float | None
    precursor_intensity: float | None
    parent_frame_id: int | None

    @property
    def precursor_mz(self) -> float:
        """Best stored precursor m/z, falling back to the isolation center."""
        for value in (self.monoisotopic_mz, self.largest_peak_mz, self.average_mz):
            if value is not None and math.isfinite(value):
                return value
        return self.isolation_mz


@dataclass(slots=True)
class RawTimsFrame:
    """Exact detector events decoded from one ``analysis.tdf_bin`` frame."""

    metadata: FrameMetadata
    scans: array
    tof_indices: array
    intensities: array

    def __len__(self) -> int:
        return len(self.tof_indices)

    def iter_peaks(self) -> Iterator[tuple[int, int, int]]:
        """Yield ``(scan, tof_index, intensity)`` tuples."""
        return zip(self.scans, self.tof_indices, self.intensities)


@dataclass(slots=True)
class CalibratedTimsFrame:
    """Raw events plus physical m/z and static inverse-mobility coordinates."""

    metadata: FrameMetadata
    scans: array
    tof_indices: array
    intensities: array
    mz_values: array
    inv_mobility_values: array
    pressure_compensated: bool = False

    def __len__(self) -> int:
        return len(self.tof_indices)

    def iter_peaks(self) -> Iterator[tuple[int, int, int, float, float]]:
        """Yield ``(scan, tof, intensity, mz, 1/K0)`` tuples."""
        return zip(
            self.scans,
            self.tof_indices,
            self.intensities,
            self.mz_values,
            self.inv_mobility_values,
        )


@dataclass(frozen=True, slots=True)
class _MzCalibration:
    calibration_id: int
    model_type: int
    digitizer_timebase: float
    digitizer_delay: float
    reference_t1: float
    reference_t2: float
    dc1: float
    dc2: float
    c0: float
    c1: float
    c2: float
    c3: float
    c4: float

    def coefficients_for_frame(
        self, frame: FrameMetadata
    ) -> tuple[float, float, float, float, float]:
        if self.model_type != 1:
            raise UnsupportedTDFError(
                f"MzCalibration ModelType={self.model_type} is unsupported; expected 1"
            )
        if self.dc2 != 0.0:
            raise UnsupportedTDFError(
                "nonzero MzCalibration.dC2 is not implemented because its vendor "
                "correction semantics cannot be verified without the TDF-SDK"
            )
        if self.c3 != 0.0 or self.c4 != 0.0:
            raise UnsupportedTDFError(
                "nonzero MzCalibration.C3/C4 coefficients are unsupported"
            )
        correction = 1.0 + self.dc1 * (self.reference_t1 - frame.t1) / 1.0e6
        effective_c1 = self.c1 * correction
        if effective_c1 <= 0.0:
            raise CorruptTDFError(
                f"MzCalibration {self.calibration_id} has nonpositive corrected C1"
            )
        return (
            self.digitizer_timebase,
            self.digitizer_delay,
            self.c0,
            effective_c1,
            self.c2,
        )


@dataclass(frozen=True, slots=True)
class _TimsCalibration:
    calibration_id: int
    model_type: int
    c0: float
    c1: float
    c2: float
    c3: float
    c4: float
    c6: float
    c7: float

    def slope_and_offset(self) -> tuple[float, float]:
        if self.model_type != 2:
            raise UnsupportedTDFError(
                f"TimsCalibration ModelType={self.model_type} is unsupported; expected 2"
            )
        if self.c1 == 0.0:
            raise CorruptTDFError("TimsCalibration.C1 cannot be zero")
        slope = (self.c3 - self.c2) / self.c1
        offset = self.c2 - slope * (self.c4 + self.c0)
        return slope, offset


class _ZstdDecoder:
    """Small ctypes binding for the stable public libzstd decompression API."""

    _CONTENTSIZE_UNKNOWN = (1 << 64) - 2
    _CONTENTSIZE_ERROR = (1 << 64) - 1

    def __init__(self, library_path: str | os.PathLike[str] | None = None):
        self.library_path, self._lib = self._load_library(library_path)
        self._configure_api()

    @staticmethod
    def _candidate_libraries(
        explicit_path: str | os.PathLike[str] | None,
    ) -> Iterator[str]:
        seen: set[str] = set()

        candidates = [
            os.fspath(explicit_path) if explicit_path is not None else None,
            os.environ.get("MASSCUBE_ZSTD_LIBRARY"),
            ctypes.util.find_library("zstd"),
        ]
        if sys.platform == "darwin":
            candidates.extend(
                [
                    "/opt/homebrew/lib/libzstd.dylib",
                    "/usr/local/lib/libzstd.dylib",
                    "/opt/local/lib/libzstd.dylib",
                    "libzstd.dylib",
                ]
            )
        elif os.name == "nt":
            candidates.extend(["libzstd.dll", "zstd.dll"])
        else:
            candidates.extend(["libzstd.so.1", "libzstd.so"])

        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                yield candidate

    @classmethod
    def _load_library(
        cls, explicit_path: str | os.PathLike[str] | None
    ) -> tuple[str, ctypes.CDLL]:
        failures: list[str] = []
        for candidate in cls._candidate_libraries(explicit_path):
            try:
                return candidate, ctypes.CDLL(candidate)
            except OSError as exc:
                failures.append(f"{candidate}: {exc}")

        details = "\n".join(failures[-4:])
        raise ZstdUnavailableError(
            "No usable libzstd was found. Python 3.11 has no standard-library "
            "ZSTD decoder. Install zstd (for example `brew install zstd`) or set "
            "MASSCUBE_ZSTD_LIBRARY to an explicit libzstd path.\n" + details
        )

    def _configure_api(self) -> None:
        lib = self._lib
        lib.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
        lib.ZSTD_decompress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.ZSTD_decompress.restype = ctypes.c_size_t
        lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        lib.ZSTD_isError.restype = ctypes.c_uint
        lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        lib.ZSTD_getErrorName.restype = ctypes.c_char_p
        if hasattr(lib, "ZSTD_decompressBound"):
            lib.ZSTD_decompressBound.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            lib.ZSTD_decompressBound.restype = ctypes.c_ulonglong

    def _error_name(self, code: int) -> str:
        value = self._lib.ZSTD_getErrorName(code)
        return value.decode("utf-8", errors="replace") if value else f"code {code}"

    def decompress(self, compressed: bytes) -> bytes:
        if not compressed:
            raise CorruptTDFError("empty ZSTD frame payload")

        source = ctypes.create_string_buffer(compressed)
        source_size = len(compressed)
        output_size = int(self._lib.ZSTD_getFrameContentSize(source, source_size))
        if output_size == self._CONTENTSIZE_ERROR:
            raise CorruptTDFError("invalid ZSTD frame header")
        if output_size == self._CONTENTSIZE_UNKNOWN:
            if not hasattr(self._lib, "ZSTD_decompressBound"):
                raise CorruptTDFError("ZSTD frame content size is unknown")
            output_size = int(self._lib.ZSTD_decompressBound(source, source_size))
            if self._lib.ZSTD_isError(output_size):
                raise CorruptTDFError(
                    "cannot determine ZSTD decompression bound: " + self._error_name(output_size)
                )
        if output_size <= 0:
            raise CorruptTDFError(f"invalid ZSTD output size: {output_size}")

        destination = ctypes.create_string_buffer(output_size)
        actual_size = int(
            self._lib.ZSTD_decompress(destination, output_size, source, source_size)
        )
        if self._lib.ZSTD_isError(actual_size):
            raise CorruptTDFError("ZSTD decompression failed: " + self._error_name(actual_size))
        return destination.raw[:actual_size]


class TDFReader:
    """Read TDF2 detector events and stored physical calibration parameters."""

    def __init__(
        self,
        data_directory: str | os.PathLike[str],
        *,
        zstd_library: str | os.PathLike[str] | None = None,
    ):
        self.data_directory = Path(data_directory).expanduser().resolve()
        self.tdf_path = self.data_directory / "analysis.tdf"
        self.bin_path = self.data_directory / "analysis.tdf_bin"
        if not self.data_directory.is_dir():
            raise FileNotFoundError(self.data_directory)
        if not self.tdf_path.is_file():
            raise FileNotFoundError(self.tdf_path)
        if not self.bin_path.is_file():
            raise FileNotFoundError(self.bin_path)
        if array("I").itemsize != 4:
            raise RuntimeError("this reader requires a platform with 32-bit unsigned int arrays")

        uri = f"{self.tdf_path.as_uri()}?mode=ro&immutable=1"
        self._connection = sqlite3.connect(uri, uri=True)
        self._connection.row_factory = sqlite3.Row
        try:
            self.global_metadata = self._read_global_metadata()
            compression_type = int(self.global_metadata["TimsCompressionType"])
            if compression_type != 2:
                raise UnsupportedTDFError(
                    f"TimsCompressionType={compression_type} is unsupported; expected 2"
                )
            self.frames = self._read_frames()
            self._mz_calibrations = self._read_mz_calibrations()
            self._tims_calibrations = self._read_tims_calibrations()
            self.pasef_windows = self._read_pasef_windows()
            self._pasef_by_frame = self._group_pasef_windows(self.pasef_windows)
            self._validate_referenced_calibrations()
            self._zstd = _ZstdDecoder(zstd_library)
            self._binary = self.bin_path.open("rb")
        except Exception:
            self._connection.close()
            raise

    @property
    def zstd_library_path(self) -> str:
        return self._zstd.library_path

    @property
    def total_peaks(self) -> int:
        return sum(frame.num_peaks for frame in self.frames.values())

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(self.frames)

    def _read_global_metadata(self) -> dict[str, str]:
        rows = self._connection.execute("SELECT Key, Value FROM GlobalMetadata")
        return {str(row["Key"]): str(row["Value"]) for row in rows}

    def _read_frames(self) -> dict[int, FrameMetadata]:
        query = """
            SELECT Id, Time, Polarity, ScanMode, MsMsType, TimsId, NumScans,
                   NumPeaks, MzCalibration, TimsCalibration, T1, T2,
                   AccumulationTime, Pressure
            FROM Frames ORDER BY Id
        """
        result: dict[int, FrameMetadata] = {}
        for row in self._connection.execute(query):
            frame = FrameMetadata(
                frame_id=int(row["Id"]),
                time_s=float(row["Time"]),
                polarity=str(row["Polarity"]),
                scan_mode=int(row["ScanMode"]),
                msms_type=int(row["MsMsType"]),
                binary_offset=None if row["TimsId"] is None else int(row["TimsId"]),
                num_scans=int(row["NumScans"]),
                num_peaks=int(row["NumPeaks"]),
                mz_calibration_id=int(row["MzCalibration"]),
                tims_calibration_id=int(row["TimsCalibration"]),
                t1=float(row["T1"]),
                t2=float(row["T2"]),
                accumulation_time_ms=float(row["AccumulationTime"]),
                pressure=None if row["Pressure"] is None else float(row["Pressure"]),
            )
            result[frame.frame_id] = frame
        if not result:
            raise CorruptTDFError("Frames table is empty")
        return result

    def _read_mz_calibrations(self) -> dict[int, _MzCalibration]:
        query = """
            SELECT Id, ModelType, DigitizerTimebase, DigitizerDelay,
                   T1, T2, dC1, dC2, C0, C1, C2, C3, C4
            FROM MzCalibration ORDER BY Id
        """
        result: dict[int, _MzCalibration] = {}
        for row in self._connection.execute(query):
            required = (
                "Id",
                "ModelType",
                "DigitizerTimebase",
                "DigitizerDelay",
                "T1",
                "dC1",
                "C0",
                "C1",
            )
            missing = [name for name in required if row[name] is None]
            if missing:
                raise CorruptTDFError("MzCalibration missing: " + ", ".join(missing))
            calibration = _MzCalibration(
                calibration_id=int(row["Id"]),
                model_type=int(row["ModelType"]),
                digitizer_timebase=float(row["DigitizerTimebase"]),
                digitizer_delay=float(row["DigitizerDelay"]),
                reference_t1=float(row["T1"]),
                reference_t2=0.0 if row["T2"] is None else float(row["T2"]),
                dc1=float(row["dC1"]),
                dc2=0.0 if row["dC2"] is None else float(row["dC2"]),
                c0=float(row["C0"]),
                c1=float(row["C1"]),
                c2=0.0 if row["C2"] is None else float(row["C2"]),
                c3=0.0 if row["C3"] is None else float(row["C3"]),
                c4=0.0 if row["C4"] is None else float(row["C4"]),
            )
            result[calibration.calibration_id] = calibration
        return result

    def _read_tims_calibrations(self) -> dict[int, _TimsCalibration]:
        query = """
            SELECT Id, ModelType, C0, C1, C2, C3, C4, C6, C7
            FROM TimsCalibration ORDER BY Id
        """
        result: dict[int, _TimsCalibration] = {}
        for row in self._connection.execute(query):
            required = ("Id", "ModelType", "C0", "C1", "C2", "C3", "C4", "C6", "C7")
            missing = [name for name in required if row[name] is None]
            if missing:
                raise CorruptTDFError("TimsCalibration missing: " + ", ".join(missing))
            calibration = _TimsCalibration(
                calibration_id=int(row["Id"]),
                model_type=int(row["ModelType"]),
                c0=float(row["C0"]),
                c1=float(row["C1"]),
                c2=float(row["C2"]),
                c3=float(row["C3"]),
                c4=float(row["C4"]),
                c6=float(row["C6"]),
                c7=float(row["C7"]),
            )
            result[calibration.calibration_id] = calibration
        return result

    def _table_exists(self, table_name: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
        ).fetchone()
        return row is not None

    def _read_pasef_windows(self) -> tuple[PasefWindow, ...]:
        if not self._table_exists("PasefFrameMsMsInfo"):
            return ()
        query = """
            SELECT p.Frame, p.ScanNumBegin, p.ScanNumEnd, p.IsolationMz,
                   p.IsolationWidth, p.CollisionEnergy, p.Precursor,
                   q.LargestPeakMz, q.AverageMz, q.MonoisotopicMz, q.Charge,
                   q.ScanNumber, q.Intensity, q.Parent
            FROM PasefFrameMsMsInfo AS p
            LEFT JOIN Precursors AS q ON q.Id = p.Precursor
            ORDER BY p.Frame, p.ScanNumBegin
        """
        windows: list[PasefWindow] = []
        for row in self._connection.execute(query):
            window = PasefWindow(
                    frame_id=int(row["Frame"]),
                    scan_begin=int(row["ScanNumBegin"]),
                    scan_end=int(row["ScanNumEnd"]),
                    isolation_mz=float(row["IsolationMz"]),
                    isolation_width=float(row["IsolationWidth"]),
                    collision_energy=float(row["CollisionEnergy"]),
                    precursor_id=None if row["Precursor"] is None else int(row["Precursor"]),
                    largest_peak_mz=None if row["LargestPeakMz"] is None else float(row["LargestPeakMz"]),
                    average_mz=None if row["AverageMz"] is None else float(row["AverageMz"]),
                    monoisotopic_mz=None if row["MonoisotopicMz"] is None else float(row["MonoisotopicMz"]),
                    charge=None if row["Charge"] is None else int(row["Charge"]),
                    precursor_scan_number=None if row["ScanNumber"] is None else float(row["ScanNumber"]),
                    precursor_intensity=None if row["Intensity"] is None else float(row["Intensity"]),
                    parent_frame_id=None if row["Parent"] is None else int(row["Parent"]),
                )
            frame = self.frames.get(window.frame_id)
            if frame is None:
                raise CorruptTDFError(
                    f"PasefFrameMsMsInfo references missing frame {window.frame_id}"
                )
            if not 0 <= window.scan_begin < window.scan_end <= frame.num_scans:
                raise CorruptTDFError(
                    f"invalid PASEF scan range [{window.scan_begin}, {window.scan_end}) "
                    f"for frame {window.frame_id} with {frame.num_scans} scans"
                )
            windows.append(window)
        return tuple(windows)

    @staticmethod
    def _group_pasef_windows(
        windows: Sequence[PasefWindow],
    ) -> dict[int, tuple[PasefWindow, ...]]:
        grouped: dict[int, list[PasefWindow]] = {}
        for window in windows:
            grouped.setdefault(window.frame_id, []).append(window)
        return {frame_id: tuple(items) for frame_id, items in grouped.items()}

    def _validate_referenced_calibrations(self) -> None:
        for frame in self.frames.values():
            try:
                mz_calibration = self._mz_calibrations[frame.mz_calibration_id]
                tims_calibration = self._tims_calibrations[frame.tims_calibration_id]
            except KeyError as exc:
                raise CorruptTDFError(
                    f"frame {frame.frame_id} references missing calibration {exc.args[0]}"
                ) from exc
            mz_calibration.coefficients_for_frame(frame)
            tims_calibration.slope_and_offset()

    def get_pasef_windows(self, frame_id: int) -> tuple[PasefWindow, ...]:
        return self._pasef_by_frame.get(int(frame_id), ())

    def read_raw_frame(self, frame_id: int) -> RawTimsFrame:
        """Decode one frame's exact ``scan/tof/intensity`` detector events."""
        frame_id = int(frame_id)
        try:
            metadata = self.frames[frame_id]
        except KeyError as exc:
            raise KeyError(f"unknown frame_id={frame_id}") from exc

        if metadata.num_peaks == 0:
            return RawTimsFrame(metadata, array("I"), array("I"), array("I"))
        if metadata.binary_offset is None:
            raise CorruptTDFError(f"frame {frame_id} has peaks but no TimsId offset")

        self._binary.seek(metadata.binary_offset)
        header = self._binary.read(8)
        if len(header) != 8:
            raise CorruptTDFError(f"truncated binary header for frame {frame_id}")
        binary_size, scan_count = struct.unpack("<II", header)
        if binary_size < 8:
            raise CorruptTDFError(f"invalid binary size {binary_size} for frame {frame_id}")
        if scan_count != metadata.num_scans:
            raise CorruptTDFError(
                f"frame {frame_id}: binary has {scan_count} scans, metadata has {metadata.num_scans}"
            )
        compressed = self._binary.read(binary_size - 8)
        if len(compressed) != binary_size - 8:
            raise CorruptTDFError(f"truncated compressed payload for frame {frame_id}")
        decompressed = self._zstd.decompress(compressed)
        return self._parse_type2_frame(metadata, decompressed)

    @staticmethod
    def _decode_byte_planes(decompressed: bytes) -> array:
        if len(decompressed) % 4:
            raise CorruptTDFError("TDF2 decompressed payload is not divisible into uint32 words")
        word_count = len(decompressed) // 4
        interleaved = bytearray(len(decompressed))
        interleaved[0::4] = decompressed[:word_count]
        interleaved[1::4] = decompressed[word_count : 2 * word_count]
        interleaved[2::4] = decompressed[2 * word_count : 3 * word_count]
        interleaved[3::4] = decompressed[3 * word_count :]
        words = array("I")
        words.frombytes(interleaved)
        if sys.byteorder != "little":
            words.byteswap()
        return words

    @classmethod
    def _parse_type2_frame(
        cls, metadata: FrameMetadata, decompressed: bytes
    ) -> RawTimsFrame:
        words = cls._decode_byte_planes(decompressed)
        scan_count = metadata.num_scans
        peak_count = metadata.num_peaks
        expected_words = scan_count + 2 * peak_count
        if len(words) != expected_words:
            raise CorruptTDFError(
                f"frame {metadata.frame_id}: decoded {len(words)} words; expected {expected_words}"
            )
        if not words or words[0] != scan_count:
            actual = None if not words else words[0]
            raise CorruptTDFError(
                f"frame {metadata.frame_id}: payload scan marker is {actual}; expected {scan_count}"
            )

        encoded_counts = array("I", (words[index] // 2 for index in range(scan_count)))
        tof_indices = array("I", words[scan_count::2])
        intensities = array("I", words[scan_count + 1 :: 2])
        scan_counts = array("I", encoded_counts[1:])
        final_count = peak_count - sum(encoded_counts[1:])
        if final_count < 0:
            raise CorruptTDFError(f"negative final scan peak count in frame {metadata.frame_id}")
        scan_counts.append(final_count)
        if len(scan_counts) != scan_count or sum(scan_counts) != peak_count:
            raise CorruptTDFError(f"invalid scan peak counts in frame {metadata.frame_id}")

        scans = array("I")
        peak_index = 0
        for scan_number, count in enumerate(scan_counts):
            if count:
                scans.extend(array("I", [scan_number]) * count)
            cumulative_tof = 0
            for _ in range(count):
                cumulative_tof += tof_indices[peak_index]
                if cumulative_tof == 0:
                    raise CorruptTDFError(
                        f"zero cumulative TOF in frame {metadata.frame_id}, scan {scan_number}"
                    )
                tof_indices[peak_index] = cumulative_tof - 1
                peak_index += 1

        if peak_index != peak_count or len(intensities) != peak_count:
            raise CorruptTDFError(f"peak count mismatch in frame {metadata.frame_id}")
        return RawTimsFrame(metadata, scans, tof_indices, intensities)

    def tof_to_mz(self, frame_id: int, tof_indices: Iterable[int]) -> array:
        frame = self.frames[int(frame_id)]
        calibration = self._mz_calibrations[frame.mz_calibration_id]
        timebase, delay, c0, effective_c1, c2 = (
            calibration.coefficients_for_frame(frame)
        )
        linear_coefficient = math.sqrt(1.0e12 / effective_c1)

        def convert(index: int) -> float:
            flight_time = float(index) * timebase + delay - c0
            if c2 == 0.0:
                sqrt_mz = flight_time / linear_coefficient
            else:
                discriminant = linear_coefficient**2 + 4.0 * c2 * flight_time
                if discriminant < 0.0:
                    raise CorruptTDFError(
                        f"frame {frame.frame_id} has a negative m/z calibration discriminant"
                    )
                denominator = linear_coefficient + math.sqrt(discriminant)
                if denominator == 0.0:
                    raise CorruptTDFError(
                        f"frame {frame.frame_id} has a singular m/z calibration"
                    )
                # Stable positive root of
                # C2 * sqrt(m/z)^2 + A * sqrt(m/z) - flight_time = 0.
                sqrt_mz = 2.0 * flight_time / denominator
            return sqrt_mz * sqrt_mz

        return array(
            "d",
            (convert(index) for index in tof_indices),
        )

    def scan_to_inv_mobility(self, frame_id: int, scans: Iterable[int]) -> array:
        frame = self.frames[int(frame_id)]
        calibration = self._tims_calibrations[frame.tims_calibration_id]
        slope, offset = calibration.slope_and_offset()
        return array(
            "d",
            (
                1.0 / (calibration.c6 + calibration.c7 / (offset + slope * float(scan)))
                for scan in scans
            ),
        )

    def correct_intensities(self, frame_id: int, intensities: Iterable[int]) -> array:
        """Normalize raw intensities to 100 ms and round like common TDF readers.

        Values returned by :meth:`read_raw_frame` remain the exact integers
        stored in ``analysis.tdf_bin``.  This method performs the separate
        accumulation-time correction used for cross-frame comparisons.
        """
        correction = self.frames[int(frame_id)].intensity_correction
        return array("I", (int(float(value) * correction + 0.5) for value in intensities))

    def calibrate_frame(self, raw_frame: RawTimsFrame) -> CalibratedTimsFrame:
        frame_id = raw_frame.metadata.frame_id
        return CalibratedTimsFrame(
            metadata=raw_frame.metadata,
            scans=raw_frame.scans,
            tof_indices=raw_frame.tof_indices,
            intensities=raw_frame.intensities,
            mz_values=self.tof_to_mz(frame_id, raw_frame.tof_indices),
            inv_mobility_values=self.scan_to_inv_mobility(frame_id, raw_frame.scans),
            pressure_compensated=False,
        )

    def read_frame(self, frame_id: int) -> CalibratedTimsFrame:
        return self.calibrate_frame(self.read_raw_frame(frame_id))

    def iter_raw_frames(
        self, frame_ids: Iterable[int] | None = None
    ) -> Iterator[RawTimsFrame]:
        if frame_ids is None:
            frame_ids = self.frames
        for frame_id in frame_ids:
            yield self.read_raw_frame(int(frame_id))

    def iter_frames(
        self, frame_ids: Iterable[int] | None = None
    ) -> Iterator[CalibratedTimsFrame]:
        for raw_frame in self.iter_raw_frames(frame_ids):
            yield self.calibrate_frame(raw_frame)

    def close(self) -> None:
        binary = getattr(self, "_binary", None)
        if binary is not None:
            binary.close()
            self._binary = None
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None

    def __enter__(self) -> "TDFReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    "CalibratedTimsFrame",
    "CorruptTDFError",
    "FrameMetadata",
    "PasefWindow",
    "RawTimsFrame",
    "TDFError",
    "TDFReader",
    "UnsupportedTDFError",
    "ZstdUnavailableError",
]
