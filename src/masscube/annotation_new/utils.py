"""Utilities for in-memory MS/MS spectrum preprocessing and matching."""

from __future__ import annotations

from typing import Optional

import numpy as np


_EMPTY_SPECTRUM = np.empty((0, 2), dtype=np.float32)
_EMPTY_INDICES = np.empty(0, dtype=np.int32)


def clean_spectrum(
    signals: np.ndarray,
    precursor_mz: Optional[float] = None,
    abs_int_tol: Optional[float] = None,
    rel_int_tol: Optional[float] = 0.01,
    precursor_mz_offset: Optional[float] = 2.0,
    min_mz: Optional[float] = None,
    top_n: Optional[int] = None,
) -> np.ndarray:
    """Filter and sort an MS/MS spectrum without modifying the input.

    Non-finite peaks and non-positive intensities are removed first. Absolute
    and relative intensity thresholds are then combined by taking the larger
    cutoff. Peaks at or above ``precursor_mz - precursor_mz_offset`` are
    removed when both precursor arguments are provided.
    """

    spectrum = _as_spectrum(signals)
    if len(spectrum) == 0:
        return spectrum.copy()

    valid = (
        np.isfinite(spectrum[:, 0])
        & np.isfinite(spectrum[:, 1])
        & (spectrum[:, 1] > 0)
    )
    spectrum = spectrum[valid]
    if len(spectrum) == 0:
        return _empty_spectrum_like(signals)

    if min_mz is not None:
        min_mz = _finite_float(min_mz, "min_mz")
        spectrum = spectrum[spectrum[:, 0] >= min_mz]

    if precursor_mz is not None and precursor_mz_offset is not None:
        precursor_mz = _finite_float(precursor_mz, "precursor_mz")
        precursor_mz_offset = _nonnegative_float(precursor_mz_offset, "precursor_mz_offset")
        spectrum = spectrum[spectrum[:, 0] < precursor_mz - precursor_mz_offset]

    if len(spectrum) == 0:
        return _empty_spectrum_like(signals)

    abs_cutoff = 0.0 if abs_int_tol is None else _nonnegative_float(abs_int_tol, "abs_int_tol")
    rel_cutoff = 0.0 if rel_int_tol is None else _nonnegative_float(rel_int_tol, "rel_int_tol")
    intensity_cutoff = max(abs_cutoff, float(np.max(spectrum[:, 1])) * rel_cutoff)
    spectrum = spectrum[spectrum[:, 1] >= intensity_cutoff]
    if len(spectrum) == 0:
        return _empty_spectrum_like(signals)

    if top_n is not None:
        top_n = int(top_n)
        if top_n <= 0:
            return _empty_spectrum_like(signals)
        if len(spectrum) > top_n:
            top_idx = np.argpartition(spectrum[:, 1], -top_n)[-top_n:]
            spectrum = spectrum[top_idx]

    return spectrum[np.argsort(spectrum[:, 0], kind="stable")]


def scale_spectrum(signals: np.ndarray, method: str = "norm_to_max") -> np.ndarray:
    """Apply intensity scaling to a spectrum copy."""

    spectrum = _as_spectrum(signals, copy=True)
    if len(spectrum) == 0:
        return spectrum

    method = str(method).strip().lower()
    if method in {"sqrt", "square_root"}:
        spectrum[:, 1] = np.sqrt(np.maximum(spectrum[:, 1], 0))
    elif method in {"log", "log1p"}:
        spectrum[:, 1] = np.log1p(np.maximum(spectrum[:, 1], 0))
    elif method in {"norm_to_max", "max", "normalize_max"}:
        max_intensity = float(np.max(spectrum[:, 1]))
        if max_intensity > 0:
            spectrum[:, 1] /= max_intensity
    else:
        raise ValueError(f"Unsupported scaling method: {method}")

    return spectrum


def scale_spcectrum(signals: np.ndarray, method: str = "norm_to_max") -> np.ndarray:
    """Compatibility alias for the formerly misspelled function name."""

    return scale_spectrum(signals, method=method)


def normalize_spectrum(signals: np.ndarray, method: str = "max") -> np.ndarray:
    """Normalize fragment intensities in a spectrum copy.

    ``method="max"`` scales the base peak to one, while ``method="sum"``
    scales the total intensity to one.
    """

    spectrum = _as_spectrum(signals, copy=True)
    if len(spectrum) == 0:
        return spectrum

    if not np.all(np.isfinite(spectrum[:, 1])):
        raise ValueError("Spectrum intensities must be finite before normalization.")

    method = str(method).strip().lower()
    if method == "max":
        denominator = float(np.max(spectrum[:, 1]))
    elif method == "sum":
        denominator = float(np.sum(spectrum[:, 1]))
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    if denominator > 0:
        spectrum[:, 1] /= denominator
    return spectrum


def centroid_spectrum(signals: np.ndarray, mz_tol: float = 0.3) -> np.ndarray:
    """Merge nearby fragments whose complete m/z span is within tolerance.

    Input order is irrelevant. The merged m/z is intensity-weighted and the
    merged intensity is the sum of the group.
    """

    mz_tol = _validate_mz_tol(mz_tol)
    spectrum = clean_spectrum(
        signals,
        abs_int_tol=0.0,
        rel_int_tol=None,
        precursor_mz_offset=None,
        min_mz=None,
    )
    if len(spectrum) == 0:
        return spectrum

    centroided = []
    group_start = 0
    for idx in range(1, len(spectrum)):
        if spectrum[idx, 0] - spectrum[group_start, 0] > mz_tol:
            centroided.append(_centroid_signal_group(spectrum[group_start:idx]))
            group_start = idx
    centroided.append(_centroid_signal_group(spectrum[group_start:]))

    centroided = np.asarray(centroided, dtype=np.float32)
    return centroided[np.argsort(centroided[:, 0], kind="stable")]


def align_fragments(
    signals_a: np.ndarray,
    signals_b: np.ndarray,
    mz_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Align fragment peaks one-to-one within an absolute m/z tolerance.

    All candidate pairs within ``mz_tol`` are ranked by descending
    ``intensity_a * intensity_b``, which is their contribution to a cosine
    dot-product numerator. The smaller absolute m/z error breaks equal-
    contribution ties. A peak can be used at most once.

    The returned indices refer to the original input arrays.
    """

    mz_tol = _validate_mz_tol(mz_tol)
    spectrum_a = _as_spectrum(signals_a)
    spectrum_b = _as_spectrum(signals_b)
    if len(spectrum_a) == 0 or len(spectrum_b) == 0:
        return _EMPTY_INDICES.copy(), _EMPTY_INDICES.copy()

    valid_a = (
        np.isfinite(spectrum_a[:, 0])
        & np.isfinite(spectrum_a[:, 1])
        & (spectrum_a[:, 1] > 0)
    )
    valid_b = (
        np.isfinite(spectrum_b[:, 0])
        & np.isfinite(spectrum_b[:, 1])
        & (spectrum_b[:, 1] > 0)
    )
    original_a = np.flatnonzero(valid_a)
    original_b = np.flatnonzero(valid_b)
    if len(original_a) == 0 or len(original_b) == 0:
        return _EMPTY_INDICES.copy(), _EMPTY_INDICES.copy()

    order_a = original_a[np.argsort(spectrum_a[original_a, 0], kind="stable")]
    order_b = original_b[np.argsort(spectrum_b[original_b, 0], kind="stable")]
    mz_a = spectrum_a[order_a, 0]
    mz_b = spectrum_b[order_b, 0]
    intensity_a = spectrum_a[order_a, 1]
    intensity_b = spectrum_b[order_b, 1]

    candidates = []
    for sorted_a, mz in enumerate(mz_a):
        left = np.searchsorted(mz_b, mz - mz_tol, side="left")
        right = np.searchsorted(mz_b, mz + mz_tol, side="right")
        for sorted_b in range(left, right):
            contribution = float(intensity_a[sorted_a]) * float(intensity_b[sorted_b])
            mass_error = abs(float(mz) - float(mz_b[sorted_b]))
            candidates.append((-contribution, mass_error, sorted_a, sorted_b))

    if len(candidates) == 0:
        return _EMPTY_INDICES.copy(), _EMPTY_INDICES.copy()

    candidates.sort()
    used_a = np.zeros(len(order_a), dtype=bool)
    used_b = np.zeros(len(order_b), dtype=bool)
    matched = []

    for _, _, sorted_a, sorted_b in candidates:
        if used_a[sorted_a] or used_b[sorted_b]:
            continue
        used_a[sorted_a] = True
        used_b[sorted_b] = True
        matched.append((int(order_a[sorted_a]), int(order_b[sorted_b])))

    matched.sort(key=lambda pair: pair[0])
    matched_a = np.fromiter((pair[0] for pair in matched), dtype=np.int32)
    matched_b = np.fromiter((pair[1] for pair in matched), dtype=np.int32)
    return matched_a, matched_b


def _as_spectrum(signals: np.ndarray, copy: bool = False) -> np.ndarray:
    """Validate and return a numeric spectrum with shape (n, 2)."""

    spectrum = np.asarray(signals)
    if spectrum.size == 0:
        return _EMPTY_SPECTRUM.copy()
    if spectrum.ndim != 2 or spectrum.shape[1] != 2:
        raise ValueError("A spectrum must be a numeric array with shape (n, 2).")
    if not np.issubdtype(spectrum.dtype, np.number):
        try:
            spectrum = spectrum.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("A spectrum must contain numeric m/z and intensity values.") from exc
    elif not np.issubdtype(spectrum.dtype, np.floating):
        spectrum = spectrum.astype(np.float64)
    elif copy:
        spectrum = spectrum.copy()
    return spectrum


def _centroid_signal_group(signals: np.ndarray) -> tuple[float, float]:
    """Return the intensity-weighted centroid and summed group intensity."""

    intensity_sum = float(np.sum(signals[:, 1]))
    mz = float(np.dot(signals[:, 0], signals[:, 1]) / intensity_sum)
    return mz, intensity_sum


def _validate_mz_tol(mz_tol: float) -> float:
    return _nonnegative_float(mz_tol, "mz_tol")


def _finite_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _nonnegative_float(value: float, name: str) -> float:
    value = _finite_float(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _empty_spectrum_like(signals: np.ndarray) -> np.ndarray:
    spectrum = np.asarray(signals)
    dtype = spectrum.dtype if np.issubdtype(spectrum.dtype, np.floating) else np.float32
    return np.empty((0, 2), dtype=dtype)
