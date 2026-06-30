"""
Utility functions for MS/MS spectrum preprocessing and matching.
"""

import numpy as np


def clean_spectrum(
    signals: np.ndarray,
    precursor_mz: float = None,
    abs_int_tol: float = None,
    rel_int_tol: float = 0.01,
    precursor_mz_offset: float = 2.0,
    min_mz: float = None,
    top_n: int = None,
) -> np.ndarray:
    """
    Clean an MS/MS spectrum using intensity and precursor-related filters.

    Parameters
    ----------
    signals : np.ndarray
        MS/MS spectrum as a 2D array shaped like [[mz, intensity], ...].
    precursor_mz : float, optional
        Precursor m/z. If provided with precursor_mz_offset, fragments at or
        above precursor_mz - precursor_mz_offset are removed.
    abs_int_tol : float, optional
        Absolute intensity cutoff. Fragments below this intensity are removed.
    rel_int_tol : float, default=0.01
        Relative intensity cutoff to the maximum fragment intensity.
    precursor_mz_offset : float, default=2.0
        Offset below precursor m/z used to remove precursor-related peaks. Set
        to None to disable this filter.
    min_mz : float, optional
        Minimum fragment m/z. Fragments below this value are removed.
    top_n : int, optional
        Keep only the top N most intense fragments after other filters.

    Returns
    -------
    np.ndarray
        Cleaned spectrum sorted by m/z.
    """

    if len(signals) == 0:
        return signals

    signals = signals[np.isfinite(signals[:, 0]) & np.isfinite(signals[:, 1])]
    signals = signals[signals[:, 1] > 0]
    if len(signals) == 0:
        return signals

    if min_mz is not None:
        signals = signals[signals[:, 0] >= min_mz]
        if len(signals) == 0:
            return signals

    if precursor_mz is not None and precursor_mz_offset is not None:
        signals = signals[signals[:, 0] < precursor_mz - precursor_mz_offset]
        if len(signals) == 0:
            return signals

    if abs_int_tol is not None:
        signals = signals[signals[:, 1] >= abs_int_tol]
        if len(signals) == 0:
            return signals

    if rel_int_tol is not None and rel_int_tol > 0:
        int_cutoff = np.max(signals[:, 1]) * rel_int_tol
        signals = signals[signals[:, 1] >= int_cutoff]
        if len(signals) == 0:
            return signals

    if top_n is not None:
        top_n = int(top_n)
        if top_n <= 0:
            return signals[:0]
        if len(signals) > top_n:
            top_idx = np.argpartition(signals[:, 1], -top_n)[-top_n:]
            signals = signals[top_idx]

    return signals[np.argsort(signals[:, 0])]


def scale_spcectrum(signals: np.ndarray, method: str = "norm_to_max") -> np.ndarray:
    """
    Scale MS/MS fragment intensities.

    Parameters
    ----------
    signals : np.ndarray
        MS/MS spectrum as a 2D array shaped like [[mz, intensity], ...].
    method : str, default="norm_to_max"
        Scaling method. Supported values are "sqrt", "log", and "norm_to_max".

    Returns
    -------
    np.ndarray
        Spectrum with scaled intensities.
    """

    if len(signals) == 0:
        return signals

    scaled_signals = signals.copy()
    method = method.lower()

    if method in {"sqrt", "square_root"}:
        scaled_signals[:, 1] = np.sqrt(np.maximum(scaled_signals[:, 1], 0))
    elif method in {"log", "log1p"}:
        scaled_signals[:, 1] = np.log1p(np.maximum(scaled_signals[:, 1], 0))
    elif method in {"norm_to_max", "max", "normalize_max"}:
        max_intensity = np.max(scaled_signals[:, 1])
        if max_intensity > 0:
            scaled_signals[:, 1] = scaled_signals[:, 1] / max_intensity
    else:
        raise ValueError("Unsupported scaling method: {}".format(method))

    return scaled_signals


def centroid_spectrum(signals: np.ndarray, mz_tol: float = 0.3) -> np.ndarray:
    """
    Centroid an MS/MS spectrum by merging nearby fragment ions.

    Fragment ions are sorted by m/z and grouped when the m/z range within a
    group is less than or equal to mz_tol. The centroid m/z is
    intensity-weighted, and the centroid intensity is the summed intensity of
    the group.

    Parameters
    ----------
    signals : np.ndarray
        MS/MS spectrum as a 2D array shaped like [[mz, intensity], ...].
    mz_tol : float, default=0.3
        Maximum adjacent m/z difference for grouping fragment ions.

    Returns
    -------
    np.ndarray
        Centroided spectrum sorted by m/z.
    """

    if len(signals) == 0:
        return signals

    signals = signals[np.isfinite(signals[:, 0]) & np.isfinite(signals[:, 1])]
    signals = signals[signals[:, 1] > 0]
    if len(signals) == 0:
        return signals

    signals = signals[np.argsort(signals[:, 0])]
    mz_tol = float(mz_tol)

    centroided = []
    group_start = 0
    for i in range(1, len(signals)):
        if signals[i, 0] - signals[group_start, 0] > mz_tol:
            centroided.append(_centroid_signal_group(signals[group_start:i]))
            group_start = i
    centroided.append(_centroid_signal_group(signals[group_start:]))

    return np.array(centroided, dtype=np.float32)


def align_fragments(
    signals_a: np.ndarray,
    signals_b: np.ndarray,
    mz_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find one-to-one matched fragment ions between two spectra.

    When multiple candidate matches exist, the closest m/z pairs are selected
    first. Each fragment ion can be matched at most once.

    Parameters
    ----------
    signals_a : np.ndarray
        First spectrum as a 2D array shaped like [[mz, intensity], ...].
    signals_b : np.ndarray
        Second spectrum as a 2D array shaped like [[mz, intensity], ...].
    mz_tol : float
        m/z tolerance for fragment matching.

    Returns
    -------
    matched_idx_a : np.ndarray
        Indices of matched fragment ions in signals_a.
    matched_idx_b : np.ndarray
        Indices of matched fragment ions in signals_b.
    """

    if len(signals_a) == 0 or len(signals_b) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    valid_a = np.isfinite(signals_a[:, 0])
    valid_b = np.isfinite(signals_b[:, 0])
    original_idx_a = np.where(valid_a)[0]
    original_idx_b = np.where(valid_b)[0]
    if len(original_idx_a) == 0 or len(original_idx_b) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    order_a = original_idx_a[np.argsort(signals_a[valid_a, 0])]
    order_b = original_idx_b[np.argsort(signals_b[valid_b, 0])]
    mz_a = signals_a[order_a, 0]
    mz_b = signals_b[order_b, 0]
    mz_tol = float(mz_tol)

    candidates = []
    for sorted_i, mz in enumerate(mz_a):
        left = np.searchsorted(mz_b, mz - mz_tol, side="left")
        right = np.searchsorted(mz_b, mz + mz_tol, side="right")
        for sorted_j in range(left, right):
            candidates.append((abs(mz - mz_b[sorted_j]), sorted_i, sorted_j))

    if len(candidates) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    candidates.sort(key=lambda item: item[0])
    used_a = np.zeros(len(signals_a), dtype=bool)
    used_b = np.zeros(len(signals_b), dtype=bool)
    matched = []

    for _, sorted_i, sorted_j in candidates:
        idx_a = order_a[sorted_i]
        idx_b = order_b[sorted_j]
        if used_a[idx_a] or used_b[idx_b]:
            continue
        used_a[idx_a] = True
        used_b[idx_b] = True
        matched.append((idx_a, idx_b))

    matched.sort(key=lambda item: item[0])
    matched_idx_a = np.array([item[0] for item in matched], dtype=np.int32)
    matched_idx_b = np.array([item[1] for item in matched], dtype=np.int32)

    return matched_idx_a, matched_idx_b


def _centroid_signal_group(signals: np.ndarray) -> list[float]:
    """
    Centroid one group of fragment ions.
    """

    intensity_sum = float(np.sum(signals[:, 1]))
    if intensity_sum == 0:
        return [float(np.mean(signals[:, 0])), 0.0]

    mz = float(np.sum(signals[:, 0] * signals[:, 1]) / intensity_sum)
    return [mz, intensity_sum]
