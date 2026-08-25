"""Core in-memory similarity scoring for tandem mass spectra."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import _as_spectrum, align_fragments


def dot_product_similarity(
    query_spectrum: np.ndarray,
    library_spectrum: np.ndarray,
    mz_tol: float,
    query_norm_l2: Optional[float] = None,
    library_norm_l2: Optional[float] = None,
) -> tuple[float, int]:
    """Calculate cosine dot-product similarity between two MS/MS spectra.

    Peaks are aligned one-to-one within ``mz_tol``. Ambiguous candidate pairs
    are selected by their intensity-product contribution to the dot product,
    with the smaller m/z error used as the tie breaker.

    Parameters
    ----------
    query_spectrum, library_spectrum
        Query and library spectra shaped as ``[[mz, intensity], ...]``.
    mz_tol
        Absolute fragment m/z tolerance in Da.
    query_norm_l2, library_norm_l2
        Optional precomputed L2 norms of the positive, finite intensities.

    Returns
    -------
    similarity
        Cosine similarity in the range [0, 1].
    matched_peak_number
        Number of one-to-one fragment matches used in the score.
    """

    query_spectrum = _as_spectrum(query_spectrum)
    library_spectrum = _as_spectrum(library_spectrum)
    if len(query_spectrum) == 0 or len(library_spectrum) == 0:
        return 0.0, 0

    if query_norm_l2 is None:
        query_norm_l2 = _positive_intensity_norm(query_spectrum)
    if library_norm_l2 is None:
        library_norm_l2 = _positive_intensity_norm(library_spectrum)

    query_norm_l2 = float(query_norm_l2)
    library_norm_l2 = float(library_norm_l2)
    if (
        not np.isfinite(query_norm_l2)
        or not np.isfinite(library_norm_l2)
        or query_norm_l2 <= 0
        or library_norm_l2 <= 0
    ):
        return 0.0, 0

    matched_query, matched_library = align_fragments(query_spectrum, library_spectrum, mz_tol)
    if len(matched_query) == 0:
        return 0.0, 0

    numerator = float(
        np.dot(
            query_spectrum[matched_query, 1],
            library_spectrum[matched_library, 1],
        )
    )
    similarity = numerator / (query_norm_l2 * library_norm_l2)

    # Protect callers from tiny floating-point excursions outside [0, 1].
    similarity = float(np.clip(similarity, 0.0, 1.0))
    return similarity, int(len(matched_query))


def _positive_intensity_norm(spectrum: np.ndarray) -> float:
    """Return the L2 norm of finite positive peaks in a spectrum."""

    valid = (
        np.isfinite(spectrum[:, 0])
        & np.isfinite(spectrum[:, 1])
        & (spectrum[:, 1] > 0)
    )
    return float(np.linalg.norm(spectrum[valid, 1]))
