"""Core utilities for tandem mass-spectrum preprocessing and comparison."""

from .similarity_score import dot_product_similarity
from .utils import (
    align_fragments,
    centroid_spectrum,
    clean_spectrum,
    normalize_spectrum,
    scale_spectrum,
)

__all__ = [
    "align_fragments",
    "centroid_spectrum",
    "clean_spectrum",
    "dot_product_similarity",
    "normalize_spectrum",
    "scale_spectrum",
]
