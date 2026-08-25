import numpy as np
import pytest

from masscube.annotation_new import (
    align_fragments,
    centroid_spectrum,
    clean_spectrum,
    dot_product_similarity,
    normalize_spectrum,
    scale_spectrum,
)
from masscube.annotation_new.utils import scale_spcectrum


@pytest.fixture
def noisy_ms2_spectrum():
    return np.array(
        [
            [100.0, 20.0],
            [49.9, 500.0],
            [75.0, 100.0],
            [60.0, 9.0],
            [147.9, 10.0],
            [148.0, 80.0],
            [np.nan, 10.0],
            [90.0, np.inf],
            [110.0, -5.0],
        ],
        dtype=np.float64,
    )


def test_clean_spectrum_combines_filters_and_does_not_mutate_input(noisy_ms2_spectrum):
    original = noisy_ms2_spectrum.copy()

    cleaned = clean_spectrum(
        noisy_ms2_spectrum,
        precursor_mz=150.0,
        precursor_mz_offset=2.0,
        min_mz=50.0,
        abs_int_tol=10.0,
        rel_int_tol=0.1,
        top_n=2,
    )

    np.testing.assert_allclose(cleaned, [[75.0, 100.0], [100.0, 20.0]])
    np.testing.assert_allclose(noisy_ms2_spectrum, original, equal_nan=True)


def test_clean_spectrum_applies_absolute_cutoff_without_relative_cutoff():
    spectrum = np.array([[60.0, 5.0], [70.0, 10.0], [80.0, 15.0]])

    cleaned = clean_spectrum(
        spectrum,
        abs_int_tol=10.0,
        rel_int_tol=None,
        precursor_mz_offset=None,
        min_mz=None,
    )

    np.testing.assert_allclose(cleaned, [[70.0, 10.0], [80.0, 15.0]])


def test_scaling_and_normalization_return_new_spectra():
    spectrum = np.array([[50.0, 0.0], [75.0, 9.0], [100.0, 16.0]])
    original = spectrum.copy()

    scaled = scale_spectrum(spectrum, method="sqrt")
    scaled_default = scale_spectrum(spectrum)
    scaled_legacy = scale_spcectrum(spectrum)
    normalized_max = normalize_spectrum(spectrum, method="max")
    normalized_sum = normalize_spectrum(spectrum, method="sum")

    np.testing.assert_allclose(scaled[:, 1], [0.0, 3.0, 4.0])
    np.testing.assert_allclose(scaled_default[:, 1], [0.0, 9.0 / 16.0, 1.0])
    np.testing.assert_allclose(scaled_legacy, scaled_default)
    np.testing.assert_allclose(normalized_max[:, 1], [0.0, 9.0 / 16.0, 1.0])
    np.testing.assert_allclose(normalized_sum[:, 1], [0.0, 9.0 / 25.0, 16.0 / 25.0])
    np.testing.assert_allclose(spectrum, original)

    with pytest.raises(ValueError, match="Unsupported scaling method"):
        scale_spectrum(spectrum, method="unknown")
    with pytest.raises(ValueError, match="Unsupported normalization method"):
        normalize_spectrum(spectrum, method="unknown")


def test_centroid_spectrum_handles_unsorted_input_and_bounded_groups():
    spectrum = np.array(
        [
            [100.008, 1.0],
            [150.000, 5.0],
            [100.000, 1.0],
            [100.004, 100.0],
        ]
    )

    centroided = centroid_spectrum(spectrum, mz_tol=0.005)

    expected_mz = (100.000 + 100.004 * 100.0) / 101.0
    np.testing.assert_allclose(centroided[0], [expected_mz, 101.0], rtol=1e-6)
    np.testing.assert_allclose(centroided[1], [100.008, 1.0])
    np.testing.assert_allclose(centroided[2], [150.0, 5.0])


def test_align_fragments_prefers_higher_intensity_ambiguous_candidate():
    spectrum_a = np.array([[100.0010, 50.0]])
    spectrum_b = np.array([[100.0012, 5.0], [100.0042, 100.0]])

    matched_a, matched_b = align_fragments(spectrum_a, spectrum_b, mz_tol=0.005)

    np.testing.assert_array_equal(matched_a, [0])
    np.testing.assert_array_equal(matched_b, [1])


def test_align_fragments_handles_multiple_to_multiple_candidates_one_to_one():
    spectrum_a = np.array([[100.000, 10.0], [100.003, 100.0]])
    spectrum_b = np.array([[100.001, 100.0], [100.004, 10.0]])

    matched_a, matched_b = align_fragments(spectrum_a, spectrum_b, mz_tol=0.005)

    assert set(zip(matched_a, matched_b)) == {(0, 1), (1, 0)}
    assert len(np.unique(matched_a)) == len(matched_a)
    assert len(np.unique(matched_b)) == len(matched_b)


def test_align_fragments_uses_mass_error_to_break_equal_contribution_ties():
    spectrum_a = np.array([[100.000, 10.0]])
    spectrum_b = np.array([[99.999, 20.0], [100.002, 20.0]])

    _, matched_b = align_fragments(spectrum_a, spectrum_b, mz_tol=0.005)

    np.testing.assert_array_equal(matched_b, [0])


def test_dot_product_similarity_for_identical_partial_and_ambiguous_spectra():
    identical = np.array([[100.0, 3.0], [200.0, 4.0]])
    score, matched = dot_product_similarity(identical, identical.copy(), mz_tol=0.005)
    assert score == pytest.approx(1.0)
    assert matched == 2

    partial = np.array([[100.001, 6.0], [300.0, 8.0]])
    score, matched = dot_product_similarity(identical, partial, mz_tol=0.005)
    assert score == pytest.approx(0.36)
    assert matched == 1

    ambiguous_a = np.array([[100.0010, 50.0]])
    ambiguous_b = np.array([[100.0012, 5.0], [100.0042, 100.0]])
    score, matched = dot_product_similarity(ambiguous_a, ambiguous_b, mz_tol=0.005)
    assert score == pytest.approx(100.0 / np.sqrt(100.0**2 + 5.0**2))
    assert matched == 1


def test_dot_product_similarity_returns_zero_without_valid_matches():
    spectrum_a = np.array([[100.0, 10.0], [110.0, 0.0]])
    spectrum_b = np.array([[200.0, 10.0], [100.0, -1.0]])

    score, matched = dot_product_similarity(spectrum_a, spectrum_b, mz_tol=0.005)

    assert score == 0.0
    assert matched == 0


def test_dot_product_similarity_is_symmetric_and_accepts_precomputed_norms():
    spectrum_a = np.array([[100.000, 10.0], [100.003, 100.0], [200.0, 20.0]])
    spectrum_b = np.array([[100.001, 100.0], [100.004, 10.0], [300.0, 20.0]])
    norm_a = np.linalg.norm(spectrum_a[:, 1])
    norm_b = np.linalg.norm(spectrum_b[:, 1])

    score_ab, matched_ab = dot_product_similarity(
        query_spectrum=spectrum_a,
        library_spectrum=spectrum_b,
        mz_tol=0.005,
        query_norm_l2=norm_a,
        library_norm_l2=norm_b,
    )
    score_ba, matched_ba = dot_product_similarity(
        query_spectrum=spectrum_b,
        library_spectrum=spectrum_a,
        mz_tol=0.005,
        query_norm_l2=norm_b,
        library_norm_l2=norm_a,
    )

    assert score_ab == pytest.approx(score_ba)
    assert matched_ab == matched_ba == 2


def test_fragment_utilities_reject_invalid_tolerances_and_shapes():
    spectrum = np.array([[100.0, 10.0]])

    with pytest.raises(ValueError, match="mz_tol must be non-negative"):
        align_fragments(spectrum, spectrum, mz_tol=-0.001)
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        clean_spectrum(np.array([100.0, 10.0]))
