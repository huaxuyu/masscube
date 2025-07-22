# Author: Huaxu Yu

# Evaluate feature quality by chromatographic peak shapes, mz errors, rt errors, etc.

# imports
import numpy as np
from numpy import dot

"""
Functions
------------------------------------------------------------------------------------------------------------------------
"""

def calculate_gaussian_similarity(x, y, len_tol=5):
    """
    calculate the gaussian similarity using Pearson correlation coefficient.

    Parameters
    ----------
    x: numpy array
        Retention times.
    y: numpy array
        MS1 signal intensities.
    len_tol: int
        Minimum length of the peak to calculate the similarity score. 
        If the length of the peak is less than this value, return 0.
    
    Returns
    -------
    float
        similarity score
    """

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    # if the length of the peak is less than the tolerance, return 0
    if len(x) < len_tol or np.max(y) <= 0:
        return 0.0

    # Estimate the parameters of the Gaussian function
    # a: amplitude, b: mean, c: standard deviation
    a = np.max(y)
    idx = np.argmax(y)
    b = x[idx]

    # Compute approximate FWHM using half-height
    right_idx = np.where(y[idx:] < a / 2)[0]
    left_idx = np.where(y[:idx] < a / 2)[0]

    c1 = (x[right_idx[0] + idx] - b) if len(right_idx) > 0 else (x[-1] - b)
    c2 = (b - x[left_idx[-1]]) if len(left_idx) > 0 else (b - x[0])

    fwhm = c1 + c2
    c = fwhm / 2.355 if fwhm > 0 else 1e-6  # Avoid division by zero

    # Generate fitted Gaussian curve
    y_fit = a * np.exp(-0.5 * ((x - b) / c) ** 2)

    # Fast correlation using dot product
    y = y.astype(float)
    y_fit = y_fit.astype(float)
    y -= y.mean()
    y_fit -= y_fit.mean()
    similarity = dot(y, y_fit) / (np.linalg.norm(y) * np.linalg.norm(y_fit) + 1e-12)
    
    return max(0, similarity)


def calculate_noise_score(y, rel_int_tol=0.05, len_tol=5):
    """
    Calculate the noise score that reflect the signal fluctuation.

    Parameters
    ----------
    y: array-like
        Intensity values of the peak.
    rel_int_tol: float
        Relative intensity threshold (filter low signals).
    len_tol: int
        Minimum length of the peak to calculate the noise score. 
        If the length of the peak is less than this value, return 0.
    
    Returns
    -------
    float
        Noise score (0 = no noise, higher = noiser)
    """

    y = np.array(y, dtype=float) if not isinstance(y, np.ndarray) else y
    
    y = y[y > np.max(y) * rel_int_tol]

    if len(y) < len_tol:
        return 0

    y = np.concatenate(([0], y, [0]))
    diff = np.diff(y)

    return np.sum(np.abs(diff)) / np.max(y) / 2 - 1


def calculate_asymmetry_factor(y, threshold_ratio=0.1):
    """
    Calculate peak asymmetry at a given threshold level (default: 10%).

    Parameters
    ----------
    y : array-like
        Intensity values
    threshold_ratio : float
        Fraction of peak height to define tails (default = 0.1 = 10%)

    Returns
    -------
    float
        Asymmetry factor:
        - ~1 = symmetric
        - >1 = right-skewed
        - <1 = left-skewed
        - np.inf = undefined
    """

    y = np.array(y, dtype=float) if not isinstance(y, np.ndarray) else y

    if len(y) < 5 or np.max(y) <= 0:
        return np.inf

    idx = np.argmax(y)

    # Define threshold
    threshold = y[idx] * threshold_ratio

    # Find last point before peak below threshold (left tail)
    left_candidates = np.where(y[:idx] < threshold)[0]
    left_idx = left_candidates[-1] if len(left_candidates) else 0

    # Find first point after peak below threshold (right tail)
    right_candidates = np.where(y[idx:] < threshold)[0]
    right_idx = (right_candidates[0] + idx) if len(right_candidates) else (len(y) - 1)

    # Avoid division by zero
    if idx == left_idx:
        return 99.0

    return (right_idx - idx) / (idx - left_idx)


def squared_error_to_smoothed_curve(original_signal, fit_signal):
    """
    Calculate the sum of squared error between the original signal and the fitted signal.

    Parameters
    ----------
    original_signal: numpy array
        The original signal.
    fit_signal: numpy array
        The fitted signal.

    Returns
    -------
    float
        The noise score.
    """

    diff = (original_signal - fit_signal) / np.max(original_signal)
    return np.sum(diff**2)


"""
Helper functions
------------------------------------------------------------------------------------------------------------------------
"""

# calculate the Gaussian similarity
def _gaussian(x, a, b, c):
    """
    Gaussian function

    Parameters
    ----------
    x: numpy array
        Retention time
    a: float
        Amplitude
    b: float
        Mean
    c: float
        Standard deviation
    """
    return a * np.exp(-0.5 * ((x - b) / c) ** 2)