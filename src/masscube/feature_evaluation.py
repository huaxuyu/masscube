# Author: Huaxu Yu

# Evaluate feature quality by chromatographic peak shapes, mz errors, rt errors, etc.

# imports
import numpy as np


"""
Functions
------------------------------------------------------------------------------------------------------------------------
"""

def calculate_gaussian_similarity(x, y):
    """
    calculate the gaussian similarity using dot product

    Parameters
    ----------
    x: numpy array
        Retention time
    y: numpy array
        Intensity
    
    Returns
    -------
    float
        similarity score
    """

    if type(x) is not np.ndarray:
        x = np.array(x)
    if type(y) is not np.ndarray:
        y = np.array(y)

    # if the length of the peak is less than 5, return 0
    if len(x) < 5:
        return 0.0

    # Estimate the parameters of the Gaussian function
    # a: amplitude, b: mean, c: standard deviation
    a = np.max(y)
    idx = np.argmax(y)
    b = x[idx]

    c1 = x[-1] - b
    c2 = b - x[0]

    if idx == len(y) - 1:
        c1 = 0
    else:
        for i in range(idx, len(y)):
            if y[i] < a / 2:
                c1 = x[i] - b
                break
    if idx == 0:
        c2 = 0
    else:
        for i in range(idx, 0, -1):
            if y[i] < a / 2:
                c2 = b - x[i]
                break
    c = (c1 + c2) / 2.355

    y_fit = _gaussian(x, a, b, c)

    similarity = np.corrcoef(y, y_fit)[0, 1]
    
    return np.max([0, similarity])


def calculate_noise_score(y, rel_int_tol=0.05):
    """
    Calculate the noise score that reflect the signal fluctuation.

    Parameters
    ----------
    y: numpy array
        Intensity
    rel_int_tol: float
        Relative intensity tolerance.
    
    Returns
    -------
    float
        noise level
    """

    y = y[y > np.max(y) * rel_int_tol]

    if len(y) < 5:
        return 0.0

    diff = np.diff(y)
    score = np.sum(np.abs(diff)) / np.max(y) / 2 - 1
    return np.max([0, score])


def calculate_asymmetry_factor(y):
    """
    Calcualte the asymmetry factor of the peak at 10% of the peak height.

    Parameters
    ----------
    y: numpy array
        Intensity

    Returns
    -------
    float
        asymmetry factor
    """

    if len(y) < 5:
        return 1.0

    idx = np.argmax(y)

    if idx == 0:
        return 99.0
    elif idx == len(y) - 1:
        return 0.0

    arr = y < 0.1 * y[idx]

    left_idx = np.where(arr[:idx])[0]
    right_idx = np.where(arr[idx:])[0]

    if len(left_idx) == 0:
        left_idx = 0
    else:
        left_idx = left_idx[-1]
    
    if len(right_idx) == 0:
        right_idx = len(y) - 1
    else:
        right_idx = right_idx[0] + idx

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