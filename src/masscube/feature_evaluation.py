# Author: Hauxu Yu

# A module to load a trained model to predict the quality of features
# Prediction is based on peak shape

# Import modules
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import warnings


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


def calculate_gaussian_similarity(x, y):
    """
    calculate the R-squared value of a Gaussian fit to the peak shape

    Parameters
    ----------
    x: numpy array
        Retention time
    y: numpy array
        Intensity
    
    Returns
    -------
    float
        R-squared value
    """

    if type(x) is not np.ndarray:
        x = np.array(x)
    if type(y) is not np.ndarray:
        y = np.array(y)

    # Initial guess for the parameters
    if len(x) < 5:
        return np.nan

    a = np.max(y)
    b = x[np.argmax(y)]
    c = np.min([x[-1]-b, b-x[0], 2]) / 4

    if c == 0:
        c = np.max([x[-1]-b, b-x[0]]) / 4

    y_fit = _gaussian(x, a, b, c)

    # use dot product to calculate the similarity
    similarity = np.dot(y, y_fit) / (np.linalg.norm(y) * np.linalg.norm(y_fit))
    
    return similarity


def calculate_noise_level(y):
    """
    Calculate the noise level of the peak shape

    Parameters
    ----------
    y: numpy array
        Intensity
    
    Returns
    -------
    float
        noise level
    """

    y = y[y > np.max(y) * 0.1]

    if len(y) < 5:
        return 0.0

    diff = np.diff(y)
    signs = np.sign(diff)
    counter = -1
    for i in range(1, len(diff)):
        if signs[i] != signs[i-1]:
            counter += 1
    if counter == -1:
        return 0.0
    
    return counter / (len(y)-2)


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

    idx = np.argmax(y)

    if idx == 0:
        return 0.0
    elif idx == len(y) - 1:
        return 99

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

    return (idx - left_idx) / (right_idx - idx)