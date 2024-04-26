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

    mask = y > np.max(y) * 0.05
    x = x[mask]
    y = y[mask]

    # Initial guess for the parameters
    if len(x) < 3:
        return 1.0
    
    initial_guess = [max(y), x[np.argmax(y)], 0.2]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, _ = curve_fit(_gaussian, x, y, p0=initial_guess)
        except:
            popt = np.array(initial_guess)
    
    y_fit = _gaussian(x, *popt)

    # if all values in y_fit is zero
    if np.all(y_fit == 0):
        return -1

    similarity = spearmanr(y, y_fit)[0]
    
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

    y = y[y > np.max(y) * 0.05]

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

    y = y[y > np.max(y) * 0.05]
    if len(y) < 5:
        return 0.0
    
    # interpolate y to get 100 points
    x = np.linspace(0, 1, 100)
    y = np.interp(x, np.linspace(0, 1, len(y)), y)

    ten_height = np.max(y) * 0.1
    idx = np.argmax(y)

    if idx == 0:
        return 99
    if idx == len(y) - 1:
        return 0

    left_idx = np.argmin(np.abs(y[:idx] - ten_height))
    right_idx = np.argmin(np.abs(y[idx:] - ten_height)) + idx

    return (right_idx - idx) / (idx - left_idx)


