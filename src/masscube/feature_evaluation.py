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

    mask = y > 0
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
