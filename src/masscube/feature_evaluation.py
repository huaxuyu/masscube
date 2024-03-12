# Author: Hauxu Yu

# A module to load a trained model to predict the quality of features
# Prediction is based on peak shape

# Import modules
import numpy as np
from scipy.interpolate import interp1d
from keras.models import load_model
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import pearsonr, spearmanr
import os
import warnings


def predict_quality(d, model=None, threshold=0.5):
    """
    Function to predict the quality of a feature as an ROI.

    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object that contains the MS data.
    model: keras model
        A keras model that is trained to predict the quality of a feature.
    threshold: float
        A threshold to determine the quality of a feature.
    """

    if model is None:
        model = load_ann_model()

    temp = np.array([peak_interpolation(roi.int_seq) for roi in d.rois])
    q = model.predict(temp, verbose=0)[:,0] > threshold

    for i in range(len(d.rois)):
        # if the roi quality is not good, then skip and don't overwrite
        if d.rois[i].quality == 'good' and q[i] == 0:
            d.rois[i].quality = 'bad peak shape'


def peak_interpolation(peak):
    '''
    A function to interpolate a peak to a vector of a given size.

    Parameters
    ----------------------------------------------------------
    peak: numpy array
        A numpy array that contains the peak to be interpolated.
    '''
    
    peak_interp_rule = interp1d(np.arange(len(peak)), peak, kind='linear')
    interp_seed = np.linspace(0, len(peak)-1, 64)
    peak_interp = peak_interp_rule(interp_seed)

    peak_interp = peak_interp / np.max(peak_interp)

    return peak_interp


def load_ann_model():
    """
    load the ANN model for peak quality prediction
    """
    data_path_ann = os.path.join(os.path.dirname(__file__), 'model', "peak_quality_NN.keras")
    ann_model = load_model(data_path_ann, compile=False)
    
    return ann_model


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
        return -1
    
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
