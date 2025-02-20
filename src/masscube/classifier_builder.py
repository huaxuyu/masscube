# Author: Huaxu Yu

"""
classifier_builder.py - Build a random forest classification model from raw data.
"""

# imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import pickle
import shutil

from .workflows import untargeted_metabolomics_workflow

"""
Functions
"""

# Feature selection
def feature_selection(X, y, k=None):
    """
    Select features for the classification model.

    Parameters
    ----------
    X : two-dimensional numpy array
        The feature matrix.
    y : one-dimensional numpy array
        The target variable.
    k : int
        The number of features to select. By default, it 
        is set to the number of samples divided by 10 (1/10 rule)
        and rounded up.

    Returns
    -------
    X_new : two-dimensional numpy array
        The selected features.
    selected_features : one-dimensional numpy array
        The indices of the selected features.
    """

    if k is None:
        k = int(np.ceil(X.shape[0]/10))
    
    if k < 5:
        k = 5
        print('Sample size is too small. Constructed model may not be reliable.')
        print('The number of features is set to 5.')

    selector = SelectKBest(k=k)

    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)

    return X_new, selected_features


def train_rdf_model(X_train, y_train):
    """
    Train a random forest model.

    Parameters
    ----------
    X_train : two-dimensional numpy array
        The feature matrix for training.
    y_train : one-dimensional numpy array
        The target variable for training.

    Returns
    -------
    model : RandomForestClassifier
        The trained random forest model.
    """

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model


def cross_validate_model(X, y, model, k=5, random_state=0):
    """
    Cross-validate the model.

    Parameters
    ----------
    X : two-dimensional numpy array
        The feature matrix.
    y : one-dimensional numpy array
        The target variable.
    model : RandomForestClassifier
        The trained random forest model.
    k : int
        The number of folds for cross-validation.
    random_state : int
        The random state for the shuffle in KFold.

    Returns
    -------
    scores : list
        The accuracy scores for each fold.
    """

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf)

    return scores


def predict(model, X_test):
    """
    Predict the samples using the trained model.

    Parameters
    ----------
    model : RandomForestClassifier
        The trained random forest model.
    X_test : two-dimensional numpy array
        The feature matrix for testing.

    Returns
    -------
    predictions : one-dimensional numpy array
        The predicted classes.
    """

    predictions = model.predict(X_test)

    return predictions


def evaluate_model(predictions, y_test):
    """
    Evaluate the model using accuracy score. See sklearn.metrics.accuracy_score
    for more information.
    
    Parameters
    ----------
    predictions : one-dimensional numpy array
        The predicted classes.
    y_test : one-dimensional numpy array
        The true classes.

    Returns
    -------
    accuracy : float
        The accuracy of the model.
    """

    accuracy = accuracy_score(y_test, predictions)

    return accuracy


def build_classifier(path=None, by_group=None, feature_num=None, gaussian_cutoff=0.6, detection_rate_cutoff=0.9, fill_ratio=0.5,
                     cross_validation_k=5):
    """
    To build classifier from raw data.

    Parameters
    ----------
    path : str
        Path to the project file.
    feature_num : int
        The number of features to select for building the model.
    gaussian_cutoff : float
        The Gaussian similarity cutoff. Default is 0.6.
    fill_percentage_cutoff : float
        The fill percentage cutoff. Default is 0.9.
    fill_ratio : float
        The zero values will be replaced by the minimum value in the feature matrix times fill_ratio. Default is 0.5.
    cross_validation_k : int
        The number of folds for cross-validation. Default is 5.
    """

    if path is None:
        path = os.getcwd()

    # process the raw data
    _, params = untargeted_metabolomics_workflow(path, return_results=True)

    if by_group is None:
        by_group = params.by_group_name

    # load the processed data
    df = pd.read_csv(os.path.join(path, 'aligned_feature_table.txt'), sep='\t', low_memory=False)
    sub_medadata = params.sample_metadata[(~params.sample_metadata['is_blank']) & (~params.sample_metadata['is_qc'])]
    reg_samples = sub_medadata.iloc[:,0].values
    
    # preprocess data by
    selected_feature_numbers = [len(df)]
    # 1. selecting annotated metabolites
    df = df[df['search_mode'] == 'identity_search']
    selected_feature_numbers.append(len(df))
    # 2. filter by Gaussian similarity
    df = df[df['Gaussian_similarity'] > gaussian_cutoff]
    selected_feature_numbers.append(len(df))
    # 3. remove in-source fragments
    df = df[df['is_in_source_fragment'] == False]
    selected_feature_numbers.append(len(df))
    # 4. remove isotopes
    df = df[df['is_isotope'] == False]
    selected_feature_numbers.append(len(df))
    df = df.drop_duplicates(subset='annotation', keep='first')
    selected_feature_numbers.append(len(df))
    # 5. only keep the samples that are not blank or qc
    df = df[df['detection_rate'] > detection_rate_cutoff]
    selected_feature_numbers.append(len(df))
    
    X = df[reg_samples].values
    for i in X:
        i[i == 0] = np.min(i[i != 0]) * fill_ratio
    X = X.T
    # standardize the data X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # factorize y
    y = sub_medadata[by_group].values
    y, fac_id = np.array(pd.factorize(y)[0]), np.array(pd.factorize(y)[1])

    X_new, selected_features = feature_selection(X, y, k=feature_num)

    # Train model
    model = train_rdf_model(X_new, y)

    # Cross-validation
    cv_scores = cross_validate_model(X_new, y, model, k=cross_validation_k)
    
    # ROC
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=0)
    model_roc = train_rdf_model(X_train, y_train)
    y_score = model_roc.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
    roc_auc = auc(fpr, tpr)

    # output the results
    new_path = os.path.join(path, 'classifier')
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    with open(os.path.join(new_path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(new_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    df_cross_val = pd.DataFrame(cv_scores, columns=['accuracy'])
    df_cross_val.to_csv(os.path.join(new_path, 'cross_validation_scores.csv'), index=False)
    df_selected_for_model = df.iloc[selected_features, :]
    df_selected_for_model.to_csv(os.path.join(new_path, 'selected_features.txt'), index=False, sep='\t')
    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    df_roc.to_csv(os.path.join(new_path, 'roc_curve.csv'), index=False)
    df_factorize = pd.DataFrame({'group_id':np.arange(len(fac_id)), 'group':fac_id})
    df_factorize.to_csv(os.path.join(new_path, 'factorize.csv'), index=False)

    # print the results
    print('Cross-validation scores:', np.mean(cv_scores).round(3), '+/-', np.std(cv_scores).round(3))
    print('ROC AUC:', roc_auc)


def predict_samples(path, mz_tol=0.01, rt_tol=0.3):
    """
    To predict the samples using the trained model.

    Parameters
    ----------
    path : str
        Path to the project file.
    mz_tol : float
        The m/z tolerance for matching the features. Default is 0.01.
    rt_tol : float
        The retention time tolerance for matching the features. Default is 0.3.
    """

    prediction_path = os.path.join(path, 'prediction')
    if not os.path.exists(prediction_path):
        raise ValueError('The prediction folder does not exist. Please prepare the prediction folder first.')
    
    # load the model
    with open(os.path.join(path, 'classifier', 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    # load the selected features
    df_selected_features = pd.read_csv(os.path.join(path, 'classifier', 'selected_features.txt'), sep='\t', low_memory=False)

    # process the new data
    # copy the parameter files to the prediction folder
    parameter_file = os.path.join(path, 'parameters.csv')
    if os.path.exists(parameter_file):
        shutil.copy(parameter_file, prediction_path)
    else:
        print("No parameter file is found in the project directory. Default parameters will be used.")

    untargeted_metabolomics_workflow(prediction_path)

    # load the processed data
    df = pd.read_csv(os.path.join(prediction_path, 'aligned_feature_table.txt'), sep='\t', low_memory=False)
    sample_table_path = os.path.join(prediction_path, 'sample_table.csv')
    if os.path.exists(sample_table_path):
        sample_table = pd.read_csv(sample_table_path)
        sample_names = sample_table.iloc[:,0].values[(sample_table.iloc[:,1].values != "blank") & (sample_table.iloc[:,1].values != "qc")]
    else:
        print("No sample table is found in the prediction directory. QC and blank samples (if exist) will be included for prediction.")
        sample_names = df.columns[22:]
    keep_columns = df.columns[:22].tolist() + sample_names.tolist()
    df = df[keep_columns]
    # find the selected features in the sample
    feature_mz = df['m/z'].values
    feature_rt = df['RT'].values
    matched_idx = np.zeros(len(df_selected_features))
    for i in range(len(df_selected_features)):
        mz = df_selected_features['m/z'][i]
        rt = df_selected_features['RT'][i]
        matched_v = np.where(np.logical_and(np.abs(feature_mz - mz) < mz_tol, np.abs(feature_rt - rt) < rt_tol))[0]
        if len(matched_v) > 0:
            matched_idx[i] = matched_v[0]
    
    X = df.iloc[matched_idx.astype(int), 22:].values
    for i in X:
        i[i == 0] = np.min(i[i != 0]) * 0.5
    X = X.T
    # standardize the data X
    with open(os.path.join(path, 'classifier', 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    X = scaler.transform(X)
    # predict
    predictions = predict(model, X)
    df_factorize = pd.read_csv(os.path.join(path, 'classifier', 'factorize.csv'))
    predictions = df_factorize['group'][predictions]

    # output the results
    result_df = pd.DataFrame({'sample': sample_names, 'group': predictions})
    result_df.to_csv(os.path.join(prediction_path, 'predictions.csv'), index=False)

    # print the results
    print('Results are saved in the prediction folder.')