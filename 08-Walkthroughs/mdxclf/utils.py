import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, roc_curve

def get_model_scores(model, train_test_sets):
    """
    Returns scores produced by trained estimator
    Uses the `predict_proba` attribute or the `decision_function`,
    whichever is available.

    Input
    -----
    model: trained (fitted) classifier
    train_test_sets: tuple of (X_tr_scaled, y_tr, X_te_scaled, y_te)

    Returns
    -------
    probs_positive_class: model scores

    """
    X_tr_scaled, y_tr, X_te_scaled, y_te = train_test_sets

    if hasattr(model, 'predict_proba'):
        probs_positive_class = model.predict_proba(X_te_scaled)[:, 1]
    else:
        probs_positive_class = model.decision_function(X_te_scaled)
        probs_positive_class = \
        (probs_positive_class-probs_positive_class.min())/(probs_positive_class.max()-probs_positive_class.min())

    return probs_positive_class


def get_calibrated_probabilities(model,
                                 train_test_sets,
                                 method='sigmoid',
                                 cv=5,
                                 plot=False):
    """
    Returns a DataFrame with calibrated probabilities
    and uncalibrated scores for given (fitted) model.

    Invokes CalibratedClassifierCV() with passed `method` and `cv`
    """
    X_tr_scaled, y_tr, X_te_scaled, y_te = train_test_sets

    # Uncalibrated
    probs_uncalibrated = get_model_scores(model=model,
                                          train_test_sets=train_test_sets)

    # Calibrated
    model_calibrated = CalibratedClassifierCV(model, method=method, cv=cv)
    model_calibrated.fit(X_tr_scaled, y_tr)

    probs_calibrated = get_model_scores(model=model_calibrated,
                                        train_test_sets=train_test_sets)

    result = pd.DataFrame({'uncalibrated': probs_uncalibrated,
                           'calibrated': probs_calibrated})

    if plot:
        result.plot.hist(subplots=True, bins=10, sharex=True)

    return result


def get_cutoff_precision_recall(y_true, y_scores, pos_label=1):
    """
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true,
                                                             y_scores,
                                                             pos_label)

    df_pr = pd.DataFrame(data=zip(precisions, recalls, thresholds),
                         columns=['precision', 'recall', 'threshold'])

    cutoff = (
        df_pr
        .eval("delta = precision-recall")
        .assign(abs_delta=lambda fr: fr['delta'].abs())
        .set_index('threshold')
        .loc[:, 'abs_delta']
        .idxmin()
        .round(2))
    
    return cutoff
    
    
def get_predictor_importance(fitted_model, names, plot=True):
    """
    1. Extract importance from `coef` or `feature_importances_`
    attribute of fitted_model
    2. Plots top 10
    3. Return a Series with all importances
    """
    if hasattr(fitted_model, 'coef_'):
        importances = fitted_model.coef_[0]
    elif hasattr(fitted_model, 'feature_importances_'):
        importances = fitted_model.feature_importances_
    else:
        print("Model has no `coef_` or `feature_importances_` attribute")
        importances = np.array([])

    assert len(importances) == len(names)

    srs_imps = pd.Series(data=importances,
                         index=names,
                         name='importances')

    if plot:
        (srs_imps
         .to_frame()
         .assign(abs=lambda fr: fr['importances'].abs())
         .sort_values('abs')
         .tail(10)
         .drop('abs', axis=1)
         .squeeze()
         .sort_values()
         .plot.barh(title="\nTop 10 Variables by Importance\n"))

    return srs_imps