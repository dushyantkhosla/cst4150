from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (precision_score,
                             recall_score,
                             precision_recall_curve)

def run_model(df, i, name, gscv, calibrate=True):
    """
    1. create balanced dataset
    2. split into train, test sets
    3. run grid search
    4. get probability scores
    5. calibrate as directed
    6. find optimal cutoff from precision-recall
    7. return predictions, data, scores
    """
    df_undersampled = pd.concat([
        df.query(target==0).sample(frac=0.3, random_state=0),
        df.query("target==1")
    ])

    X = df_undersampled.drop("target", axis=1).copy()
    y = df_undersampled.loc[:, "target"].copy()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.7, stratify=y)

    model = gscv.fit(X_tr, y_tr)

    # Probabilities
    y_scores = model.predict_proba(X_te)[:, 1]

    if calibrate:
        sigmoid = CalibratedClassifierCV(model, cv=2, method="sigmoid")
        sigmoid.fit(X_tr, y_tr)
        y_probs = sigmoid.predict_proba(X_te)[:, 1]
    else:
        y_probs = np.array(y_scores)

    # Cutoff
    p, r, t = precision_recall_curve(y_te, y_probs, pos_label=1)

    df_pr = (pd.DataFrame(data=zip(p, r, t),
                          columns=["precision", "recall", "threshold"])
             .set_index("threshold"))

    cutoff = (pd.Series(data=np.abs(df_pr["precision"] - df_pr["recall"]),
                        index=df_pr.index)
              .idxmin()
              .round(2))

    # Predictions
    y_pred = (y_probs >= cutoff).astype(int)

    dict_data = {
        "X_tr": X_tr,
        "X_te": X_te,
        "y_tr": y_tr,
        "y_te": y_te,
        "y_scores": y_scores,
        "y_probs": y_probs,
        "y_pred": y_pred,
    }

    dict_scores = {
        "precision": precision_score(y_te, y_pred),
        "recall": recall_score(y_te, y_pred),
    }

    payload = {
        "name": name,
        "model": model,
        "data": dict_data,
        "scores": dict_scores
    }

    return payload