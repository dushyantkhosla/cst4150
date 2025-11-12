import subprocess
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.tree import export_graphviz
from .utils import get_model_scores


def plot_precision_recall_curve(y_true, y_scores, label_pos_class, plot=False):
    """
    y_scores: output of get_model_scores()

    - Intuitively,
        - **precision** is the ability of the classifier
        not to label a -ve sample as +ve,
        - **recall** is the ability of the classifier
        to find all the positive samples.

    `precision_recall_curve`
    calculates precision and recall at different thresholds
    Helps visualize the optimal threshold to get predictions from scores
    """
    pos_label = y_true.max() if label_pos_class is None else label_pos_class

    precisions, recalls, thresholds = precision_recall_curve(
        y_true,
        y_scores,
        pos_label=pos_label
    )

    df_pr = pd.DataFrame(data=zip(precisions, recalls, thresholds),
                         columns=['precision', 'recall', 'threshold'])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5), sharey=True)

    df_pr.plot.line(x='recall',
                    y='precision',
                    legend=None,
                    ax=axs[0])

    if plot:
        (df_pr
         .set_index('threshold')
         .query("precision > 0.4 & recall > 0.4")
         .plot(ax=axs[1]))

        axs[0].set_xlabel("\nRecall →\n")
        axs[0].set_ylabel("\nPrecision →\n")
        axs[0].set_title("\nPrecision vs. Recall\n")

        axs[1].set_xlabel("\nThreshold →\n")

        try:
            intersection = (df_pr
                            .round(2)
                            .set_index('threshold')
                            .query("precision == recall")
                            .iloc[0]
                            .name)
            axs[1].set_title(f"\nCurves intersect at: t={intersection}\n")
        except Exception as e:
            print(f"Failed with {e}")

    return df_pr


def plot_calibration_curves(estimator, name, train_test_sets, cv):
    """
    Inputs
    ------
    estimator: trained (fitted) classifier
    name: label for the estimator
    train_test_sets: tuple of (X_tr_scaled, y_tr, X_te_scaled, y_te)
    cv: cross-validation

    """
    X_tr_scaled, y_tr, X_te_scaled, y_te = train_test_sets

    # for comparison
    tree = DecisionTreeClassifier()
    svm = SVC()
    logreg = LogisticRegression()

    # Provided estimator, calibrated
    sigmoid = CalibratedClassifierCV(estimator, method='sigmoid', cv=5)
    isotonic = CalibratedClassifierCV(estimator, method='isotonic', cv=5)

    list_ = [
        (tree, 'tree'),
        (svm, 'svm'),
        (logreg, 'logreg'),
        (estimator, name),
        (sigmoid, f"sigmoid + {name}"),
        (isotonic, f"isotonic + {name}")
    ]

    fig, axs = plt.subplots(nrows=2,
                            ncols=1,
                            sharex=True,
                            figsize=(15, 10))

    for model, name in list_:
        model.fit(X_tr_scaled, y_tr)
        y_pred = model.predict(X_te_scaled)
        prob_pos = get_model_scores(model,
                                    train_test_sets=train_test_sets)
        model_score = brier_score_loss(y_te, prob_pos,
                                       pos_label=y_tr.max())

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_te,
            prob_pos,
            n_bins=10)

        (pd.Series(data=fraction_of_positives,
                   index=mean_predicted_value)
         .plot(ax=axs[0],
               label=name,
               kind='line',
               style="s-",
               alpha=0.5))

        (pd.Series(data=prob_pos)
         .plot
         .hist(bins=10,
               range=(0, 1),
               histtype='step',
               ax=axs[1],
               label=name))

    axs[0].set_title("\nCalibration Curves\n")
    axs[0].set_ylabel("Fraction of positives\n")
    axs[0].set_ylim([-0.05, 1.05])
    axs[0].legend(loc="best")
    axs[0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    axs[1].set_xlabel("\nMean Predicted Value\n")
    axs[1].legend(loc="best")


def plot_roc_curve(y_true, y_scores, pos_label=1):
    """
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

    area_under_roc = roc_auc_score(y_true, y_scores)

    df_roc = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'threshold': thresholds
    }).round(3).set_index('threshold')

    ax = df_roc.plot.line(x='fpr', y='tpr', legend=None)

    ax.set_title(f"\nArea under ROC Curve: {area_under_roc:.2f}\n")
    ax.set_xlabel("\nFalse Positive Rate →\n(1-Specificity)")
    ax.set_ylabel("True Positive Rate →\n(Sensitivity)\n")

    return df_roc


def plot_tree(tree, feature_names):
    """
    Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree,
                        out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]

    try:
        subprocess.check_call(command)
    except Exception as e:
        exit(f"Could not run graphviz, to produce visualization. {e}")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm_txt = np.array([['TN', 'FP'], ['FN', 'TP']])
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, 
                 # format(cm[i, j], fmt),
                 f"{cm[i,j]:{fmt}}\n{cm_txt[i,j]}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')