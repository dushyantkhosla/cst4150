import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from .model import run_model

# -------------- SETUP: LOGREG --------------

pipe_logreg = Pipeline([
    ('scale', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced',
                                 max_iter=10**4,
                                 n_jobs=-1))
])

grid_logreg = {
    'model__C': [0.05, 0.1, 1]
}

# -------------- SETUP: DTREE --------------

pipe_dtree = Pipeline([
    ("scale", StandardScaler()),
    ("model", DecisionTreeClassifier(min_samples_leaf=5,
                                     class_weight="balanced",
                                     max_features=15))
])

grid_dtree = {"model__max_depth": [4, 6, 8]}

# -------------- SETUP: KNN --------------

pipe_knn = Pipeline([
    ('scale', StandardScaler()),
    ('model', KNeighborsClassifier())
])

grid_knn = {
    'model__weights': ['uniform', 'distance'],
    'model__n_neighbors': [5, 10, 15]
}

# -------------- SETUP: Support Vector Machine --------------

pipe_svc = Pipeline([
    ('scale', StandardScaler()),
    ('model', SVC(class_weight='balanced', probability=True))
])

grid_svc = {
    'model__C': [1, 5, 10],
    'model__kernel': ['linear', 'rbf']
}

# -------------- SETUP: Adaptive Boosting --------------

pipe_adaboost = Pipeline([
    ('scale', StandardScaler()),
    ('model', AdaBoostClassifier())
])

grid_adaboost = {
    'model__n_estimators': [50, 100, 200]
}

# -------------- SETUP: Random Forest --------------

pipe_rfo = Pipeline([
    ('scale', StandardScaler()),
    ('model', RandomForestClassifier(max_depth=2,
                                     min_samples_leaf=5))
])

grid_rfo = {
    'model__n_estimators': [50, 100, 200]
}

# -------------- SETUP: Payloads --------------

payloads = {
    'dtree': {
        'pipe': pipe_dtree,
        'grid': grid_dtree
    },
    'knn': {
        'pipe': pipe_knn,
        'grid': grid_knn
    },
    'logreg': {
        'pipe': pipe_logreg,
        'grid': grid_logreg
    },
    'svc': {
        'pipe': pipe_svc,
        'grid': grid_svc
    },
    'adaboost': {
        'pipe': pipe_adaboost,
        'grid': grid_adaboost
    },
    'rfo': {
        'pipe': pipe_rfo,
        'grid': grid_rfo
    }
}

name = 'knn'

result = run_model(
    df=df,
    name=f"{name}",
    calibrate=True,
    gscv=GridSearchCV(payloads[name]['pipe'],
                      payloads[name]['grid'],
                      scoring="roc_auc",
                      cv=3,
                      n_jobs=-1)
)