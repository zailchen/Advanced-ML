import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from problem import get_train_data
data_train, labels_train = get_train_data()


# Load data

# fMRI data (3 atlas) - basc064, basc122, basc197
# Only 1D data for this analysis
# We are focus on correlation connectivity measure for this analysis.


with open('data/fmri_correlation_features/train/1D/msdl_corr_1d.npy', 'rb') as p_f:
    train_msdl_corr = np.load(p_f)

with open('data/fmri_correlation_features/train/1D/basc064_corr_1d.npy', 'rb') as p_f:
    train_basc064_corr = np.load(p_f)

with open('data/fmri_correlation_features/train/1D/basc122_corr_1d.npy', 'rb') as p_f:
    train_basc122_corr = np.load(p_f)

with open('data/fmri_correlation_features/train/1D/basc197_corr_1d.npy', 'rb') as p_f:
    train_basc197_corr = np.load(p_f)

with open('data/fmri_correlation_features/train/1D/craddock_scorr_mean_corr_1d.npy', 'rb') as p_f:
    train_craddock_corr = np.load(p_f)

with open('data/fmri_correlation_features/train/1D/harvard_oxford_cort_prob_2mm_corr_1d.npy', 'rb') as p_f:
    train_harvard_corr = np.load(p_f)

with open('data/fmri_correlation_features/train/1D/power_2011_corr_1d.npy', 'rb') as p_f:
    train_power_corr = np.load(p_f)


with open('data/fmri_correlation_features/test/1D/msdl_corr_1d.npy', 'rb') as p_f:
    test_msdl_corr = np.load(p_f)

with open('data/fmri_correlation_features/test/1D/basc064_corr_1d.npy', 'rb') as p_f:
    test_basc064_corr = np.load(p_f)

with open('data/fmri_correlation_features/test/1D/basc122_corr_1d.npy', 'rb') as p_f:
    test_basc122_corr = np.load(p_f)

with open('data/fmri_correlation_features/test/1D/basc197_corr_1d.npy', 'rb') as p_f:
    test_basc197_corr = np.load(p_f)

with open('data/fmri_correlation_features/test/1D/craddock_scorr_mean_corr_1d.npy', 'rb') as p_f:
    test_craddock_corr = np.load(p_f)

with open('data/fmri_correlation_features/test/1D/harvard_oxford_cort_prob_2mm_corr_1d.npy', 'rb') as p_f:
    test_harvard_corr = np.load(p_f)

with open('data/fmri_correlation_features/test/1D/power_2011_corr_1d.npy', 'rb') as p_f:
    test_power_corr = np.load(p_f)


with open('data/fmri_correlation_features/y_train.pkl', 'rb') as f:
    label_train = pickle.load(f)

with open('data/fmri_correlation_features/y_test.pkl', 'rb') as f:
    label_test = pickle.load(f)


X_train = np.concatenate([train_msdl_corr, train_basc064_corr, train_basc122_corr,
                          train_basc197_corr, train_craddock_corr, train_harvard_corr,
                          train_power_corr], axis=1)
y_train = label_train

X_test = np.concatenate([test_msdl_corr, test_basc064_corr, test_basc122_corr,
                         test_basc197_corr, test_craddock_corr, test_harvard_corr,
                         test_power_corr], axis=1)
y_test = label_test


# Model -- Simple Logistic Regression

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# StandardScaler : Standardize features by removing the mean and scaling to unit variance
# refer to http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

class Classifier(BaseEstimator):
    def __init__(self):
        self.params = {'logisticregression__penalty': ['l1', 'l2'],
                       'logisticregression__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10]}
        self.clf = make_pipeline(StandardScaler(), LR())
        self.grid = GridSearchCV(self.clf, self.params, scoring='accuracy', cv=5, verbose=1)

    def fit(self, X, y):
        return self.grid.fit(X, y)

    def predict(self, X):
        return self.grid.predict(X)

    def predict_proba(self, X):
        return self.grid.predict_proba(X)


# Fit
fit_all = Classifier().fit(X_train, y_train)

# Evaluation
y_pred_proba = fit_all.predict_proba(X_test)[:,1]
y_pred = fit_all.predict(X_test)

# Accuracy
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

ROC_AUC = roc_auc_score(y_test, y_pred_proba)
Accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Area Under the Curve of the Receiver Operating Characteristic (ROC-AUC)
print("ROC-AUC:", ROC_AUC, ", Accuracy=", Accuracy, ", F1 score=", f1)









