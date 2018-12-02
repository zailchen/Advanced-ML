from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.discriminant_analysis import LinearClassifierMixin as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB as GNB
import numpy as np
import pandas as pd

# StandardScaler : Standardize features by removing the mean and scaling to unit variance
# refer to http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html


class Classifier(BaseEstimator):
    def __init__(self):
        # self.params = {'svc__C': [0.01, 0.1, 1, 10],
        #                'svc__gamma': [10, 5, 1, 0,1, 0.01]}
        # self.clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
        # self.grid = GridSearchCV(self.clf, self.params, scoring='roc_auc', cv=5, verbose=1)

        self.params = {'logisticregression__penalty': ['l1', 'l2'],
                       'logisticregression__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10]}
        self.clf = make_pipeline(StandardScaler(), LR())
        self.grid = GridSearchCV(self.clf, self.params, scoring='roc_auc', cv=5, verbose=1)

    def fit(self, X, y):
        return self.grid.fit(X, y)

    def predict(self, X):
        return self.grid.predict(X)

    def predict_proba(self, X):
        return self.grid.predict_proba(X)










'''
class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = make_pipeline(StandardScaler(), LR(penalty='l1', C=0.1, random_state=42))

    def fit(self, X, y):
        selected = X[((X['anatomy_select'] == 1) | (X['fmri_select'] == 1))].index.values
        y = pd.Series(y, index=X.index)
        y = y.loc[selected]
        X = X.loc[selected].drop(['anatomy_select', 'fmri_select'], axis=1)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        X = X.drop(['anatomy_select', 'fmri_select'], axis=1)
        return self.clf.predict(X)

    def predict_proba(self, X):
        X = X.drop(['anatomy_select', 'fmri_select'], axis=1)
        return self.clf.predict_proba(X)



class Classifier(BaseEstimator):
    def __init__(self):
        self.clf_connectome = make_pipeline(LR(penalty='l1',C=.1))
        self.clf_anatomy = make_pipeline(LR(penalty='l1', C=.1))
        self.meta_clf = LR(penalty='l1',C=.1)

    def fit(self, X, y):
        X_anatomy = X[[col for col in X.columns if col.startswith('anatomy')]]
        X_connectome = X[[col for col in X.columns
                          if col.startswith('connectome')]]
        test_idx, validation_idx = test_test_split(range(y.size),
                                                     test_size=0.33,
                                                     shuffle=True,
                                                     random_state=42)
        X_anatomy_test = X_anatomy.iloc[test_idx]
        X_anatomy_validation = X_anatomy.iloc[validation_idx]
        X_connectome_test = X_connectome.iloc[test_idx]
        X_connectome_validation = X_connectome.iloc[validation_idx]
        y_test = y[test_idx]
        y_validation = y[validation_idx]

        self.clf_connectome.fit(X_connectome_test, y_test)
        self.clf_anatomy.fit(X_anatomy_test, y_test)

        y_connectome_pred = self.clf_connectome.predict_proba(
            X_connectome_validation)
        y_anatomy_pred = self.clf_anatomy.predict_proba(
            X_anatomy_validation)

        self.meta_clf.fit(
            np.concatenate([y_connectome_pred, y_anatomy_pred], axis=1),
            y_validation)
        return self

    def predict(self, X):
        X_anatomy = X[[col for col in X.columns if col.startswith('anatomy')]]
        X_connectome = X[[col for col in X.columns
                          if col.startswith('connectome')]]

        y_anatomy_pred = self.clf_anatomy.predict_proba(X_anatomy)
        y_connectome_pred = self.clf_connectome.predict_proba(X_connectome)

        return self.meta_clf.predict(
            np.concatenate([y_connectome_pred, y_anatomy_pred], axis=1))

    def predict_proba(self, X):
        X_anatomy = X[[col for col in X.columns if col.startswith('anatomy')]]
        X_connectome = X[[col for col in X.columns
                          if col.startswith('connectome')]]

        y_anatomy_pred = self.clf_anatomy.predict_proba(X_anatomy)
        y_connectome_pred = self.clf_connectome.predict_proba(X_connectome)

        return self.meta_clf.predict_proba(
            np.concatenate([y_connectome_pred, y_anatomy_pred], axis=1))
'''