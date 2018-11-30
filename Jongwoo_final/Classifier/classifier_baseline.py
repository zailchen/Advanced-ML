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


