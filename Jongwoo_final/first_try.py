import numpy as np
np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt



############
# The data #
############

# Load feature extracted data
import pickle

with open('fmri_corr_feature/train_corr_fmri_msdl.pkl', 'rb') as p_f:
    corr_msdl = pickle.load(p_f)
with open('fmri_corr_feature/train_corr_fmri_basc064.pkl', 'rb') as p_f:
    corr_basc064 = pickle.load(p_f)
with open('fmri_corr_feature/train_corr_fmri_basc122.pkl', 'rb') as p_f:
    corr_basc122 = pickle.load(p_f)
with open('fmri_corr_feature/train_corr_fmri_basc197.pkl', 'rb') as p_f:
    corr_basc197 = pickle.load(p_f)
with open('fmri_corr_feature/train_corr_fmri_craddock_scorr_mean.pkl', 'rb') as p_f:
    corr_craddock = pickle.load(p_f)
with open('fmri_corr_feature/train_corr_fmri_harvard_oxford_cort_prob_2mm.pkl', 'rb') as p_f:
    corr_harvard = pickle.load(p_f)
with open('fmri_corr_feature/train_corr_fmri_power_2011.pkl', 'rb') as p_f:
    corr_power = pickle.load(p_f)


from problem import get_train_data, get_test_data
# Raw data
data_train, labels_train = get_train_data()
#data_test, labels_test = get_test_data()


x_msdl = pd.DataFrame(corr_msdl, index=data_train.index)
x_msdl.columns = ['msdl_{}'.format(i) for i in range(x_msdl.columns.size)]

x_basc064 = pd.DataFrame(corr_basc064, index=data_train.index)
x_basc064.columns = ['basc064_{}'.format(i) for i in range(x_basc064.columns.size)]

x_basc122 = pd.DataFrame(corr_basc122, index=data_train.index)
x_basc122.columns = ['basc122_{}'.format(i) for i in range(x_basc122.columns.size)]

x_basc197 = pd.DataFrame(corr_basc197, index=data_train.index)
x_basc197.columns = ['basc197_{}'.format(i) for i in range(x_basc197.columns.size)]

x_craddock = pd.DataFrame(corr_craddock, index=data_train.index)
x_craddock.columns = ['craddock_{}'.format(i) for i in range(x_craddock.columns.size)]

x_harvard = pd.DataFrame(corr_harvard, index=data_train.index)
x_harvard.columns = ['harvard_{}'.format(i) for i in range(x_harvard.columns.size)]

x_power = pd.DataFrame(corr_power, index=data_train.index)
x_power.columns = ['power_{}'.format(i) for i in range(x_power.columns.size)]


X_fmri_all = pd.concat([x_msdl, x_basc064, x_basc122, x_basc197,
                        x_craddock, x_harvard, x_power], axis=1) #shape: (1127, 96164)


nan_ind = np.unique(np.where(np.isnan(X_fmri_all))[0]) # Which subject had nan..

y = np.delete(labels_train, nan_ind) # Select appropriate y

X_fmri = X_fmri_all.dropna() # remove nan -> shape: (1103, 96164) # QC ?



############
# Training #
############

# Only for fMRI data
# 1. Combine everything

# Feature selection by Random Forest


# Split train/test 8:2
from sklearn.model_selection import StratifiedShuffleSplit

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=42)
    return cv.split(X, y)

for train, test in get_cv(X_fmri, y):
    x_train = X_fmri.iloc[train]  # train data set
    y_train = y[train]  # label train
    x_test = X_fmri.iloc[test]  # test set
    y_test = y[test]  # label test


# RF for feature selection
from sklearn.ensemble import RandomForestClassifier as RF

rf = RF(n_estimators=1000, random_state=0)
fit1 = rf.fit(x_train, y_train)


indices = np.argsort(rf.feature_importances_)[::-1] # Importance features
top_feature_ind = 200
selected_X_fmri_train = x_train.iloc[:, indices[:top_feature_ind]] # top 200
selected_X_fmri_test = x_test.iloc[:, indices[:top_feature_ind]]


# Training model using SVM or Logistic Regression
# from final.Classifier import Classifier

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# StandardScaler : Standardize features by removing the mean and scaling to unit variance
# refer to http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

class Classifier(BaseEstimator):
    def __init__(self):
        # self.params = {'svc__C': [0.01, 0.1, 1, 10, 1000],
        #                'svc__gamma': [10, 5, 1, 0,1, 0.01, 0.0001]}
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


logistic_model1_train = Classifier().fit(selected_X_fmri_train, y_train)
logistic_model1_final = LR(C=0.01, penalty='l2').fit(selected_X_fmri_train, y_train)


#svc_model1_train = Classifier().fit(selected_X_fmri_train, y_train)
#svc_model1_final = SVC(kernel='linear', C=0.01, gamma=10, probability=True)
#svc_model1_final.fit(selected_X_fmri_train, y_train)



##############
# Prediction #
##############

y_pred_proba = logistic_model1_final.predict_proba(selected_X_fmri_test)[:,1]
y_pred = logistic_model1_final.predict(selected_X_fmri_test)

#y_pred_proba = svc_model1_final.predict_proba(selected_X_fmri_test)[:,1]
#y_pred = svc_model1_final.predict(selected_X_fmri_test)

# Accuracy
from sklearn.metrics import roc_auc_score, accuracy_score

ROC_AUC = roc_auc_score(y_test, y_pred_proba)
Accuracy = accuracy_score(y_test, y_pred)

# Area Under the Curve of the Receiver Operating Characteristic (ROC-AUC)
print("ROC-AUC:", ROC_AUC, ", Accuracy=", Accuracy)


