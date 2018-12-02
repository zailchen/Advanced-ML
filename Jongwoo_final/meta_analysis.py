
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

################
##### DATA #####
################

'''
We first load data from problem.py that the code is provided. From get_train_data function, we read
all training data which has 1127 subjects and 220 features. Those features include anatomy feature, functional feature and 
subject's site, age, sex. 

For Structural MRI data (Anatomy), a set of structural features have been extracted for each subject:
 (i) normalized brain volume computed using subcortical segmentation of FreeSurfer and 
 (ii) cortical thickness and area for right and left hemisphere of FreeSurfer.
The data includes 208 features that the column name starts with 'anatomy_'. 
Note that the column anatomy_select contain a label affected during a manual quality check
(i.e. 0 and 3 reject, 1 accept, 2 accept with reserve). This column can be used during training to exclude noisy data for instance.

For resting-state functional MRI data, each subject comes with fMRI signals extracted on different brain parcellations and atlases, 
and a set of confound signals. Those brain atlases and parcellations are: (i) BASC parcellations with 64, 122, and 197 regions (Bellec 2010), 
(ii) Ncuts parcellations (Craddock 2012), (iii) Harvard-Oxford anatomical parcellations, (iv) MSDL functional atlas (Varoquaux 2011),
 and (v) Power atlas (Power 2011). The script used for this extraction can be found 'extract_time_series.py' in preprocessing folder.
The size of .csv file is [200, regionNum], which is a time series data, e.g., for parcellations with 64 would be [200,64].
For the column 'fmri_select', 0 means bad and 1 means good. 
Eventually we need to filter anatomy_select and fmri_select that is 1

'''

# Prepare data for group analysis
from problem import get_train_data
data_train, labels_train = get_train_data()


# Load data

# fMRI data (7 atlas)
# Only 1D data for this analysis
# We are focus on correlation connectivity measure for this analysis.

import pickle

with open('fmri_corr_features/1D/msdl_corr_1d.pkl', 'rb') as p_f:
    msdl_corr = pickle.load(p_f)
with open('fmri_corr_features/1D/basc064_corr_1d.pkl', 'rb') as p_f:
    basc064_corr = pickle.load(p_f)
with open('fmri_corr_features/1D/basc122_corr_1d.pkl', 'rb') as p_f:
    basc122_corr = pickle.load(p_f)
with open('fmri_corr_features/1D/basc197_corr_1d.pkl', 'rb') as p_f:
    basc197_corr = pickle.load(p_f)
with open('fmri_corr_features/1D/crad_corr_1d.pkl', 'rb') as p_f:
    crad_corr = pickle.load(p_f)
with open('fmri_corr_features/1D/harvard_corr_1d.pkl', 'rb') as p_f:
    harvard_corr = pickle.load(p_f)
with open('fmri_corr_features/1D/power_corr_1d.pkl', 'rb') as p_f:
    power_corr = pickle.load(p_f)


# Anatomy T1 data
X_anatomy = data_train[[col for col in data_train.columns if col.startswith('anatomy')]]  # read anatomy data
X_anatomy = X_anatomy.drop(columns=['anatomy_Left-WM-hypointensities', 'anatomy_Right-WM-hypointensities',
                                    'anatomy_Left-non-WM-hypointensities',
                                    'anatomy_Right-non-WM-hypointensities',
                                    'anatomy_select']) # Exclude unnecessary columns. All of them are zero in the excel file '../data/anatomy.csv'

X_anatomy = pd.DataFrame(X_anatomy, index=data_train.index) # shape: (1127, 203) #ended up 203 columns

# replace '-' with '_' because regression needs '_' instead of '-'
bad_cols = [col for col in X_anatomy.columns if '-' in col]
X_anatomy.columns = X_anatomy.columns.str.replace('-', '_')


# Demographic features
X_demo = data_train[['participants_age', 'participants_sex']] # shape: (1127, 2)


# testing adjust age (~18 years old)
X_demo_adolescent = X_demo[X_demo.participants_age < 18]


# Input data for classification
# You can try different combinations here
X = pd.concat([X_anatomy, crad_corr, basc122_corr, power_corr, basc197_corr], axis=1)
y = labels_train

# Split train/test 8:2
from sklearn.model_selection import StratifiedShuffleSplit

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=42)
    return cv.split(X, y)

for train, test in get_cv(X, y):
    x_train = X.iloc[train]  # train data set
    y_train = y[train]  # label train
    x_test = X.iloc[test]  # test set
    y_test = y[test]  # label test


##############
## Analysis ##
##############

# Load meta_classifier from './Classifier'
from Jongwoo_final.Classifier.meta_classifier import Classifier

fit1 = Classifier().fit(x_train, y_train)


# Prediction

y_pred_proba = fit1.predict_proba(x_test)[:,1]
y_pred = fit1.predict(x_test)

# Evaluate
from sklearn.metrics import roc_auc_score, accuracy_score

ROC_AUC = roc_auc_score(y_test, y_pred_proba)
Accuracy = accuracy_score(y_test, y_pred)

# Area Under the Curve of the Receiver Operating Characteristic (ROC-AUC)
print("ROC-AUC:", ROC_AUC, ", Accuracy=", Accuracy)


