
import sys
import os
import numpy as np
import pandas as pd


################
### Get Data ###
################
# Prepare data for group analysis
from problem import get_train_data, get_test_data
data_train, labels_train = get_train_data()
data_test, labels_test = get_test_data()


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

