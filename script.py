
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

# merge the train and test data
data_train = pd.concat([data_train, data_test])
labels_train = np.concatenate([labels_train, labels_test])

# import FeatureExtractor
#from GroupAnalysis.function.FeatureExtractor_groupAnalysis import FeatureExtractor

# fMRI data (7 atlas)
atlas_names = [
            'msdl'
            ,'basc064', 'basc122', 'basc197',
            'harvard_oxford_cort_prob_2mm', 'craddock_scorr_mean',
            'power_2011']

# Or just one atlas -- Try different combinations !
# atlas_names = ['craddock_scorr_mean']

# Train data for all atlas 2D and 3D, correlation
X_train_correlation_2d_all_atlas = FeatureExtractor(atlas_names=atlas_names, kind='correlation').fit_transform(data_train, labels_train) # shape : (subject#, #pairs connectivity of lower triangle) = (1127, 96164)
X_train_correlation_3d_all_atlas = FeatureExtractor(atlas_names=atlas_names, kind='correlation', vectorize=False).fit_transform(data_train, labels_train) # len(X_train_correlation_3d_all_atlas) = 7. This means we have 7 different atlas and each one has shape of (subject#, width of connectivity matrix, height of connectivity matrix)

# train data for Each atlas, 2D (use lower traingular part of matrix and stretch to 1D)
X_msdl_correlation_2d = X_train_correlation_2d_all_atlas[X_train_correlation_2d_all_atlas.columns[X_train_correlation_2d_all_atlas.columns.str.contains('msdl')]] # shape: (1127, 741)
X_basc064_correlation_2d = X_train_correlation_2d_all_atlas[X_train_correlation_2d_all_atlas.columns[X_train_correlation_2d_all_atlas.columns.str.contains('basc064')]] # shape: (1127, 2016)
X_basc122_correlation_2d = X_train_correlation_2d_all_atlas[X_train_correlation_2d_all_atlas.columns[X_train_correlation_2d_all_atlas.columns.str.contains('basc122')]] # shape: (1127, 7381)
X_basc197_correlation_2d = X_train_correlation_2d_all_atlas[X_train_correlation_2d_all_atlas.columns[X_train_correlation_2d_all_atlas.columns.str.contains('basc197')]] # shape: (1127, 19306)
X_harvard_correlation_2d = X_train_correlation_2d_all_atlas[X_train_correlation_2d_all_atlas.columns[X_train_correlation_2d_all_atlas.columns.str.contains('harvard')]] # shape: (1127, 1128)
X_crad_correlation_2d = X_train_correlation_2d_all_atlas[X_train_correlation_2d_all_atlas.columns[X_train_correlation_2d_all_atlas.columns.str.contains('crad')]]  # shape: (1127, 30876)
X_power_correlation_2d = X_train_correlation_2d_all_atlas[X_train_correlation_2d_all_atlas.columns[X_train_correlation_2d_all_atlas.columns.str.contains('power')]] # shape: (1127, 34716)

# train data for Each atlas, 3D (2D matrix)
X_msdl_correlation_3d = X_train_correlation_3d_all_atlas[atlas_names.index('msdl')] # shape: (1127, 39, 39)
X_basc064_correlation_3d = X_train_correlation_3d_all_atlas[atlas_names.index('basc064')] # shape: (1127, 64, 64)
X_basc122_correlation_3d = X_train_correlation_3d_all_atlas[atlas_names.index('basc122')] # shape: (1127, 122, 122)
X_basc197_correlation_3d = X_train_correlation_3d_all_atlas[atlas_names.index('basc197')] # shape: (1127, 197, 197)
X_harvard_correlation_3d = X_train_correlation_3d_all_atlas[atlas_names.index('harvard_oxford_cort_prob_2mm')] # shape: (1127, 48, 48)
X_crad_correlation_3d = X_train_correlation_3d_all_atlas[atlas_names.index('craddock_scorr_mean')] # shape: (1127, 249, 249)
X_power_correlation_3d = X_train_correlation_3d_all_atlas[atlas_names.index('power_2011')] # shape: (1127, 264, 264)


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

