import numpy as np
import pandas as pd
np.random.seed(42)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from nilearn.connectome import ConnectivityMeasure
from nilearn.signal import clean


def _load_fmri_motion_correction(fmri_filenames, fmri_motions):
    fmri = []
    for (i, j) in zip(fmri_filenames, fmri_motions):
        x = pd.read_csv(i, header=None).values
        y = np.loadtxt(j)
        fmri.append(clean(x, detrend=False, standardize=True, confounds=y))
    return np.array(fmri)

# fmri_filenames = data_test['fmri_msdl']
# fmri_motions = data_test['fmri_motions']
# fmri = _load_fmri_motion_correction(fmri_filenames, fmri_motions)

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, atlas_names='msdl', kind='correlation', vectorize=True, discard_diagonal=True):
        # matrix kind ref : http://nilearn.github.io/modules/generated/nilearn.connectome.ConnectivityMeasure.html
        # Example ref: http://nilearn.github.io/auto_examples/03_connectivity/plot_group_level_connectivity.html#sphx-glr-auto-examples-03-connectivity-plot-group-level-connectivity-py
        # Correlation:  It models the full (marginal) connectivity between pairwise ROIs. It computes individual correlation matrices
        # Partial Correlation: Direct connections. Most of direct connections are weaker than full connections.
        # Tangent: Extract subjects variabilities around a robust group connectivity. This use both correlations and partial correlations to
        #          capture reproducible connectivity patterns at the group-level and build a robust group connectivity matrix.


        # Are we discard diagonal values in matrix??

        # self.atlas_names = [
        #     'msdl'
        #     ,'basc064', 'basc122', 'basc197',
        #     'harvard_oxford_cort_prob_2mm', 'craddock_scorr_mean',
        #     'power_2011'
        # ]

        self.atlas_names = atlas_names
        self.transformer = dict()
        self.vectorize = vectorize   # vectorize = True -> connectivity matrices are reshaped into 1D arrays and only their flattened lower triangular parts are returned.
        self.diagonal = discard_diagonal
        self.kind = kind    # try all ['correlation', 'partial correlation', 'tangent', 'covariance', 'precision']

        for atlas in self.atlas_names:
            self.transformer[atlas] = make_pipeline(
                    FunctionTransformer(func=_load_fmri_motion_correction, kw_args={'fmri_motions': None}, validate=False),
                    ConnectivityMeasure(kind=self.kind, vectorize=self.vectorize, discard_diagonal=self.diagonal))
                # if vectorize = False, it returns 2D array. e.g. for MSDL, (1127, 39, 39)


    def fit(self, X_df, y):

        for atlas in self.atlas_names:
            fmri_filenames = X_df['fmri_{}'.format(atlas)]
            self.transformer[atlas].named_steps['functiontransformer'].set_params(
                kw_args={'fmri_motions': X_df['fmri_motions']})
            self.transformer[atlas].fit(fmri_filenames, y)

        return self


    def transform(self, X_df):
        # fMRI features
        X_fmri = []
        if self.vectorize == True:
            for atlas in self.atlas_names:
                fmri_filenames = X_df['fmri_{}'.format(atlas)]
                self.transformer[atlas].named_steps['functiontransformer'].set_params(
                    kw_args={'fmri_motions': X_df['fmri_motions']})
                X_connectome = self.transformer[atlas].transform(fmri_filenames)

                columns = ['connectome_{}_{}'.format(atlas, i) for i in range(X_connectome.shape[1])]
                X_connectome = pd.DataFrame(data=X_connectome, index=X_df.index, columns=columns)

                X_fmri.append(X_connectome)

            X_fmri = pd.concat(X_fmri, axis=1)

        else:
            for atlas in self.atlas_names:
                fmri_filenames = X_df['fmri_{}'.format(atlas)]
                self.transformer[atlas].named_steps['functiontransformer'].set_params(
                    kw_args={'fmri_motions': X_df['fmri_motions']})
                X_connectome = self.transformer[atlas].transform(fmri_filenames)

                X_fmri.append(X_connectome)

        return X_fmri


# Save data
from problem import get_train_data
data_train, labels_train = get_train_data()

# fMRI data (7 atlas)
atlas_names = [
            'msdl',
            'basc064', 'basc122', 'basc197',
            'harvard_oxford_cort_prob_2mm', 'craddock_scorr_mean',
            'power_2011'
            ]

# Train data for all atlas 2D and 3D, correlation
X_train_correlation_1d_all_atlas = FeatureExtractor(atlas_names=atlas_names, kind='correlation').fit_transform(data_train, labels_train) # shape : (subject#, #pairs connectivity of lower triangle) = (1127, 96164)
X_train_correlation_2d_all_atlas = FeatureExtractor(atlas_names=atlas_names, kind='correlation', vectorize=False).fit_transform(data_train, labels_train) # len(X_train_correlation_3d_all_atlas) = 7. This means we have 7 different atlas and each one has shape of (subject#, width of connectivity matrix, height of connectivity matrix)


# train data for Each atlas, 1D (use lower traingular part of matrix and stretch to 1D)
X_msdl_corr_1d = X_train_correlation_1d_all_atlas[X_train_correlation_1d_all_atlas.columns[X_train_correlation_1d_all_atlas.columns.str.contains('msdl')]] # shape: (1127, 741)
X_basc064_corr_1d = X_train_correlation_1d_all_atlas[X_train_correlation_1d_all_atlas.columns[X_train_correlation_1d_all_atlas.columns.str.contains('basc064')]] # shape: (1127, 2016)
X_basc122_corr_1d = X_train_correlation_1d_all_atlas[X_train_correlation_1d_all_atlas.columns[X_train_correlation_1d_all_atlas.columns.str.contains('basc122')]] # shape: (1127, 7381)
X_basc197_corr_1d  = X_train_correlation_1d_all_atlas[X_train_correlation_1d_all_atlas.columns[X_train_correlation_1d_all_atlas.columns.str.contains('basc197')]] # shape: (1127, 19306)
X_harvard_corr_1d  = X_train_correlation_1d_all_atlas[X_train_correlation_1d_all_atlas.columns[X_train_correlation_1d_all_atlas.columns.str.contains('harvard')]] # shape: (1127, 1128)
X_crad_corr_1d  = X_train_correlation_1d_all_atlas[X_train_correlation_1d_all_atlas.columns[X_train_correlation_1d_all_atlas.columns.str.contains('crad')]]  # shape: (1127, 30876)
X_power_corr_1d  = X_train_correlation_1d_all_atlas[X_train_correlation_1d_all_atlas.columns[X_train_correlation_1d_all_atlas.columns.str.contains('power')]] # shape: (1127, 34716)

# train data for Each atlas, 2D
X_msdl_corr_2d = X_train_correlation_2d_all_atlas[atlas_names.index('msdl')] # shape: (1127, 39, 39)
X_basc064_corr_2d = X_train_correlation_2d_all_atlas[atlas_names.index('basc064')] # shape: (1127, 64, 64)
X_basc122_corr_2d = X_train_correlation_2d_all_atlas[atlas_names.index('basc122')] # shape: (1127, 122, 122)
X_basc197_corr_2d = X_train_correlation_2d_all_atlas[atlas_names.index('basc197')] # shape: (1127, 197, 197)
X_harvard_corr_2d = X_train_correlation_2d_all_atlas[atlas_names.index('harvard_oxford_cort_prob_2mm')] # shape: (1127, 48, 48)
X_crad_corr_2d = X_train_correlation_2d_all_atlas[atlas_names.index('craddock_scorr_mean')] # shape: (1127, 249, 249)
X_power_corr_2d = X_train_correlation_2d_all_atlas[atlas_names.index('power_2011')] # shape: (1127, 264, 264)


import pickle

X_msdl_corr_1d.to_pickle('final/fmri_corr_features/1D/msdl_corr_1d.pkl')
X_basc064_corr_1d.to_pickle('final/fmri_corr_features/1D/basc064_corr_1d.pkl')
X_basc122_corr_1d.to_pickle('final/fmri_corr_features/1D/basc122_corr_1d.pkl')
X_basc197_corr_1d.to_pickle('final/fmri_corr_features/1D/basc197_corr_1d.pkl')
X_harvard_corr_1d.to_pickle('final/fmri_corr_features/1D/harvard_corr_1d.pkl')
X_crad_corr_1d.to_pickle('final/fmri_corr_features/1D/crad_corr_1d.pkl')
X_power_corr_1d.to_pickle('final/fmri_corr_features/1D/power_corr_1d.pkl')

np.save('final/fmri_corr_features/2D/msdl_corr_2d', X_msdl_corr_2d)
np.save('final/fmri_corr_features/2D/basc064_corr_2d', X_basc064_corr_2d)
np.save('final/fmri_corr_features/2D/basc122_corr_2d', X_basc122_corr_2d)
np.save('final/fmri_corr_features/2D/basc197_corr_2d', X_basc197_corr_2d)
np.save('final/fmri_corr_features/2D/harvard_corr_2d', X_harvard_corr_2d)
np.save('final/fmri_corr_features/2D/crad_corr_2d', X_crad_corr_2d)
np.save('final/fmri_corr_features/2D/power_corr_2d', X_power_corr_2d)














