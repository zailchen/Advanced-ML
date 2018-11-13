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
    def __init__(self, atlas_names='msdl', kind='tangent', vectorize=True, discard_diagonal=True):
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