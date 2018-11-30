import numpy as np

np.random.seed(42)

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier

import keras    # refer to https://keras.io/
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D


def _preprocess_data_1D(X_df, y = None):
    # Scale data
    X = StandardScaler().fit_transform(X_df)
    # Reshape data
    X = X.reshape(X.shape[0], X.shape[1], 1).astype('float32')

    if y is not None:
        return np.array(X), np.array(keras.utils.to_categorical(y)) # convert class vectors to binary class matrices
    else:
        return np.array(X)


class Conv_1d(BaseEstimator):
    def __init__(self, nfilt=15, kernel_size=3, strides=1, pool_size=4, pool_strides=4, fc=15, epochs=1000):
        self.nfilt = nfilt
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.fc = fc
        self.epochs = epochs
        self.model_ = None

    def fit(self, X_df, y, **kwargs):
        x_train, y_train = _preprocess_data_1D(X_df, y)

        self.set_params(**kwargs)
        model = Sequential()
        model.add(Conv1D(filters=self.nfilt, kernel_size=self.kernel_size, activation='relu', input_shape=x_train.shape[1:3], strides=self.strides))
        model.add(MaxPooling1D(pool_size=self.pool_size, strides=self.pool_strides))
        #model.add(Conv1D(filters=16, kernel_size=5, activation='relu', strides=1))
        #model.add(MaxPooling1D(pool_size=4, strides=4))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(self.fc, activation='sigmoid'))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.adam(),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=self.epochs,
                  batch_size=32, verbose=1, shuffle=True)
        self.model_ = model
        return self

    def predict(self, X):
        X_test = _preprocess_data_1D(X)
        y_pred = self.model_.predict(X_test)
        probs = np.mean(np.reshape(y_pred, (-1, 2)), axis=1, keepdims=True)
        return (probs > 0.5).astype('int32')

    def predict_proba(self, X):
        X_test = _preprocess_data_1D(X)
        y_pred = self.model_.predict(X_test)
        probs = np.mean(np.reshape(y_pred, (-1, 2)), axis=1, keepdims=True)
        probs = np.hstack([1 - probs, probs])
        return probs


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf_anat = make_pipeline(StandardScaler(),
                                      BaggingClassifier(base_estimator=LogisticRegression(C=0.1),
                                                        n_estimators=50,
                                                        max_samples=0.8,
                                                        max_features=0.8,
                                                        random_state=42))
        self.clf_fc1 = make_pipeline(StandardScaler(),
                                     BaggingClassifier(base_estimator=LogisticRegression(C=0.1),
                                                       n_estimators=50,
                                                       max_samples=0.8,
                                                       max_features=0.8,
                                                       random_state=42))
        self.clf_fc2 = make_pipeline(StandardScaler(),
                                     BaggingClassifier(base_estimator=LogisticRegression(C=0.1),
                                                       n_estimators=50,
                                                       max_samples=0.8,
                                                       max_features=0.8,
                                                       random_state=42))
        self.clf_nn1 = Conv_1d()
        self.clf_nn2 = Conv_1d() # try different parameters

        self.meta_clf = make_pipeline(StandardScaler(), LogisticRegression())

    def fit(self, X, y):
        X_anat = X.iloc[:,X.columns.str.contains('anatomy')]
        X_fc1 = X.iloc[:,X.columns.str.contains('crad')]
        X_fc2 = X.iloc[:,X.columns.str.contains('basc122')]
        X_nn1 = np.array(X.iloc[:,X.columns.str.contains('power')])
        X_nn2 = np.array(X.iloc[:,X.columns.str.contains('basc197')])

        cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
        y_pred = []
        y_val = []
        for train_idx, validation_idx in cv.split(X, y):
            X_anat_train = X_anat.iloc[train_idx]
            X_anat_validation = X_anat.iloc[validation_idx]
            X_fc1_train = X_fc1.iloc[train_idx]
            X_fc1_validation = X_fc1.iloc[validation_idx]
            X_fc2_train = X_fc2.iloc[train_idx]
            X_fc2_validation = X_fc2.iloc[validation_idx]
            X_nn1_train = X_nn1[train_idx]
            X_nn1_validation = X_nn1[validation_idx]
            X_nn2_train = X_nn2[train_idx]
            X_nn2_validation = X_nn2[validation_idx]

            y_train = y[train_idx]
            y_validation = y[validation_idx]

            self.clf_anat.fit(X_anat_train, y_train)
            self.clf_fc1.fit(X_fc1_train, y_train)
            self.clf_fc2.fit(X_fc2_train, y_train)
            self.clf_nn1.fit(X_nn1_train, y_train)
            self.clf_nn2.fit(X_nn2_train, y_train)

            y_anat_pred = self.clf_anat.predict_proba(X_anat_validation)[:, 0].reshape(-1, 1)
            y_fc1_pred = self.clf_fc1.predict_proba(X_fc1_validation)[:, 0].reshape(-1, 1)
            y_fc2_pred = self.clf_fc2.predict_proba(X_fc2_validation)[:, 0].reshape(-1, 1)
            y_nn1_pred = self.clf_nn1.predict_proba(X_nn1_validation)[:, 0].reshape(-1, 1)
            y_nn2_pred = self.clf_nn2.predict_proba(X_nn2_validation)[:, 0].reshape(-1, 1)

            y_pred.extend(np.concatenate([y_anat_pred,
                                          y_fc1_pred,
                                          y_fc2_pred,
                                          y_nn1_pred,
                                          y_nn2_pred,
                                          y_anat_pred * y_fc1_pred,
                                          y_anat_pred * y_nn1_pred,
                                          y_fc2_pred * y_nn2_pred], axis=1))
            y_val.extend(y_validation)

        self.clf_anat.fit(X_anat, y)
        self.clf_fc1.fit(X_fc1, y)
        self.clf_fc2.fit(X_fc2, y)
        self.clf_nn1.fit(X_nn1, y)
        self.clf_nn2.fit(X_nn2, y)

        self.meta_clf.fit(y_pred, y_val)
        return self

    def predict(self, X):
        X_anat = X.iloc[:,X.columns.str.contains('anatomy')]
        X_fc1 = X.iloc[:,X.columns.str.contains('crad')]
        X_fc2 = X.iloc[:,X.columns.str.contains('basc122')]
        X_nn1 = np.array(X.iloc[:,X.columns.str.contains('power')])
        X_nn2 = np.array(X.iloc[:,X.columns.str.contains('basc197')])

        y_anat_pred = self.clf_anat.predict_proba(X_anat)[:, 0].reshape(-1, 1)
        y_fc1_pred = self.clf_fc1.predict_proba(X_fc1)[:, 0].reshape(-1, 1)
        y_fc2_pred = self.clf_fc2.predict_proba(X_fc2)[:, 0].reshape(-1, 1)
        y_nn1_pred = self.clf_nn1.predict_proba(X_nn1)[:, 0].reshape(-1, 1)
        y_nn2_pred = self.clf_nn2.predict_proba(X_nn2)[:, 0].reshape(-1, 1)

        return self.meta_clf.predict(
            np.concatenate([y_anat_pred,
                            y_fc1_pred,
                            y_fc2_pred,
                            y_nn1_pred,
                            y_nn2_pred,
                            y_anat_pred * y_fc1_pred,
                            y_anat_pred * y_nn1_pred,
                            y_fc2_pred * y_nn2_pred], axis=1))

    def predict_proba(self, X):
        X_anat = X.iloc[:,X.columns.str.contains('anatomy')]
        X_fc1 = X.iloc[:,X.columns.str.contains('crad')]
        X_fc2 = X.iloc[:,X.columns.str.contains('basc122')]
        X_nn1 = np.array(X.iloc[:,X.columns.str.contains('power')])
        X_nn2 = np.array(X.iloc[:,X.columns.str.contains('basc197')])

        y_anat_pred = self.clf_anat.predict_proba(X_anat)[:, 0].reshape(-1, 1)
        y_fc1_pred = self.clf_fc1.predict_proba(X_fc1)[:, 0].reshape(-1, 1)
        y_fc2_pred = self.clf_fc2.predict_proba(X_fc2)[:, 0].reshape(-1, 1)
        y_nn1_pred = self.clf_nn1.predict_proba(X_nn1)[:, 0].reshape(-1, 1)
        y_nn2_pred = self.clf_nn2.predict_proba(X_nn2)[:, 0].reshape(-1, 1)

        return self.meta_clf.predict_proba(
            np.concatenate([y_anat_pred,
                            y_fc1_pred,
                            y_fc2_pred,
                            y_nn1_pred,
                            y_nn2_pred,
                            y_anat_pred * y_fc1_pred,
                            y_anat_pred * y_nn1_pred,
                            y_fc2_pred * y_nn2_pred], axis=1))

