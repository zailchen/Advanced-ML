# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import pickle
from problem import get_train_data, get_test_data

data_train, labels_train = get_train_data()
data_test, labels_test = get_test_data()

data_combine = pd.concat([data_train, data_test])
labels_combine = np.concatenate([labels_train, labels_test])

n = len(labels_combine)
train_ratio = 0.8
random.seed(42)
index = np.arange(n)
random.shuffle(index)

train_id = index[:int(n*0.8)]
test_id = index[int(n*0.8):]

X_train = data_combine.iloc[train_id,:]
y_train = labels_combine[train_id]

X_test = data_combine.iloc[test_id,:]
y_test = labels_combine[test_id]

with open('data/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('data/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('data/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('data/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
