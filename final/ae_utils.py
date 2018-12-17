#!/usr/bin/env python
import os
import re
import sys
import time
import random
import string
import multiprocessing
import pandas as pd
import numpy as np
import tensorflow as tf
from model import ae


def reset():
    tf.reset_default_graph()
    random.seed(19)
    np.random.seed(19)
    tf.set_random_seed(19)
    
class SafeFormat(dict):
    def __missing__(self, key):
        return "{" + key + "}"

    def __getitem__(self, key):
        if key not in self:
            return self.__missing__(key)
        return dict.__getitem__(self, key)
    
def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result
    
def format_config(s, *d):
    merge_dic = merge_dicts(*d)
    return string.Formatter().vformat(s, [], SafeFormat(merge_dic))


def to_softmax(n_classes, classe):
    sm = [0.0] * n_classes
    sm[int(classe)] = 1.0
    return sm


def load_ae_encoder(input_size, code_size, model_path):
    model = ae(input_size, code_size)
    init = tf.global_variables_initializer()
    try:
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)
            if os.path.isfile(model_path):
                print("Restoring", model_path)
                saver.restore(sess, model_path)
            params = sess.run(model["params"])
            return {"W_enc": params["W_enc"], "b_enc": params["b_enc"]}
    finally:
        reset()


def sparsity_penalty(x, p, coeff):
    p_hat = tf.reduce_mean(tf.abs(x), 0)
    kl = p * tf.log(p / p_hat) + \
        (1 - p) * tf.log((1 - p) / (1 - p_hat))
    return coeff * tf.reduce_sum(kl)
