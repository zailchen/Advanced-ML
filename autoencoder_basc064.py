# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import pickle
import random
import tensorflow as tf
from nilearn.signal import clean
from sklearn.metrics import roc_auc_score
from utils import (format_config, sparsity_penalty, reset, to_softmax, load_ae_encoder)
from model import ae, nn
from problem import get_train_data
import random
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def run_autoencoder1(experiment,
                     X_train, y_train, X_valid, y_valid, X_test, y_test,
                     model_path, code_size=1000):
    """
    Run the first autoencoder.
    It takes the original data dimensionality and compresses it into `code_size`
    """

    # Hyperparameters
    learning_rate = 0.0001
    sparse = True  # Add sparsity penalty
    sparse_p = 0.2
    sparse_coeff = 0.5
    corruption = 0.75  # Data corruption ratio for denoising
    ae_enc = tf.nn.tanh  # Tangent hyperbolic
    ae_dec = None  # Linear activation

    training_iters = 100
    batch_size = 64
    n_classes = 2

    if os.path.isfile(model_path) or \
            os.path.isfile(model_path + ".meta"):
        return

    # Create model and add sparsity penalty (if requested)
    model = ae(X_train.shape[1], code_size, corruption=corruption, enc=ae_enc, dec=ae_dec)
    if sparse:
        model["cost"] += sparsity_penalty(model["encode"], sparse_p, sparse_coeff)

    # Use GD for optimization of model cost
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.8).minimize(model["cost"])

    # Initialize Tensorflow session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Define model saver
        saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)

        # Initialize with an absurd cost for model selection
        prev_costs = np.array([9999999999] * 3)

        for epoch in range(training_iters):

            # randomly shuffle data
            index = np.arange(X_train.shape[0])
            random.shuffle(index)

            X_train = X_train[index,]
            y_train = y_train[index]

            # Break training set into batches
            batches = range(len(X_train) // batch_size)
            costs = np.zeros((len(batches), 3))

            for ib in batches:
                # Compute start and end of batch from training set data array
                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size

                # Select current batch
                batch_xs, batch_ys = X_train[from_i:to_i], y_train[from_i:to_i]

                # Run optimization and retrieve training cost
                _, cost_train = sess.run(
                    [optimizer, model["cost"]],
                    feed_dict={
                        model["input"]: batch_xs
                    }
                )

                # Compute validation cost
                cost_valid = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: X_valid
                    }
                )

                # Compute test cost
                cost_test = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: X_test
                    }
                )

                costs[ib] = [cost_train, cost_valid, cost_test]

            # Compute the average costs from all batches
            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs

            # Pretty print training info
            print(format_config(
                "Exp={experiment}, Model=ae1, Iter={epoch:5d}, Cost={cost_train:.6f} {cost_valid:.6f} {cost_test:.6f}",
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "cost_train": cost_train,
                    "cost_valid": cost_valid,
                    "cost_test": cost_test,
                }
            ))

            # Save better model if optimization achieves a lower cost
            if cost_valid < prev_costs[1]:
                print("Saving better model")
                saver.save(sess, model_path)
                prev_costs = costs
            else:
                print


def run_autoencoder2(experiment,
                     X_train, y_train, X_valid, y_valid, X_test, y_test,
                     model_path, prev_model_path,
                     code_size=600, prev_code_size=1000):
    """
    Run the second autoencoder.
    It takes the dimensionality from first autoencoder and compresses it into the new `code_size`
    Firstly, we need to convert original data to the new projection from autoencoder 1.
    """

    if os.path.isfile(model_path) or \
            os.path.isfile(model_path + ".meta"):
        return

    # Convert training, validation and test set to the new representation
    prev_model = ae(X_train.shape[1], prev_code_size,
                    corruption=0.0,  # Disable corruption for conversion
                    enc=tf.nn.tanh, dec=None)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver(prev_model["params"], write_version=tf.train.SaverDef.V2)
        if os.path.isfile(prev_model_path):
            saver.restore(sess, prev_model_path)
        X_train = sess.run(prev_model["encode"], feed_dict={prev_model["input"]: X_train})
        X_valid = sess.run(prev_model["encode"], feed_dict={prev_model["input"]: X_valid})
        X_test = sess.run(prev_model["encode"], feed_dict={prev_model["input"]: X_test})
    del prev_model

    reset()

    # Hyperparameters
    learning_rate = 0.002
    corruption = 0.68
    ae_enc = tf.nn.tanh
    ae_dec = None

    training_iters = 100
    batch_size = 50
    n_classes = 2

    # Load model
    model = ae(prev_code_size, code_size, corruption=corruption, enc=ae_enc, dec=ae_dec)

    # Use GD for optimization of model cost
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9).minimize(model["cost"])

    # Initialize Tensorflow session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Define model saver
        saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)

        # Initialize with an absurd cost for model selection
        prev_costs = np.array([9999999999] * 3)

        # Iterate Epochs
        for epoch in range(training_iters):

            # randomly shuffle data
            index = np.arange(X_train.shape[0])
            random.shuffle(index)

            X_train = X_train[index,]
            y_train = y_train[index]

            # Break training set into batches
            batches = range(len(X_train) // batch_size)
            costs = np.zeros((len(batches), 3))

            for ib in batches:
                # Compute start and end of batch from training set data array
                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size

                # Select current batch
                batch_xs, batch_ys = X_train[from_i:to_i], y_train[from_i:to_i]

                # Run optimization and retrieve training cost
                _, cost_train = sess.run(
                    [optimizer, model["cost"]],
                    feed_dict={
                        model["input"]: batch_xs
                    }
                )

                # Compute validation cost
                cost_valid = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: X_valid
                    }
                )

                # Compute test cost
                cost_test = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: X_test
                    }
                )

                costs[ib] = [cost_train, cost_valid, cost_test]

            # Compute the average costs from all batches
            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs

            # Pretty print training info
            print(format_config(
                "Exp={experiment}, Model=ae2, Iter={epoch:5d}, Cost={cost_train:.6f} {cost_valid:.6f} {cost_test:.6f}",
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "cost_train": cost_train,
                    "cost_valid": cost_valid,
                    "cost_test": cost_test,
                }
            ))

            # Save better model if optimization achieves a lower cost
            if cost_valid < prev_costs[1]:
                print("Saving better model")
                saver.save(sess, model_path)
                prev_costs = costs
            else:
                print


def run_finetuning(experiment,
                   X_train, y_train, X_valid, y_valid, X_test, y_test,
                   model_path, prev_model_1_path, prev_model_2_path,
                   code_size_1=1000, code_size_2=600):
    """
    Run the pre-trained NN for fine-tuning, using first and second autoencoders' weights
    """

    # Hyperparameters
    learning_rate = 0.0003
    dropout_1 = 0.4
    dropout_2 = 0.6
    # initial_momentum = 0.1
    # final_momentum = 0.9  # Increase momentum along epochs to avoid fluctiations
    # saturate_momentum = 100

    training_iters = 150
    start_saving_at = 5
    batch_size = 32
    n_classes = 2

    if os.path.isfile(model_path) or \
            os.path.isfile(model_path + ".meta"):
        return

    # Convert output to one-hot encoding
    y_train = np.array([to_softmax(n_classes, y) for y in y_train])
    y_valid = np.array([to_softmax(n_classes, y) for y in y_valid])
    y_test = np.array([to_softmax(n_classes, y) for y in y_test])

    # Load pretrained encoder weights
    ae1 = load_ae_encoder(X_train.shape[1], code_size_1, prev_model_1_path)
    ae2 = load_ae_encoder(code_size_1, code_size_2, prev_model_2_path)

    # Initialize NN model with the encoder weights
    model = nn(X_train.shape[1], n_classes, [
        {"size": code_size_1, "actv": tf.nn.tanh},
        {"size": code_size_2, "actv": tf.nn.tanh},
    ], [
                   {"W": ae1["W_enc"], "b": ae1["b_enc"]},
                   {"W": ae2["W_enc"], "b": ae2["b_enc"]},
               ])

    # Place GD + momentum optimizer
    # model["momentum"] = tf.placeholder("float32")
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9).minimize(model["cost"])

    # Place Adam optimizer
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model["cost"]) # cross entropy

    # Make prediction and Compute accuracies
    logits = model["output"]
    pred = tf.argmax(model["output"], 1)
    correct_prediction = tf.equal(pred, tf.argmax(model["expected"], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Store prediction
    best_prediction = None
    best_logits = None
    acc_list = np.zeros((training_iters, 3))
    loss_list = np.zeros((training_iters, 3))
    auc_list = np.zeros((training_iters, 3))

    # Initialize Tensorflow session
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)

        # Define model saver
        saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)

        # Initialize with an absurd cost, accuracy, auc for model selection
        best_acc = 0.0

        # Iterate Epochs
        for epoch in range(training_iters):

            # randomly shuffle data
            index = np.arange(X_train.shape[0])
            random.shuffle(index)

            X_train = X_train[index,]
            y_train = y_train[index]

            # exclude case that are all 1 or all 0
            if 0 in np.sum(y_train, axis=0):
                continue

            # Break training set into batches
            batches = range(len(X_train) // batch_size)
            costs = np.zeros((len(batches), 3))
            accs = np.zeros((len(batches), 3))
            AUCs = np.zeros((len(batches), 3))
            prediction = None
            logit = None

            # Compute momentum saturation
            # alpha = float(epoch) / float(saturate_momentum)
            # if alpha < 0.:
            #     alpha = 0.
            # if alpha > 1.:
            #     alpha = 1.
            # momentum = initial_momentum * (1 - alpha) + alpha * final_momentum

            for ib in batches:

                # Compute start and end of batch from training set data array
                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size

                # Select current batch
                batch_xs, batch_ys = X_train[from_i:to_i], y_train[from_i:to_i]
                if 0 in np.sum(batch_ys, axis=0):
                    continue

                # Run optimization and retrieve training cost and accuracy
                _, cost_train, acc_train, true_y, pred_y = sess.run(
                    [optimizer, model["cost"], accuracy, model['expected'], model['output']],
                    feed_dict={
                        model["input"]: batch_xs,
                        model["expected"]: batch_ys,
                        model["dropouts"][0]: dropout_1,
                        model["dropouts"][1]: dropout_2
                    }
                )
                # Compute AUC score
                AUC_train = roc_auc_score(np.argmax(true_y, 1), pred_y[:, 1])

                # Compute validation cost and accuracy
                cost_valid, acc_valid, true_y, pred_y = sess.run(
                    [model["cost"], accuracy, model['expected'], model['output']],
                    feed_dict={
                        model["input"]: X_valid,
                        model["expected"]: y_valid,
                        model["dropouts"][0]: 1.0,
                        model["dropouts"][1]: 1.0
                    }
                )
                AUC_valid = roc_auc_score(np.argmax(true_y, 1), pred_y[:, 1])

                # Compute test cost and accuracy
                logit, prediction, cost_test, acc_test, true_y, pred_y = sess.run(
                    [logits, pred, model["cost"], accuracy, model['expected'], model['output']],
                    feed_dict={
                        model["input"]: X_test,
                        model["expected"]: y_test,
                        model["dropouts"][0]: 1.0,
                        model["dropouts"][1]: 1.0
                    }
                )
                AUC_test = roc_auc_score(np.argmax(true_y, 1), pred_y[:, 1])

                costs[ib] = [cost_train, cost_valid, cost_test]
                accs[ib] = [acc_train, acc_valid, acc_test]
                AUCs[ib] = [AUC_train, AUC_valid, AUC_test]

            # Compute the average costs from all batches
            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs
            loss_list[epoch] = cost_train, cost_valid, cost_test

            # Compute the average accuracy from all batches
            accs = accs.mean(axis=0)
            acc_train, acc_valid, acc_test = accs
            acc_list[epoch] = acc_train, acc_valid, acc_test

            # Compute the average AUC for all batches
            AUCs = AUCs.mean(axis=0)
            AUC_train, AUC_valid, AUC_test = AUCs
            auc_list[epoch] = AUC_train, AUC_valid, AUC_test

            # Pretty print training info
            print(format_config(
                "Exp={experiment}, Model=mlp, Iter={epoch:5d}, Acc={acc_train:.6f} {acc_valid:.6f} {acc_test:.6f}, \
                AUC={AUC_train:.6f} {AUC_valid:.6f} {AUC_test:.6f}",
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "acc_train": acc_train,
                    "acc_valid": acc_valid,
                    "acc_test": acc_test,
                    "AUC_train": AUC_train,
                    "AUC_valid": AUC_valid,
                    "AUC_test": AUC_test
                }
            ))

            # Save better model if optimization achieves a lower accuracy
            # and avoid initial epochs because of the fluctuations
            if acc_valid > best_acc and epoch > start_saving_at:
                best_prediction = prediction
                best_logits = logit
                print("Saving better model")
                saver.save(sess, model_path)
                best_acc = acc_valid
    return best_prediction, best_logits, acc_list, loss_list, auc_list




####################################
## Run Experiments
####################################
if __name__ == '__main__':
    prediction = []
    logits = np.zeros((230, 2))

    # data_train, labels_train = get_train_data()

    #feature_list = ['basc064', 'basc122', 'basc197', 'power']

    # for feature in feature_list:
    feature = 'basc064'
    with open('1D-train/' + feature + '_corr_1d.npy', 'rb') as f:
        matrix_train = np.load(f)
    with open('1D-train/y_train.pkl', 'rb') as f:
        label_train = pickle.load(f)
    with open('1D-test/' + feature + '_corr_1d.npy', 'rb') as f:
        X_test = np.load(f)
    with open('1D-test/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    X_train, X_valid, y_train, y_valid = train_test_split(matrix_train, label_train, shuffle=True, test_size=0.25, random_state=10)

    ae1_model_path = format_config("./data/model_basc064/{experiment}_autoencoder-1.ckpt", {
        "experiment": feature,
    })
    ae2_model_path = format_config("./data/model_basc064/{experiment}_autoencoder-2.ckpt", {
        "experiment": feature,
    })
    nn_model_path = format_config("./data/model_basc064/{experiment}_mlp.ckpt", {
        "experiment": feature,
    })

    code_size_1 = int(matrix_train.shape[1] * 0.06)
    code_size_2 = int(code_size_1 * 0.5)

    reset()

    # Run first autoencoder
    run_autoencoder1(feature,
                     X_train, y_train, X_valid, y_valid, X_test, y_test,
                     model_path=ae1_model_path,
                     code_size=code_size_1)

    reset()

    # Run second autoencoder
    run_autoencoder2(feature,
                     X_train, y_train, X_valid, y_valid, X_test, y_test,
                     model_path=ae2_model_path,
                     prev_model_path=ae1_model_path,
                     prev_code_size=code_size_1,
                     code_size=code_size_2)

    reset()

    # Run multilayer NN with pre-trained autoencoders
    pred, logit, acc_list, loss_list, auc_list = run_finetuning(feature,
                                                                X_train, y_train, X_valid, y_valid, X_test, y_test,
                                                                model_path=nn_model_path,
                                                                prev_model_1_path=ae1_model_path,
                                                                prev_model_2_path=ae2_model_path,
                                                                code_size_1=code_size_1,
                                                                code_size_2=code_size_2)

    prediction.append(pred)
    logits = logits + logit

    # Computing metrics scores
    predictions = np.argmax(logits, axis=1)
    preds = pd.DataFrame(prediction)
    y_pred = [mode(preds[i])[0][0] for i in range(len(preds.T))]

    y_true = y_test
    [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    auc = roc_auc_score(y_true, y_pred)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    sensitivity = recall = TP / (TP + FN)
    fscore = 2 * TP / (2 * TP + FP + FN)

    print('acc: ', accuracy)
    print('auc: ', auc)
    print('precision: ', precision)
    print('recall: ', recall)
    print('fscore: ', fscore)


    # Draw plots on training and validation performance
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(acc_list.T[0])
    plt.plot(acc_list.T[1])
    plt.plot(acc_list.T[2])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid', 'test'], loc='upper left')
    plt.subplot(2,1,2)
    plt.plot(loss_list.T[0])
    plt.plot(loss_list.T[1])
    plt.plot(loss_list.T[2])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid', 'test'], loc='upper left')
    plt.tight_layout()
    fig
