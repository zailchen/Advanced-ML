# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import pickle
import os
from sklearn.metrics import roc_auc_score, f1_score


class CCNN(object):

    def __init__(self, image_size, num_labels, num_channels, num_hidden,
            num_filters, learning_rate, keep_prob):
                
        # Set hyper-parameters
        self.image_size = image_size
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.embed_size = image_size
        self.num_hidden = num_hidden
        self.num_filters = num_filters
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        
        # Input placeholders
        self.train_X = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size, self.num_channels))
        self.train_y = tf.placeholder(tf.float32, shape=(None, self.num_labels))
        self.is_training = tf.placeholder(tf.bool)

        self.computation_graph()
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.train_y))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss) 
        self.eval()

        # summaries for loss and acc
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.acc)
        self.summary_op = tf.summary.merge_all()
        

    def computation_graph(self):

        conv1 = tf.layers.conv2d(self.train_X, filters = self.num_filters, kernel_size = [5,5], 
            strides = (1, 1), padding = 'valid', data_format = 'channels_last', activation = tf.nn.relu)
        dropout1 = tf.layers.dropout(conv1, rate = self.keep_prob, training = self.is_training)

        #conv2 = tf.layers.conv2d(dropout1, filters = self.num_filters, kernel_size = [self.embed_size,1], 
        #    strides = (1, 1), padding = 'valid', data_format = 'channels_last', activation = tf.nn.relu)
        #dropout2 = tf.layers.dropout(conv2,rate = self.keep_prob, training = self.is_training)
        
        flatten = tf.layers.flatten(dropout1)
        #hidden = tf.layers.dense(flatten, self.num_hidden)
        dropout3 = tf.layers.dropout(flatten, rate = self.keep_prob, training = self.is_training)
        
        self.logits = tf.layers.dense(dropout3, self.num_labels)
        self.pred = tf.nn.softmax(self.logits)

        
    def eval(self):
        # Compute accuracies
        correct_pred = tf.equal(tf.argmax(self.pred, 1),tf.argmax(self.train_y, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, "float"))

    def run_train_step(self, sess, batch_X, batch_y):
        pred, loss, opt, acc, summary = sess.run([self.pred, self.loss, self.optimizer, self.acc, self.summary_op], 
            feed_dict = {self.train_X: batch_X, 
                         self.train_y: batch_y,
                         self.is_training: True})
        return pred, loss, opt, acc, summary
    
    def run_test_step(self, sess, batch_X, batch_y):
        pred, acc = sess.run([self.pred, self.acc], 
                             feed_dict = {self.train_X: batch_X, 
                                          self.train_y: batch_y,
                                          self.is_training: False})
        return pred, acc


class CVSolver(object):
    def __init__(self, numROI):

        self.image_size = numROI
        self.num_labels = 2
        self.num_channels = 1
        self.num_hidden = 64
        self.num_filters = 16
        self.learning_rate = 0.0001
        self.batch_size = 20
        self.epoch_num = 50
        self.keep_prob = 0.6


    def run(self, X_train, y_train, X_valid, y_valid, k):
        
        # Convert output to one-hot encoding
        def to_softmax(n_classes, y):
            one_hot_y = [0.0] * n_classes
            one_hot_y[int(y)] = 1.0
            return one_hot_y

        y_train = np.array([to_softmax(self.num_labels, y) for y in y_train])
        y_valid = np.array([to_softmax(self.num_labels, y) for y in y_valid])
        
        # Convert feature matrix into channel form
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], self.num_channels)) 
        X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], self.num_channels)) 

        tf.reset_default_graph()
        with tf.Session() as sess:
            
            model = CCNN(self.image_size, self.num_labels, self.num_channels,
                        self.num_hidden, self.num_filters, self.learning_rate, self.keep_prob)
            #summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
            #self.saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())
        
            # Start training
            print("Start training ==========")
            best_acc = 0.0
            best_AUC = 0.0
            best_f1 = 0.0
            last_epoch = 0
            global_step = 0

            for epoch in range(self.epoch_num):

                # randomly shuffle data
                index = np.arange(X_train.shape[0])
                random.shuffle(index)
            
                X_train = X_train[index,:,:]
                y_train = y_train[index]
                
                # Generate train batches
                batches = len(X_train) // self.batch_size

                for ib in range(batches):
                    global_step += 1

                    # Compute start and end of batch from training set data array
                    from_i = ib * self.batch_size
                    to_i = (ib + 1) * self.batch_size

                    # Select current batch
                    batch_X, batch_y = X_train[from_i:to_i], y_train[from_i:to_i]

                    # exclude cases that are all 1 or all 0
                    if 0 in np.sum(batch_y, axis=0):
                        continue
                
                    # Run optimization and retrieve training accuracy, AUC, F1
                    train_pred, train_loss, _, train_acc, _ = model.run_train_step(sess, batch_X, batch_y)
                    #self.summary_writer.add_summary(train_summary, global_step)
                    train_AUC = roc_auc_score(np.argmax(batch_y, 1), train_pred[:,1])
                    train_f1 = f1_score(np.argmax(batch_y, 1), np.argmax(train_pred, 1))
                    
                
                # Compute validation accuracy, AUC, F1
                valid_pred, valid_acc = model.run_test_step(sess, X_valid, y_valid)
                valid_AUC = roc_auc_score(np.argmax(y_valid, 1), valid_pred[:,1])
                valid_f1 = f1_score(np.argmax(y_valid, 1), np.argmax(valid_pred, 1))

                
                print("Fold: %d, Epoch: %d, train_loss: %.4f" % (k, epoch, train_loss))
                print("train_acc: %.4f, train_AUC: %.4f, train_f1: %.4f" % (train_acc, train_AUC, train_f1))
                print("valid_acc: %.4f, valid_AUC: %.4f, valid_f1: %.4f" % (valid_acc, valid_AUC, valid_f1))


                # Save model if validation accuracy is larger than previous one
                if sum([valid_acc > best_acc, valid_AUC > best_AUC, valid_f1 > best_f1]) >= 2:
                    print('saving better =====================')
                    best_acc = valid_acc
                    best_AUC = valid_AUC
                    best_f1 = valid_f1
                    last_epoch = epoch
                    #checkpoint_path = self.model_dir + '_model' + '.ckpt'
                    #self.saver.save(self.sess, checkpoint_path)
                    #print("model has been saved to %s" % (checkpoint_path))

        return best_acc, best_AUC, best_f1, last_epoch


class TestSolver(CVSolver):
    def __init__(self, numROI, epoch_num, model_dir, summary_dir):
        super(TestSolver, self).__init__(numROI)

        self.epoch_num = 100
        self.model_dir = model_dir
        self.summary_dir = summary_dir
        if not os.path.exists(self.summary_dir):
            os.mkdir(self.summary_dir)

    def run(self, X_train, y_train, X_test, y_test):
        
        # Convert output to one-hot encoding
        def to_softmax(n_classes, y):
            one_hot_y = [0.0] * n_classes
            one_hot_y[int(y)] = 1.0
            return one_hot_y

        y_train = np.array([to_softmax(self.num_labels, y) for y in y_train])
        y_test = np.array([to_softmax(self.num_labels, y) for y in y_test])
        
        # Convert feature matrix into channel form
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], self.num_channels)) 
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], self.num_channels)) 

        tf.reset_default_graph()
        with tf.Session() as sess:
            
            model = CCNN(self.image_size, self.num_labels, self.num_channels,
                        self.num_hidden, self.num_filters, self.learning_rate, self.keep_prob)
            summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)
            saver = tf.train.Saver(tf.global_variables())

            sess.run(tf.global_variables_initializer())
        
            # Start training
            global_step = 0

            for epoch in range(self.epoch_num):
            
                # randomly shuffle data
                index = np.arange(X_train.shape[0])
                random.shuffle(index)
            
                X_train = X_train[index,:,:]
                y_train = y_train[index]
                
                # Generate train batches
                batches = len(X_train) // self.batch_size

                for ib in range(batches):
                    global_step += 1

                    # Compute start and end of batch from training set data array
                    from_i = ib * self.batch_size
                    to_i = (ib + 1) * self.batch_size

                    # Select current batch
                    batch_X, batch_y = X_train[from_i:to_i], y_train[from_i:to_i]

                    # exclude case that are all 1 or all 0
                    if 0 in np.sum(batch_y, axis=0):
                        continue
                
                    # Run optimization and retrieve training accuracy and AUC score
                    train_pred, train_loss, _, train_acc, train_summary = model.run_train_step(sess, batch_X, batch_y)
                    summary_writer.add_summary(train_summary, global_step)
                    train_AUC = roc_auc_score(np.argmax(batch_y, 1), train_pred[:,1])
                    train_f1 = f1_score(np.argmax(batch_y, 1), np.argmax(train_pred, 1))

                # Compute test accuracy and AUC
                test_pred, test_acc = model.run_test_step(sess, X_test, y_test)
                test_AUC = roc_auc_score(np.argmax(y_test, 1), test_pred[:,1])
                test_f1 = f1_score(np.argmax(y_test, 1), np.argmax(test_pred, 1))
     
                print("Epoch: %d, train_loss: %.4f" % (epoch, train_loss))
                print("train_acc: %.4f, train_AUC: %.4f, train_f1: %.4f" % (train_acc, train_AUC, train_f1))
                print("test_acc: %.4f, test_AUC: %.4f, test_f1: %.4f" % (test_acc, test_AUC, test_f1))

            # Save model
            checkpoint_path = self.model_dir + '_model' + '.ckpt'
            saver.save(sess, checkpoint_path)
            print("model has been saved to %s" % (checkpoint_path))

        return test_acc, test_AUC, test_f1


def split(matrix, labels, ratio = 0.8, seed=None):
    n = matrix.shape[0]
    index = np.arange(n)
    if seed:
        random.seed(seed)
    random.shuffle(index)
    
    train_id = index[:int(n*ratio)]
    valid_id = index[int(n*ratio):]
    
    X_train = matrix[train_id,:,:]
    y_train = labels[train_id]
    
    X_valid = matrix[valid_id,:,:]
    y_valid = labels[valid_id]
    
    return X_train, y_train, X_valid, y_valid


if __name__ == "__main__":
    
    K = 5
    valid_ratio = 0.8
    model_dir = './model/CNN-1'
    summary_dir = './summary/CNN-1'
    
    fmri_feature = ['basc064','basc122','basc197','craddock_scorr_mean','harvard_oxford_cort_prob_2mm','msdl','power_2011']
    
    with open('data_final/train/2D/' + fmri_feature[3] + '_corr_2d.npy', 'rb') as f:
        matrix_train = np.load(f)
    with open('data_final/y_train.pkl', 'rb') as f:
        label_train = pickle.load(f)
    with open('data_final/test/2D/' + fmri_feature[3] + '_corr_2d.npy', 'rb') as f:
        X_test = np.load(f)
    with open('data_final/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    
    acc = np.zeros(K)
    AUC = np.zeros(K)
    f1 = np.zeros(K)
    epoch = np.zeros(K)
    
    for k in range(K):
        X_train, y_train, X_valid, y_valid = split(matrix_train, label_train, valid_ratio)
        cv_solver = CVSolver(X_train.shape[1])
        acc[k], AUC[k], f1[k], epoch[k] = cv_solver.run(X_train, y_train, X_valid, y_valid, k)

    print('CV results =====================')
    print('acc:',acc)
    print('AUC:',AUC)
    print('F1:',f1)
    print('epoch:',epoch)
    print('valid_acc:',np.mean(acc),'valid_AUC:',np.mean(AUC),'valid_f1:',np.mean(f1),'mean_epoch:',np.mean(epoch))

    '''
    epoch = 81
    print('Testing ========================')
    test_solver = TestSolver(matrix_train.shape[1], int(np.mean(epoch)), model_dir, summary_dir)
    test_acc, test_AUC, test_f1 = test_solver.run(matrix_train, label_train, X_test, y_test)
    print('test_acc:',test_acc,'test_AUC:',test_AUC,'test_f1:',test_f1)
    '''