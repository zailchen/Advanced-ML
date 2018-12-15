# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import pickle
import os
from sklearn.metrics import roc_auc_score, f1_score


class CCNN(object):

    def __init__(self, image_size, num_labels, num_channels, num_hidden,
            num_filters, learning_rate):
                
        # Set hyper-parameters
        self.image_size = image_size
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.embed_size = image_size
        self.num_hidden = num_hidden
        self.num_filters = num_filters
        self.learning_rate = learning_rate
        
        # Input placeholders
        self.train_X = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size, self.num_channels))
        self.train_y = tf.placeholder(tf.float32, shape=(None, self.num_labels))
        self.keep_prob = tf.placeholder(dtype=tf.float32)

        # network weight variables: Xavier initialization for better convergence in deep layers
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        self.w_conv1 = tf.get_variable('w_conv1',shape=[1, self.embed_size, self.num_channels, self.num_filters],
           initializer = self.initializer)
        self.b_conv1 = tf.Variable(tf.constant(0.001, shape=[self.num_filters]))
        
        self.w_conv2 = tf.get_variable('w_conv2',shape=[self.embed_size, 1, self.num_filters, self.num_filters],
           initializer = self.initializer)
        self.b_conv2 = tf.Variable(tf.constant(0.001, shape=[self.num_filters]))
        
        self.w_fc1 = tf.get_variable('w_fc1',shape=[self.num_filters, self.num_hidden],
           initializer = self.initializer)
        self.b_fc1 = tf.Variable(tf.constant(0.01, shape=[self.num_hidden]))
        
        self.w_fc2 = tf.get_variable('w_fc2', shape=[self.num_hidden, self.num_labels],
           initializer = self.initializer)
        self.b_fc2 = tf.Variable(tf.constant(0.01, shape=[self.num_labels]))
        
        self.computation_graph()
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.train_y))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss) 
        self.eval()

        # summaries for loss and acc
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.acc)
        self.summary_op = tf.summary.merge_all()
        

    def computation_graph(self):

        # 1st layer: line-by-line convolution with ReLU and dropout
        conv = tf.nn.conv2d(self.train_X, self.w_conv1, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.dropout(tf.nn.relu(conv + self.b_conv1), self.keep_prob)
        
        # 2nd layer: convolution by column with ReLU and dropout
        conv = tf.nn.conv2d(hidden, self.w_conv2, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.dropout(tf.nn.relu(conv + self.b_conv2), self.keep_prob)
        
        # 3rd layer: fully connected hidden layer with dropout and ReLU
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [-1, shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, self.w_fc1) + self.b_fc1), self.keep_prob)
        
        # 4th (output) layer: fully connected layer with logits as output
        self.logits = tf.matmul(hidden, self.w_fc2) + self.b_fc2
        self.pred = tf.nn.softmax(self.logits)
        
    def eval(self):
        # Compute accuracies
        correct_pred = tf.equal(tf.argmax(self.pred, 1),tf.argmax(self.train_y, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, "float"))

    def run_train_step(self, sess, batch_X, batch_y, keep_prob):
        pred, loss, opt, acc, summary = sess.run([self.pred, self.loss, self.optimizer, self.acc, self.summary_op], 
            feed_dict = {self.train_X: batch_X, 
                         self.train_y: batch_y, 
                         self.keep_prob: keep_prob})
        return pred, loss, opt, acc, summary
    
    def run_test_step(self, sess, batch_X, batch_y):
        pred, acc = sess.run([self.pred, self.acc], 
                             feed_dict = {self.train_X: batch_X, 
                                          self.train_y: batch_y,
                                          self.keep_prob: 1.0})
        return pred, acc


class CCNNSolver(object):
    def __init__(self, numROI, X_train, y_train, X_valid, y_valid, X_test, y_test, model_dir, summary_dir):

        self.image_size = numROI
        self.num_labels = 2
        self.num_channels = 1
        self.num_hidden = 32
        self.num_filters = 16
        self.learning_rate = 0.0001
        self.batch_size = 20
        self.epoch_num = 100
        self.keep_prob = 0.8
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        
        self.model_dir = model_dir
        self.summary_dir = summary_dir
        if not os.path.exists(self.summary_dir):
            os.mkdir(self.summary_dir)
        
        self.run_train()

    def run_train(self):
        
        # Convert output to one-hot encoding
        def to_softmax(n_classes, y):
            one_hot_y = [0.0] * n_classes
            one_hot_y[int(y)] = 1.0
            return one_hot_y

        self.y_train = np.array([to_softmax(self.num_labels, y) for y in self.y_train])
        self.y_valid = np.array([to_softmax(self.num_labels, y) for y in self.y_valid])
        self.y_test = np.array([to_softmax(self.num_labels, y) for y in self.y_test])
        
        # Convert feature matrix into channel form
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], self.num_channels)) 
        self.X_valid = np.reshape(self.X_valid, (self.X_valid.shape[0], self.X_valid.shape[1], self.X_valid.shape[2], self.num_channels)) 
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2], self.num_channels)) 

        tf.reset_default_graph()
        with tf.Session() as self.sess:
            
            self.model = CCNN(self.image_size, self.num_labels, self.num_channels,
                               self.num_hidden, self.num_filters, self.learning_rate)
            self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
            self.saver = tf.train.Saver(tf.global_variables())

            self.sess.run(tf.global_variables_initializer())
        
            # Start training
            print("Start training ==========")
            best_acc = 0.0
            last_AUC = 0.0
            last_f1 = 0.0
            last_epoch = 0
            global_step = 0

            for epoch in range(self.epoch_num):
            
                # randomly shuffle data
                index = np.arange(self.X_train.shape[0])
                random.shuffle(index)
            
                self.X_train = self.X_train[index,:,:]
                self.y_train = self.y_train[index]
                
            
                # Generate train batches
                batches = len(self.X_train) // self.batch_size

                for ib in range(batches):
                    global_step += 1

                    # Compute start and end of batch from training set data array
                    from_i = ib * self.batch_size
                    to_i = (ib + 1) * self.batch_size

                    # Select current batch
                    batch_X, batch_y = self.X_train[from_i:to_i], self.y_train[from_i:to_i]

                    # exclude case that are all 1 or all 0
                    if 0 in np.sum(batch_y, axis=0):
                        continue
                
                    # Run optimization and retrieve training accuracy and AUC score
                    train_pred, train_loss, _, train_acc, train_summary = self.model.run_train_step(self.sess, batch_X, batch_y, self.keep_prob)
                    self.summary_writer.add_summary(train_summary, global_step)
                    train_AUC = roc_auc_score(np.argmax(batch_y, 1), train_pred[:,1])
                    train_f1 = f1_score(np.argmax(batch_y, 1), np.argmax(train_pred, 1))
                    
                
                # Compute validation accuracy and AUC
                valid_pred, valid_acc = self.model.run_test_step(self.sess, self.X_valid, self.y_valid)
                valid_AUC = roc_auc_score(np.argmax(self.y_valid, 1), valid_pred[:,1])
                valid_f1 = f1_score(np.argmax(self.y_valid, 1), np.argmax(valid_pred, 1))
                    
                # Compute test accuracy and AUC
                test_pred, test_acc = self.model.run_test_step(self.sess, self.X_test, self.y_test)
                test_AUC = roc_auc_score(np.argmax(self.y_test, 1), test_pred[:,1])
                test_f1 = f1_score(np.argmax(self.y_test, 1), np.argmax(test_pred, 1))

                
                print("Epoch: %d, train_loss: %.4f" % (epoch, train_loss))
                print("train_acc: %.4f, valid_acc: %.4f, test_acc: %.4f" % (train_acc, valid_acc, test_acc))
                print("train_AUC: %.4f, valid_AUC: %.4f, test_AUC: %.4f" % (train_AUC, valid_AUC, test_AUC))
                print("train_f1: %.4f, valid_f1: %.4f, test_f1: %.4f" % (train_f1, valid_f1, test_f1))


                # Save model if validation accuracy is larger than previous one
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    last_acc = test_acc
                    last_AUC = test_AUC
                    last_f1 = test_f1
                    last_epoch = epoch
                    checkpoint_path = self.model_dir + '_model' + '.ckpt'
                    self.saver.save(self.sess, checkpoint_path)
                    print("model has been saved to %s" % (checkpoint_path))
        
        print('Results ==============================')
        print(last_acc, last_AUC, last_f1, last_epoch)


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
    
    K = 10
    valid_ratio = 0.8
    model_dir = './model/CNN-2'
    summary_dir = './summary/CNN-2'
    
    fmri_feature = ['basc064','basc122','basc197','craddock_scorr_mean','harvard_oxford_cort_prob_2mm','msdl','power_2011']
    
    with open('data_final/train/2D/' + fmri_feature[3] + '_corr_2d.npy', 'rb') as f:
        matrix_train = np.load(f)
    with open('data_final/y_train.pkl', 'rb') as f:
        label_train = pickle.load(f)
    with open('data_final/test/2D/' + fmri_feature[3] + '_corr_2d.npy', 'rb') as f:
        X_test = np.load(f)
    with open('data_final/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    X_train, y_train, X_valid, y_valid = split(matrix_train, label_train, valid_ratio)
    CCNNSolver(X_train.shape[1], X_train, y_train, X_valid, y_valid, X_test, y_test, model_dir, summary_dir)
