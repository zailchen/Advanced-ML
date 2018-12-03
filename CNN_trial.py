# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
from problem import get_train_data
from utils import to_softmax, reset
from sklearn.metrics import roc_auc_score


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
        
        self.layer1_w = tf.get_variable("layer1_weights", shape=[1, self.embed_size, self.num_channels, self.num_filters],
           initializer = self.initializer)
        self.layer1_b = tf.Variable(tf.constant(0.001, shape=[self.num_filters]))
        
        self.layer2_w = tf.get_variable("layer2_weights", shape=[self.embed_size, 1, self.num_filters, 2*self.num_filters],
           initializer = self.initializer)
        self.layer2_b = tf.Variable(tf.constant(0.001, shape=[2*self.num_filters]))
        
        self.layer3_w = tf.get_variable("layer3_weights", shape=[2*self.num_filters, self.num_hidden],
           initializer = self.initializer)
        self.layer3_b = tf.Variable(tf.constant(0.01, shape=[self.num_hidden]))
        
        self.layer4_w = tf.get_variable("layer4_weights", shape=[self.num_hidden, self.num_labels],
           initializer = self.initializer)
        self.layer4_b = tf.Variable(tf.constant(0.01, shape=[self.num_labels]))
        
        self.computation_graph()
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.train_y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss) 
        self.eval()
        

    def computation_graph(self):

        # 1st layer: line-by-line convolution with ReLU and dropout
        conv = tf.nn.conv2d(self.train_X, self.layer1_w, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.dropout(tf.nn.relu(conv + self.layer1_b), self.keep_prob)
        
        # 2nd layer: convolution by column with ReLU and dropout
        conv = tf.nn.conv2d(hidden, self.layer2_w, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.dropout(tf.nn.relu(conv + self.layer2_b), self.keep_prob)
        
        # 3rd layer: fully connected hidden layer with dropout and ReLU
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [-1, shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, self.layer3_w) + self.layer3_b), self.keep_prob)
        
        # 4th (output) layer: fully connected layer with logits as output
        self.logits = tf.matmul(hidden, self.layer4_w) + self.layer4_b
        self.pred = tf.nn.softmax(self.logits)
        
    def eval(self):
        # Compute accuracies
        correct_pred = tf.equal(tf.argmax(self.pred, 1),tf.argmax(self.train_y, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, "float"))

    def run_train_step(self, sess, batch_X, batch_y, keep_prob):
        pred, loss, opt, acc = sess.run([self.pred, self.loss, self.optimizer, self.acc], 
            feed_dict = {self.train_X: batch_X, 
                         self.train_y: batch_y, 
                         self.keep_prob: keep_prob})
        return pred, loss, opt, acc
    
    def run_test_step(self, sess, batch_X, batch_y):
        pred, acc = sess.run([self.pred, self.acc], 
                             feed_dict = {self.train_X: batch_X, 
                                          self.train_y: batch_y,
                                          self.keep_prob: 1.0})
        return pred, acc


class CCNNSolver(object):
    def __init__(self, numROI, X_train, y_train, X_valid, y_valid, X_test, y_test, model_dir):

        self.image_size = numROI
        self.num_labels = 2
        self.num_channels = 1
        self.num_hidden = 96
        self.num_filters = 64
        self.learning_rate = 0.001
        self.batch_size = 20
        self.epoch_num = 50
        self.keep_prob = 0.6
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        
        self.model_dir = model_dir
        
        self.run_train()

    def run_train(self):
        
        # Convert output to one-hot encoding
        self.y_train = np.array([to_softmax(self.num_labels, y) for y in self.y_train])
        self.y_valid = np.array([to_softmax(self.num_labels, y) for y in self.y_valid])
        self.y_test = np.array([to_softmax(self.num_labels, y) for y in self.y_test])
        
        # Convert feature matrix into channel form
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], self.num_channels)) 
        self.X_valid = np.reshape(self.X_valid, (self.X_valid.shape[0], self.X_valid.shape[1], self.X_valid.shape[2], self.num_channels)) 
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2], self.num_channels)) 

    
        with tf.Session() as self.sess:
            
            self.model = CCNN(self.image_size, self.num_labels, self.num_channels,
                               self.num_hidden, self.num_filters, self.learning_rate)
            self.saver = tf.train.Saver(tf.global_variables())

            self.sess.run(tf.global_variables_initializer())
        
            # Start training
            print("Start training ==========")
            best_acc = 0.0
            
            for epoch in range(self.epoch_num):
            
                # randomly shuffle data
                index = np.arange(self.X_train.shape[0])
                random.shuffle(index)
            
                self.X_train = self.X_train[index,:,:]
                self.y_train = self.y_train[index]
                
            
                # Generate train batches
                batches = range(len(self.X_train) // self.batch_size)
                accs = np.zeros((len(batches), 3))
                AUCs = np.zeros((len(batches), 3))

                for ib in batches:
                    
                    # Compute start and end of batch from training set data array
                    from_i = ib * self.batch_size
                    to_i = (ib + 1) * self.batch_size

                    # Select current batch
                    batch_X, batch_y = self.X_train[from_i:to_i], self.y_train[from_i:to_i]

                    # exclude case that are all 1 or all 0
                    if 0 in np.sum(batch_y, axis=0):
                        continue
                
                    # Run optimization and retrieve training accuracy and AUC score
                    train_pred, train_loss, _, train_acc = self.model.run_train_step(self.sess, batch_X, batch_y, self.keep_prob)
                    train_AUC = roc_auc_score(np.argmax(batch_y, 1), train_pred[:,1])
                    
                    # Compute validation accuracy and AUC
                    valid_pred, valid_acc = self.model.run_test_step(self.sess, self.X_valid, self.y_valid)
                    valid_AUC = roc_auc_score(np.argmax(self.y_valid, 1), valid_pred[:,1])
                    
                    # Compute test accuracy and AUC
                    test_pred, test_acc = self.model.run_test_step(self.sess, self.X_test, self.y_test)
                    test_AUC = roc_auc_score(np.argmax(self.y_test, 1), test_pred[:,1])

                    
                    accs[ib] = [train_acc, valid_acc, test_acc]
                    AUCs[ib] = [train_AUC, valid_AUC, test_AUC]
            

            # Compute the average accuracy from all batches
            accs = accs.mean(axis=0)
            train_acc, valid_acc, test_acc = accs
            
            # Compute the average AUC for all batches
            AUCs = AUCs.mean(axis=0)
            train_AUC, valid_AUC, test_AUC = AUCs
            
            print("Epoch: %d, batch: %d/%d, train_loss: %.4f, \
                  train_acc: %.4f, valid_acc: %.4f, test_acc: %.4f,  \
                  train_AUC: %.4f, valid_AUC: %.4f, test_AUC: %.4f" 
                    % (epoch, ib, batches, train_loss, \
                       train_acc, valid_acc, test_acc, \
                       train_AUC, valid_AUC, test_AUC))

            # Save model if validation accuracy is larger than previous one
            if valid_acc > best_acc:
                best_acc = valid_acc
                checkpoint_path = self.model_dir + '_model_' + '.ckpt'
                self.saver.save(self.sess, checkpoint_path)
                print("model has been saved to %s" % (checkpoint_path))


def split(matrix, labels_train, train_ratio = 0.8, valid_ratio = 0.9):
    n = matrix.shape[0]
    index = np.arange(n)
    random.shuffle(index)
    
    train_id = index[:int(n*train_ratio)]
    valid_id = index[int(n*train_ratio):int(n*valid_ratio)]
    test_id = index[int(n*valid_ratio):]
    
    X_train = matrix[train_id,:,:]
    y_train = labels_train[train_id]
    
    X_valid = matrix[valid_id,:,:]
    y_valid = labels_train[valid_id]
    
    X_test = matrix[test_id,:,:]
    y_test = labels_train[test_id]
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


if __name__ == "__main__":
    
    train_ratio = 0.8
    valid_ratio = 0.9
    model_dir = './model/CNN1'
    
    data_train, labels_train = get_train_data()
    
    fmri_feature = ['basc064','basc122','basc197','crad','harvard','msdl','power']
    
    feature_path = 'fmri_corr_features/2D/' + fmri_feature[3] + '_corr_2d.npy'
    with open(feature_path, 'rb') as f:
        matrix2d = np.load(f)

    X_train, y_train, X_valid, y_valid, X_test, y_test = split(matrix2d, labels_train, train_ratio, valid_ratio)
    
    reset()
    
    CCNNSolver(X_train.shape[1], X_train, y_train, X_valid, y_valid, X_test, y_test, model_dir)
