# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:38:32 2020

@author: SIM
"""
from __future__ import print_function
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import time
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from common.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '../dataset/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

seconds = int(time.time()) // 60
minute = seconds % 60
hour = (seconds // 60 % 24) + 9
start = "Started from" + str(hour) + ":" + str(minute) + ":" + str(seconds)

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)
    
#    if epochs == 3:
#        training = mean_loss
        
    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}".format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}".format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct

# Design Your model for DNN
# Validation set의 accuracy 70% 이상 되도록 deep convolution network을 설계하시오
# Alexnet, Vggnet, resnet 등 다양한 모델을 사용해도 가능함

def my_model(X,is_training):
    #vggnet활용
    c0_0 = tf.layers.conv2d(X, 64, [3, 3], strides=[1,1], padding='SAME', activation=tf.nn.relu)
    b0_0 = tf.layers.batch_normalization(c0_0, epsilon=1e-3, training=is_training)
    c0_1= tf.layers.conv2d(b0_0, 64, [3, 3], strides=[1,1], padding='SAME', activation=tf.nn.relu)
    b0_1 = tf.layers.batch_normalization(c0_1, epsilon=1e-3, training=is_training)
    m0 = tf.layers.max_pooling2d(b0_1, pool_size=[2,2], strides=2, padding='SAME')
    d0 = tf.nn.dropout(m0, keep_prob=0.75)

    
    c1_0 = tf.layers.conv2d(d0, 128, [3, 3], strides=[1,1], padding='SAME', activation=tf.nn.relu)
    b1_0 = tf.layers.batch_normalization(c1_0, epsilon=1e-3, training=is_training)
    c1_1= tf.layers.conv2d(b1_0, 128, [3, 3], strides=[1,1], padding='SAME', activation=tf.nn.relu)
    b1_1 = tf.layers.batch_normalization(c1_1, epsilon=1e-3, training=is_training)
    m1 = tf.layers.max_pooling2d(b1_1, pool_size=[2,2], strides=2, padding='SAME')
    d1 = tf.nn.dropout(m1, keep_prob=0.75)
    
    c2_0 = tf.layers.conv2d(d1, 256, [3, 3], strides=[1,1], padding='SAME', activation=tf.nn.relu)
    b2_0 = tf.layers.batch_normalization(c2_0, epsilon=1e-3, training=is_training)
    c2_1= tf.layers.conv2d(b2_0, 256, [3, 3], strides=[1,1], padding='SAME', activation=tf.nn.relu)
    b2_1 = tf.layers.batch_normalization(c2_1, epsilon=1e-3, training=is_training)
    m2 = tf.layers.max_pooling2d(b2_1, pool_size=[2,2], strides=2, padding='SAME')
    d2 = tf.nn.dropout(m2, keep_prob=0.75)
    
    c3_0 = tf.layers.conv2d(d2, 512, [3, 3], strides=[1,1], padding='SAME', activation=tf.nn.relu)
    b3_0 = tf.layers.batch_normalization(c3_0, epsilon=1e-3, training=is_training)
    c3_1= tf.layers.conv2d(b3_0, 512, [3, 3], strides=[1,1], padding='SAME', activation=tf.nn.relu)
    b3_1 = tf.layers.batch_normalization(c3_1, epsilon=1e-3, training=is_training)
    c3_2= tf.layers.conv2d(b3_1, 512, [3, 3], strides=[1,1], padding='SAME', activation=tf.nn.relu)
    b3_2 = tf.layers.batch_normalization(c3_2, epsilon=1e-3, training=is_training)
    m3 = tf.layers.max_pooling2d(b3_2, pool_size=[2,2], strides=2, padding='SAME')
    d3 = tf.nn.dropout(m3, keep_prob=0.75)
    
    c4_0 = tf.layers.conv2d(d3, 512, [3, 3], strides=[1,1], padding='SAME', activation=tf.nn.relu)
    b4_0 = tf.layers.batch_normalization(c4_0, epsilon=1e-3, training=is_training)
    c4_1= tf.layers.conv2d(b4_0, 512, [3, 3], strides=[1,1], padding='SAME', activation=tf.nn.relu)
    b4_1 = tf.layers.batch_normalization(c4_1, epsilon=1e-3, training=is_training)
    c4_2= tf.layers.conv2d(b4_1, 512, [3, 3], strides=[1,1], padding='SAME', activation=tf.nn.relu)
    b4_2 = tf.layers.batch_normalization(c4_2, epsilon=1e-3, training=is_training)
    m4 = tf.layers.max_pooling2d(b4_2, pool_size=[2,2], strides=2, padding='SAME')
    d4 = tf.nn.dropout(m4, keep_prob=0.75)
 
    m4_flat = tf.contrib.layers.flatten(d4) 
    
    fc0 = tf.layers.dense(inputs=m4_flat, units=10, activation=tf.nn.relu)
    b0 = tf.layers.batch_normalization(fc0, epsilon=1e-3, training=is_training)
    fc1 = tf.layers.dense(inputs=b0, units=10, activation=tf.nn.relu)
    b1 = tf.layers.batch_normalization(fc1, epsilon=1e-3, training=is_training)
    y_out = tf.layers.dense(inputs=b1, units=10)

    return y_out


# clear old variables
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
z = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
# prediction
y_out = my_model(X, is_training)


# loss function and optimizer
# Layers, Activations, Loss functions : https://www.tensorflow.org/api_guides/python/nn
# You can use another loss function
mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,10), logits=y_out))

# Optimizers: https://www.tensorflow.org/api_guides/python/train#Optimizers
# You can use another optimizer
#optimizer = tf.train.AdamOptimizer(learning_rate=5e-4)
optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# batch normalization in tensorflow requires this extra dependency
# You should decide whether you use Batch Normal or not
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True   

# Train the model on one epoch
with tf.Session(config=config) as sess:
    with tf.device('/gpu:0'): #"/cpu:0" or "/gpu:0"
        sess.run(tf.global_variables_initializer())
        print('\nTraining')
        run_model(sess,y_out,mean_loss,X_train,y_train,300,64,100,train_step,False)
        print('\nValidation')
        run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
        print('\nTest')
        run_model(sess,y_out,mean_loss,X_test,y_test,1,64)
