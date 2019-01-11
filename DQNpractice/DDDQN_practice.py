#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:12:21 2019

@author: shaydineen
"""

import tensorflow as tf

class DDDQN(object):
    
    def __init__(self, action_size, state_size, lr, name):
        self.action_size = action_size
        self.state_size = state_size
        self.lr = lr
        self.name = name
        
        with tf.variable_scope(self.name):
            
            self.inputs_ = tf.placeholder(dtype = tf.float32, shape = [None, *self.state_size], name = 'inputs')
            self.actions_ = tf.placeholder(dtype = tf.float32, shape = [None, self.action_size], name = 'actions')
            
            self.ISWeights = tf.placeholder(dtype= tf.float32, shape = [None, 1], name = 'ISWeights')
            self.target_Q = tf.placeholder(dtype = tf.float32, shape = [None], name = 'target_Q')
            
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            
            
            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")
            
            
            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            
            
            self.flatten = tf.layers.flatten(self.conv3_out)
            
            
            ## Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="value_fc")
            
            self.value = tf.layers.dense(inputs = self.value_fc,
                                        units = 1,
                                        activation = None,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="value")
            
            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="advantage_fc")
            
            self.advantage = tf.layers.dense(inputs = self.advantage_fc,
                                        units = self.action_size,
                                        activation = None,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="advantages")
            
            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis = 1) 
            
            self.absolute_errors = tf.abs(self.target_Q - self.Q)
            
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target_Q, self.Q))
            
            self.optim = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
