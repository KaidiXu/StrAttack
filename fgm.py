## Based on the CleverHans FGM implementation
## Modified by Yash Sharma for epsilon search and to match attack structure for attack code

## fgm.py -- attack a network with the Fast Gradient Method 
##
## Copyright (C) 2017, Yash Sharma <ysharma1126@gmail.com>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np
from six.moves import xrange

class FGM:
    def __init__(self, sess, model, batch_size=9, ord=np.inf, clip_min=-0.5, clip_max=0.5, targeted=True, inception=False):

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.model = model
        self.targeted = targeted
        self.batch_size = batch_size
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.inception = inception

        shape = (batch_size,image_size,image_size,num_channels)

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.eps = tf.Variable(0., dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_eps = tf.placeholder(tf.float32)

        self.tlab_new = self.tlab / tf.reduce_sum(self.tlab, 1, keep_dims=True)
        
        # prediction BEFORE-SOFTMAX of the model
        self.output = self.model.predict(self.timg)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.tlab_new)

        if self.targeted:
            self.loss = -self.loss

        self.gradients, = tf.gradients(self.loss, self.timg)

        if self.ord == np.inf:
            self.signed_grad = tf.sign(self.gradients)
        elif self.ord == 1:
            reduc_ind = list(xrange(1, len(shape)))
            self.signed_grad = self.gradients / tf.reduce_sum(tf.abs(self.gradients),
                                               reduction_indices=reduc_ind,
                                               keep_dims=True)
        elif self.ord == 2:
            reduc_ind = list(xrange(1, len(shape)))
            self.signed_grad = self.gradients / tf.sqrt(tf.reduce_sum(tf.square(self.gradients),
                                                       reduction_indices=reduc_ind,
                                                       keep_dims=True))

        self.adv_x = tf.clip_by_value(tf.stop_gradient(self.timg + self.eps*self.signed_grad), self.clip_min, self.clip_max)

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.eps.assign(self.assign_eps))
        

    def attack(self, inputs, targets):
        adv_ = []

        grad_ = []
        print('go up to',len(inputs))
        for i in range(0,len(inputs),self.batch_size):
            print('tick',i)
            batch = inputs[i:i+self.batch_size]
            batchlab = targets[i:i+self.batch_size]

            self.sess.run(self.setup, {self.assign_timg: batch, self.assign_tlab: batchlab, self.assign_eps: 0.})
            adv,grad = self.sess.run([self.adv_x, self.signed_grad])
            adv_.extend(adv)
            grad_.extend(grad)
        adv_ = np.array(adv_)
        print(adv_.shape)
        grad_ = np.array(grad_)
        print(grad_.shape)

        if self.ord == np.inf:
            step_size = 1e-3
            eps = np.arange(1e-3,1e+0,step_size)
        elif self.ord == 2:
            step_size = 1e-2
            eps = np.arange(1e-2,1e+1,step_size)
        elif self.ord == 1:
            step_size = 1e+0
            eps = np.arange(1e+0,1e+3,step_size)
        loop_iter = np.arange(0,len(inputs))
        for i,c in enumerate(eps):
            adv = np.clip(np.add(inputs,np.multiply(c,grad_)), self.clip_min, self.clip_max)
            for j in loop_iter:
                pred = self.model.model.predict(adv[j:j+1])
                if self.inception:
                    pred = np.reshape(pred, (targets[0:1].shape))
                if(np.argmax(pred,1) == np.argmax(targets[j:j+1],1)):
                    loop_iter = np.setdiff1d(loop_iter, j)
                    print(len(loop_iter))
                    adv_[j] = adv[j]
        adv = adv_
        return adv
