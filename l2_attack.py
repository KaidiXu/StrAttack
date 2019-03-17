## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-3     # larger values converge faster to less accurate results e-2 for MNIST -3 for others
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 1           # how strong the adversarial example should be
INITIAL_CONST = 1     # the initial constant c to pick as a first guess

class CarliniL2:
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 boxmin = -0.5, boxmax = 0.5):
        """
        The L_2 optimized attack. 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.image_size = model.image_size
        self.num_channels = model.num_channels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.model = model

        self.repeat = binary_search_steps >= 10

        shape = (batch_size,image_size,image_size,num_channels)
        
        # the variable we're going to optimize over
        self.modifier = tf.Variable(np.zeros(shape,dtype=np.float32))
#        self.modifier2 = tf.Variable(np.zeros(shape, dtype=np.float32))
#        self.modifier3 = tf.Variable(np.zeros(shape, dtype=np.float32))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        
        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.newimg = tf.tanh(self.modifier + self.timg) * self.boxmul + self.boxplus
        
        # prediction BEFORE-SOFTMAX of the model
        #self.output = model.predict(self.newimg)
        #model.x_input = self.newimg
        self.output = model.predict(self.newimg)        
        
        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)),[1,2,3])
        
        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab)*self.output,1)
        other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other-real+self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real-other+self.CONFIDENCE)

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const*loss1)
        self.loss = self.loss1+self.loss2
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[self.modifier])

        end_vars = tf.global_variables()
       # self.modifier1 = end_vars[14]
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))

  #      self.modifier = self.modifier2 + self.modifier3 - self.modifier

        self.init = tf.variables_initializer(var_list=[self.modifier]+new_vars)

    #+[self.modifier2] + [self.modifier3]
    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        rv = []
        print('go up to', len(imgs))
        for i in range(0, len(imgs), self.batch_size):
            print('tick', i)
            r1, r2 = self.attack_batch(imgs[i:i + self.batch_size], targets[i:i + self.batch_size])
            r.extend(r1)
            rv = np.append(rv, r2)
        rv = rv.reshape([-1,3])
        rv = np.mean(rv, axis = 0)

        print("none zeros groups:", rv[0], "\nl2 mean:", rv[1], "\nli mean:", rv[2], "\n")
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # convert to tanh-space
        imgs2 = imgs
        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        
        if self.model.image_size>32: #imagenet
            filterSize = 13
            stride = 13
        else: # cifar mnist
            filterSize = 2
            stride = 2
        n = self.image_size * self.image_size * self.num_channels

        P = np.floor((self.image_size - filterSize) / stride) + 1
        P = P.astype(np.int32)
        Q = P

        index = np.ones([P*Q,filterSize * filterSize * self.num_channels],dtype=int)
        #index2 = np.ones([P*Q,filterSize * filterSize * self.model.num_channels],dtype=int)
        
        tmpidx = 0
        for q in range(Q):
            # plus = 0
            plus1 = q * stride * self.image_size * self.num_channels
            for p in range(P):
                index_ = np.array([], dtype=int)
                #index2_ = np.array([], dtype=int)
                for i in range(filterSize):
                    index_ = np.append(index_,
                                      np.arange(p * stride * self.num_channels + i * self.image_size * self.num_channels + plus1,
                                                p * stride * self.num_channels + i * self.image_size * self.num_channels + plus1 + filterSize * self.num_channels,
                                                dtype=int))
                index[tmpidx] = index_
                tmpidx += 1
        index = np.tile(index, (self.batch_size,1,1))
        
        
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(outer_step, o_bestl2, CONST)
    #        print(self.modifier)
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]
    
            bestl2 = [1e10]*batch_size
            bestscore = [-1]*batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})
            
            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                if iteration % 200 == 0:
                    print(iteration, o_bestl2)
                # perform the attack 
                ttt, l, l2s, scores, nimg = self.sess.run([self.train, self.loss,
                                                         self.l2dist, self.output, 
                                                         self.newimg])
                modi = self.sess.run(self.modifier)
                # print out the losses every 10%
                #if iteration%(self.MAX_ITERATIONS//10) == 0:
                #    print(iteration,self.sess.run((self.loss,self.loss1,self.loss2)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                    if l > prev*.9999:
                        break
                    prev = l

                # adjust the best result found so far
                for e,(l2,sc,ii) in enumerate(zip(l2s,scores,nimg)):
                    if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        #print("change", e, o_bestl2[e] - l2)
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        
        o_besty = np.array(o_bestattack) - imgs2
        rVector = [0, 0, 0]             
        resultl2 = np.array([])  
        resultli = np.array([])  
        for b in (range(batch_size)):
            for k in range(index.shape[1]):
                ry0D = np.take(o_besty[b], index[b,k])
                ry0D2 = np.linalg.norm(ry0D)
                if ry0D2 != 0:
                    resultl2 = np.append(resultl2, ry0D2)
                    resultli = np.append(resultli, np.max(np.abs(ry0D)))
        
        rVector[0] =  len(resultl2)/batch_size
        rVector[1] =  np.mean(resultl2)
        rVector[2] =  np.mean(resultli)
     
        print("\ntotal groups:", P*Q)
        
        return o_bestattack, rVector
    
        
    
    
    

