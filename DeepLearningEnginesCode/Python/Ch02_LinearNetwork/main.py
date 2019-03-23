#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear associative net, adapted from backprop network code by Suart Wilson.
* Author: Stuart Wilson, modified by JVStone.
* Date created: 2016.
* Original Source: Personal communication.
* License: MIT.
* Description: 
* Network: Linear associative  network learns to map 4 2D input vectors to 4 scalars, using gradient descent.
"""

import numpy as np
# use pylab for ploting.
import pylab as pl

########## Define class MLP ##########
class LinearNetwork():
    
    ########## Define Methods for LinearNetwork class ##########
    # Initialize network topology
    def __init__(self,nInput,nOutput):
        
        # set number of units in each layer
        self.nInput = nInput
        self.nOutput = nOutput
        
        # define weights between input and output layers
        # use of +1 input units defines bias unit in input layer
        # example in book does not have bias term, so this is an extra in code.
        self.W = np.random.rand(self.nOutput,self.nInput+1)
        
        # space to store change in weights
        self.dw = np.zeros([self.nOutput,self.nInput+1])
        
        # define vector to store state of output layer
        self.state = np.zeros([self.nOutput])
        
        # define vector to store delta terms of output layer
        self.deltaO = np.zeros([self.nOutput])

    # Iterate the network by one step 
    def step(self,inp,tar,alpha):
        # get input vector with bias unit by appending 1 to end of input vector using 
        # example in book does not have bias term, so this is an extra in code.
        input = np.append([inp], [1.0])
        
        # use input layer state to get output layer state
        for k in range(self.nOutput):
            # get total input u to output unit
            u = np.dot(self.W[k,:],input)
            # use u to find state of output unit
            self.state[k] = self.activation(u)

        # Learning algorithm
        if (alpha>0.):
            # get delta terms of output layer units
            for k in range(self.nOutput):
                self.deltaO[k] = (self.state[k] - tar[k])

            for k in range(self.nOutput):
                self.dw[k,:] -= alpha * self.deltaO[k] * input

    # Define linear unit activation function
    def activation(self,x):
        return x


########## set parameters ##########
# set random seed so get same sequence of random numbers each time prog is run.
np.random.seed(1)
nip = 2 # num input units
nop = 1 # num output units
# Initialize network
M = LinearNetwork(nip,nop) 

# Input vectors 
inputvectors = np.array([[0,0],[0,1],[1,0],[1,1]])

# targets for XOR, not used here, but left here for comparison
# target = np.array([[0],[1],[1],[0]])

# targets for simple linearly separable problem
targets = np.array([[0],[1],[0],[1]])

# Timesteps = number of learning trials
numiter = 20

# num training vectors
T = targets.shape[0]

# set learning rate alpha
alpha = 0.2

# Store cost function values E
E = np.zeros(numiter)

########## Run learning ##########
for iter in range(numiter):

    # reset weight changes to zero
    M.dw = M.dw*0
    
    Et = 0.
    
    for t in range(T):
        
        # find weight change for one association for one step
        inputvector = inputvectors[t]
        target = targets[t]
        
        # get network output and delta term at output 
        M.step(inputvector,target,alpha)
        
        # Compute the error
        dif =(target - M.state)
        Et += np.sum(dif*dif)
    
    E[iter] = Et
    
    # update weights
    for k in range(M.nOutput):
        M.W[k,:] += M.dw[k,:]

# Print comparison of target and output
for k in range(T):
    inputvector = inputvectors[k,:]
    target = targets[t]
    M.step(inputvector,target,0.)
    print('input vector:' + str(inputvector))
    print( 'target: ' + str(target) + ', output: ' + ("%.2f" % M.state))

########## Plot ##########
F = pl.figure(0,figsize=(4,4))
f = F.add_subplot(111)
f.plot(E)
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xlabel('Training epoch')
f.set_ylabel('Error')
f.set_title('Error during training')
pl.show()

########## The End ##########
