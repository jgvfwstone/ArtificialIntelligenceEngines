#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:13:01 2019
@author: JimStone
"""
#  Linear associative net, adapted from backprop network code by Suart Wilson.

import numpy as np
import pylab as pl

########## Define class MLP ##########
class MLP():
    
    ########## Define Methods for MLP class ##########
    # Initialize network topology
    def __init__(self,nInput,nOutput):
        
        # set number of units in each layer
        self.nInput = nInput
        self.nOutput = nOutput
        
        # define weights between layers
        self.wO = np.random.rand(self.nOutput,self.nInput+1)
        
        # define vector to store state of output layer
        self.activityO = np.zeros([self.nOutput])
        
        # define vector to store delta terms of output layer
        self.deltaO = np.zeros([self.nOutput])

    # Iterate the network (activation and learning)
    def step(self,inp,tar,alpha):
        # get input vector with bias unit by appending 1 to end of input vector using 
         # hstack=Stack arrays in sequence horizontally (column wise).
        input = np.hstack([inp,1.])
        
        # use input layer state to get output layer state
        for k in range(self.nOutput):
            self.activityO[k] = self.activation(np.dot(self.wO[k,:],input))

        # Backpropagation learning algorithm
        if (alpha>0.):
            input = np.hstack([inp,1.])
            # get delta terms of output layer units
            for k in range(self.nOutput):
                self.deltaO[k] = (self.activityO[k] - tar[k])

            for k in range(self.nOutput):
                    self.wO[k,:] -= alpha * self.deltaO[k] * input

    # Linear neuron activation function
    def activation(self,x):
        y=1.0*x
        return y


########## set parameters ##########

# Initialize network
M = MLP(2,1) # JVS  works with 2 h units, but graphics not set up for that.

# Input space and targets for XOR
input = np.array([[0,0],[0,1],[1,0],[1,1]])

# targets for XOR
target = np.array([[0],[1],[1],[0]])

# targets for simple linearly separable problem
target = np.array([[0],[1],[0],[1]])

# Timesteps
T = 50

# Store dynamic values
WO = np.zeros([T,M.nOutput,M.nInput+1])
CE = np.zeros(T)

########## Run ##########

for t in range(T):

    # Index random input / target pair
    ip = int(np.floor(np.random.rand()*4.))
    
    # Iterate the backprop algorithm
    M.step(input[ip],target[ip],0.5)
    
    # Compute the error
    ce = 0.
    for k in range(target.shape[0]):
        M.step(input[k,:],target[k,:],0.)
#        ce += np.sum(target[k,:]*np.log(M.activityO)+(1.-target[k,:])*np.log(1.-M.activityO))
        a=(target[k,:]-M.activityO)
        ce += np.sum(a*a)

    CE[t] = ce

# Print comparison of target and output
for k in range(target.shape[0]):
    M.step(input[k,:],target[k,:],0.)
    print( 'target: ' + str(target[k,0]) + ', output: ' + ("%.2f" % M.activityO))

########## Plot ##########

F = pl.figure(0,figsize=(4,4))
f = F.add_subplot(111)
f.plot(CE)
f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
f.set_xlabel('Training epoch')
f.set_ylabel('Error')
f.set_title('Error during training')
pl.show()

########## The End ##########
