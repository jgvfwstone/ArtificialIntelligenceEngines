#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:45:14 2018
backprop for the XOR problem.
19th March 2019 JVS modified code and added graphics.
"""

import numpy as np
import matplotlib.pyplot as plt

# set random seed so get same sequence of random numbers each time prog is run.
np.random.seed(1)

########## set input and output values ##########

# set input vectors for XOR
#Input array
X = np.array([[0,0], [0,1], [1,0] , [1,1]])

# set output values for XOR
targetvectors = np.array([[0],[1],[1],[0]])

# define unit activcation function as sigmoid function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

# define derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

########## set parameters ##########
niter = 3000 # number of training iterations
plotinterval = 100 # interval between plotting graphs

errors = np.zeros(niter) # record of errors during training
numcorrects = np.zeros(niter) # number of correct outputs during training

# decide if want to have sigmoidal or linear output units
sigmoidalOutputUnits = 0

if (sigmoidalOutputUnits):
    lr = 0.5 # learning rate
else:
    lr = 0.1 # learning rate

inputlayer_neurons = X.shape[1] # number of units in input layer 
hiddenlayer_neurons = 2 # number of hidden layer units
output_neurons = 1 # number of units in output layer

# weight and bias initialization
# weights between input and hidden layer
wh = np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
# biases of hidden layer units
bh = np.random.uniform(size=(1,hiddenlayer_neurons))

# weights of output layer units
wout = np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
# biases of output layer units
bout = np.random.uniform(size=(1,output_neurons))

########## SET UP GRAPHS ##########
# set interactive plotting on, so graphs appear outside of console.
plt.ion()

fig, axes = plt.subplots(1,2)
axerror = axes[0]
axnumcorrect = axes[1]

axerror.set_xlabel('Epoch')
axerror.set_ylabel('Error')
        
axnumcorrect.set_ylabel('Number correct')
axnumcorrect.set_xlabel('Epoch')

# set state of bias unit; this code works if set to -1 or +1.
biasunitstate = -1.0

########## LEARN ##########
for iter in range(niter):

    # Forward Propogation
    hidden_layer_input1 = np.dot(X,wh) # input from input layer
    hidden_layer_input = hidden_layer_input1 + bh*biasunitstate # add input from bias unit
    hiddenlayer_states = sigmoid(hidden_layer_input)
    
    output_layer_input1 = np.dot(hiddenlayer_states,wout)
    output_layer_input= output_layer_input1 + bout * biasunitstate
    
    # Backpropagation
    # get derivatives of errors wrt unit inputs ...
    # ... of output layer
    if (sigmoidalOutputUnits):
        output = sigmoid(output_layer_input)
        slope_output_layer = derivatives_sigmoid(output)
    else: # output units are linear
        output = output_layer_input
        slope_output_layer = output*0 + 1 # each derivative = 1

    d = targetvectors - output # delta terms = errors in output layer
        
    # get derivatives of errors wrt unit inputs of hidden units
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_states)
    
    # get delta terms of output units = d_output
    d_output = d * slope_output_layer
    
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    
    # update weights of output units
    wout += hiddenlayer_states.T.dot(d_output) * lr
    # update biases of output units
    bout += np.sum(d_output, axis=0,keepdims=True) * biasunitstate * lr
    
    # update weights and biases of hidden units
    wh += X.T.dot(d_hiddenlayer) * lr    
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) * biasunitstate * lr

    error = np.linalg.norm(d)
    errors[iter] = error
    
    # count number of correct responses
    a = (output<0.5)
    b = (targetvectors<0.5)
    numcorrect = sum(a==b)
    numcorrects[iter] = numcorrect
    
    ########## Plot ##########
    if (iter % plotinterval == 0):
        axerror.plot(errors[0:niter],'k')
        plt.show()
        plt.pause(0.001)
        axnumcorrect.plot(numcorrects[0:niter],'k')
        plt.show()
        plt.pause(0.001)

########## Print results ##########
print('Target values')
print(targetvectors)
print('Output values')
print(output)
error=np.linalg.norm(d)
    #print(i)
print('Final error:')
print(error)
########## The End ##########
