#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:45:14 2018

backprop for XOR.

JVS added graphics.
"""

import numpy as np
import matplotlib.pyplot as plt

# set random seed so get same sequence of random numbers each time prog is run.
np.random.seed(1)

#Input array
#X=np.array([[1,0,1,0], [1,0,1,1], [0,1,0,1]])
X=np.array([[0,0], [0,1], [1,0] , [1,1]])

#Output
y=np.array([[0],[1],[1],[0]])

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
epoch=5000 #Setting training iterations
plotinterval=500

errors = np.zeros(epoch)
numcorrects = np.zeros(epoch)

lr=0.5 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 2 #number of hidden layer neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

plt.ion()

figerror = plt.figure(1);
#figerror.clf()
axerror = figerror.add_subplot(111)
axerror.set_xlabel('Epoch')
axerror.set_ylabel('Error')
        
fignumcorrect = plt.figure(2);
 #fignumcorrect.clf()
axnumcorrect = fignumcorrect.add_subplot(111)
axnumcorrect.set_ylabel('Number correct')
axnumcorrect.set_xlabel('Epoch')

for i in range(epoch):

    #Forward Propogation
    hidden_layer_input1=np.dot(X,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)
    
    #Backpropagation
    d = y-output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = d * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

    error=np.linalg.norm(d)
    errors[i]=error
    
    # count number of correct responses
    a=(output<0.5)
    b=(y<0.5)
    numcorrect = sum(a==b)
    numcorrects[i]=numcorrect
    
    if (i % plotinterval ==0):
        axerror.plot(errors[0:i],'k')
        plt.show()
        plt.pause(0.001)
        
        axnumcorrect.plot(numcorrects[0:i],'k')
        plt.show()
        plt.pause(0.001)

print('Target values')
print(y)
print('Output values')
print(output)
error=np.linalg.norm(d)
    #print(i)
print('Final error:')
print(error)
    #print('i = %s ' %i)
    #print('Error = %s' % EE)
plt.show()



 
