#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perceptron
Original Source: https://github.com/Honghe/perceptron/tree/master/perceptron
Author: Honghe, modified by JVS.
"""

import matplotlib.pyplot as plt
import numpy as np

# set random seed so get same sequence of random numbers each time prog is run.
# np.random.seed(20)

class Perceptron(object):
    """docstring for Perceptron"""
    def __init__(self):
        super(Perceptron, self).__init__()
        self.w = [np.random.rand() * 2 - 1 for _ in range(2)] # weights
        self.learningRate = 0.001

    def response(self, x):
        # perceptron output
        y = x[0] * self.w[0] + x[1] * self.w[1] # dot product between w and x
        # y = sum([i * j for i, j in zip(self.w, x)]) # more pythonic
        #y = np.dot(x[0:1],self.w)
        y=y*1.0
        if y >= 0:
            res = 1.0
        else:
            res = -1.0        
        return res
    
    def updateWeights(self, x, iterError):
        """
        upates the wights status, w at time t+1 is
        w(t+1) = w(t) + learningRate * (d - r) * x
        iterError is (d - r)
        """
        self.w[0] += self.learningRate * iterError * x[0]
        self.w[1] += self.learningRate * iterError * x[1]
        #self.w = \
            #[i + self.learningRate * iterError * j for i, j in zip(self.w, x)]

    def train(self, data):
        """
        Trains all the vector in data.
        Every vector in data must three elements.
        The third element of x (ie x[2]) is the label(desired output)
        """
        learned = False
        iteration = 0
        while not learned:
            numcorrect = 0
            globalError = 0.0
            for x in data: # for each input vector
                r = self.response(x)
                if x[2] != r: # if have a wrong response
                    iterError = x[2] - r # desired response - actual response
                    self.updateWeights(x, iterError)
                    globalError += abs(iterError)
                else:
                    numcorrect += 1
            print('num correctly classified = %s' % numcorrect)
            iteration += 1
            if globalError == 0.0 or iteration >= 100: # stop criteria
                print('iterations = %s' % iteration)
                learned = True # stop learning
                
            ########## Plot ##########
            plotinterval = 2
            if (iteration % plotinterval == 0):
                plotData(data,self.w)
                plt.pause(0.001)

def generateData(n):
    """
    generates a 2D linearly separable dataset with 2n samples.
    The third element of the sample is the correct response.
    """
    # class A=blue dots has all y values above 0
    xb = (np.random.rand(n) * 2 -1)  
    yb = (np.random.rand(n) * 2 -1) / 2 + 0.55
    
    # class B=red dots has all y values below 0
    xr = (np.random.rand(n) * 2 -1) 
    yr = (np.random.rand(n) * 2 -1) / 2 - 0.55
    
    inputs = []
    inputs.extend([[xb[i], yb[i], 1] for i in range(n)])
    inputs.extend([[xr[i], yr[i], -1] for i in range(n)])
    return inputs

def plotData(data,w):
    plt.clf()
    for x in data:
        if x[1] < 0:
            plt.plot(x[0], x[1], 'ob')
        else:
            plt.plot(x[0], x[1], 'or')
    # plot the decision boundary. 
    # The decision boundary is orthogonal to w.
    n = np.linalg.norm(w) # aka the length of p.w vector
    ww = w / n # a unit vector
    ww1 = [ww[1], -ww[0]]
    ww2 = [-ww[1], ww[0]]
    plt.ion()
    plt.plot([ww1[0], ww2[0]], [ww1[1], ww2[1]], '--k')
    plt.title('Two classes separated by a line orthogonal to the weight vector')
    plt.show()
    
trainset = generateData(80) # train set generation
# testset = generateData(20) # test set generation

p = Perceptron() 

plotData(trainset,p.w)
p.train(trainset)
plotData(trainset,p.w)

########## The End ##########
