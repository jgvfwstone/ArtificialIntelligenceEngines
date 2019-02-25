#!/usr/bin/env python
#coding: utf-8
# modified by JVStone, was test.py
# JVS put all code in this file.

# local module
# import generate_data
#import perceptron
import matplotlib.pyplot as plt
import numpy as np

# set random seed so get same sequence of random numbers each time prog is run.
np.random.seed(2)

class Perceptron(object):
    """docstring for Perceptron"""
    def __init__(self):
        super(Perceptron, self).__init__()
        self.w = [np.random.rand() * 2 - 1 for _ in range(2)] # weights
        self.learningRate = 0.1

    def response(self, x):
        """perceptron output"""
        # y = x[0] * self.w[0] + x[1] * self.w[1] # dot product between w and x
        y = sum([i * j for i, j in zip(self.w, x)]) # more pythonic
        if y >= 0:
            return 1
        else:
            return -1

    def updateWeights(self, x, iterError):
        """
        upates the wights status, w at time t+1 is
        w(t+1) = w(t) + learningRate * (d - r) * x
        iterError is (d - r)
        """
        # self.w[0] += self.learningRate * iterError * x[0]
        # self.w[1] += self.learningRate * iterError * x[1]
        self.w = \
            [i + self.learningRate * iterError * j for i, j in zip(self.w, x)]

    def train(self, data):
        """
        trains all the vector in data.
        Every vector in data must three elements.
        the third eclemnt(x[2]) must be the label(desired output)
        """
        learned = False
        iteration = 0
        while not learned:
            globalError = 0.0
            for x in data: # for each sample
                r = self.response(x)
                if x[2] != r: # if have a wrong response
                    iterError = x[2] - r # desired response - actual response
                    self.updateWeights(x, iterError)
                    globalError += abs(iterError)
            iteration += 1
            if globalError == 0.0 or iteration >= 100: # stop criteria
                print('iterations: %s' % iteration)
                learned = True # stop learing

def generateData(n):
    """
    generates a 2D linearly separable dataset with n samples.
    The thired element of the sample is the label
    """
    xb = (np.random.rand(n) * 2 -1) / 2 - 0.5
    yb = (np.random.rand(n) * 2 -1) / 2 + 0.5
    xr = (np.random.rand(n) * 2 -1) / 2 + 0.5
    yr = (np.random.rand(n) * 2 -1) / 2 - 0.5
    inputs = []
    inputs.extend([[xb[i], yb[i], 1] for i in range(n)])
    inputs.extend([[xr[i], yr[i], -1] for i in range(n)])
    return inputs

trainset = generateData(80) # train set generation
testset = generateData(20) # test set generation
p = Perceptron() # use a short
p.train(trainset)

#Perceptron test
for x in testset:
    r = p.response(x)
    if r != x[2]: # if the response is not correct
        print('not hit.')
    if r == 1:
        plt.plot(x[0], x[1], 'ob')
    else:
        plt.plot(x[0], x[1], 'or')

# plot of the separation line. 
# The centor of line is the coordinate origin
# So the length of line is 2
# The separation line is orthogonal to w
n = np.linalg.norm(p.w) # aka the length of p.w vector
ww = p.w / n # a unit vector
ww1 = [ww[1], -ww[0]]
ww2 = [-ww[1], ww[0]]
plt.ion()
plt.plot([ww1[0], ww2[0]], [ww1[1], ww2[1]], '--k')
plt.title('Two classes separated by a line orthogonal to the weight vector')
plt.show()
