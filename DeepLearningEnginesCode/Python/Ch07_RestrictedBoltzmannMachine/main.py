#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RBM network.
@author: Gabriel Bianconi, modified by JimStone
# This takes about 10 minutes to run on a 2015 mac. 
# Description: Applies a single 2-layer RBM to the MNIST dataset of images of digits from 0-9. The trained model uses a SciPy-based logistic regression to classify outputs. It achieves 92.8% classification accuracy on the test set of images.
Result: 9237/10000
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms

########## CONFIGURATION ##########

BATCH_SIZE = 64 # batch size for training

VISIBLE_UNITS = 784  # 784 = 28 x 28 images

HIDDEN_UNITS = 128

CD_K = 2 # number of contrastive divergence (CD) steps per training vector

EPOCHS = 2 # number of times the entire data set is used to train RBM

DATA_FOLDER = 'data/mnist'

# CUDA = Compute Unified Device Architecture, a GPU processor.
# Don't worry if you don't have this, the code can work without it.
CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

########## DEFINITIONS ##########

class RBM():

    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5,weight_decay=1e-4, use_cuda=True):
        
        self.num_visible    = num_visible
        self.num_hidden     = num_hidden
        self.k              = k # number of contrastive divergence steps
        self.learning_rate  = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay   = weight_decay
        self.use_cuda       = use_cuda

        # weights between input and hidden units
        self.weights        = torch.randn(num_visible, num_hidden) * 0.1
        # bias terms of visible units
        self.visible_bias   = torch.ones(num_visible) * 0.5
        # bias terms of hidden units
        self.hidden_bias    = torch.zeros(num_hidden)

        # create space for momentum parameters
        self.weights_momentum       = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum  = torch.zeros(num_visible)
        self.hidden_bias_momentum   = torch.zeros(num_hidden)

        if self.use_cuda:
            self.weights        = self.weights.cuda()
            self.visible_bias   = self.visible_bias.cuda()
            self.hidden_bias    = self.hidden_bias.cuda()

            self.weights_momentum       = self.weights_momentum.cuda()
            self.visible_bias_momentum  = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum   = self.hidden_bias_momentum.cuda()

    def sample_hidden(self, visible_probabilities):
        # get input to each hidden unit
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        # use hidden unit inputs to find prob(on)=hidden_probabilities
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities):
        # get input to each visible unit
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        # use visible unit inputs to find prob(on)=hidden_probabilities
        visible_probabilities = self._sigmoid(visible_activations)
        return visible_probabilities

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()
        return random_probabilities
    
# ===========================================================
    def contrastive_divergence(self, input_data):
        # input_data is 64 (batch size) by 784 (real valued pixels in input image)
        # =Positive phase==================================================
        # Positive phase = use 'clamped' visible unit states to sample hidden unit states.
        # sample_hidden() treats each real-valued pixel as a probability
        positive_hidden_probabilities   = self.sample_hidden(input_data) # 64 x 128 hidden units
        # use positive_hidden_probabilities to get sample of binary hidden unit states
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float() # BATCH_SIZE = 64 x 128 hidden units
        
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)
        # print((positive_associations.shape)) # torch.Size([784, 128]) HIDDEN_UNITS = 128
        # .t() = transpose: https://pytorch.org/docs/0.3.1/torch.html#torch.t
        # positive_associations measures correlation between visible and hidden unit states when visible units are  clamped to training data.
        
        # =Negative phase==================================================
        # Negative phase, initialise with final binary positive_hidden_activations
        hidden_activations = positive_hidden_activations # 64 x 128

        for step in range(self.k): # number of contrastive divergence steps
            visible_probabilities   = self.sample_visible(hidden_activations)
            hidden_probabilities    = self.sample_hidden(visible_probabilities)
            hidden_activations      = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_visible_probabilities  = visible_probabilities
        negative_hidden_probabilities   = hidden_probabilities
        # negative_associations measures correlation between visible and hidden unit states when visible units are not clamped to training data.
        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update weight change
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        # Update visible bias terms
        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        # Update hidden bias terms
        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)

        self.weights        += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias   += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias    += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities)**2)

        return error
# ===========================================================


########## DEFINITIONS DONE ##########

########## LOAD DATASET ##########
print('Loading MNIST dataset of images of digits between 0 and 9 ...')

# training data
train_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

# test data
test_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)


########## TRAINING RBM ##########
print('Training RBM for %d epochs ...' % EPOCHS)
# create RBM network with one visible and one hidden laayer
rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=CUDA)

for epoch in range(EPOCHS):
    epoch_error = 0.0

    for batch, _ in train_loader:
        batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

        if CUDA:
            batch = batch.cuda()

        batch_error = rbm.contrastive_divergence(batch)

        epoch_error += batch_error

    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))


########## EXTRACT FEATURES ##########
print('Extracting features...')

train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))
train_labels = np.zeros(len(train_dataset))
test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
test_labels = np.zeros(len(test_dataset))

for i, (batch, labels) in enumerate(train_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

    if CUDA:
        batch = batch.cuda()

    train_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    train_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()

for i, (batch, labels) in enumerate(test_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

    if CUDA:
        batch = batch.cuda()

    test_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    test_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()


########## CLASSIFICATION ##########
print('Fitting linear classifier ...')

clf = LogisticRegression(solver='newton-cg',multi_class='auto')
# train_features = 60,000 hidden unit outputs (1 per training vector) x 128 hidden units
# use outputs of trained hidden units to fit linear classifier to correct labels ...
clf.fit(train_features, train_labels)
# use fitted linear classifier to classify test data ... test_features = 10,000 hidden layer states, 1 per test vector
predictions = clf.predict(test_features) # predictions = 10,000 classifications (0-9)

# compare 10,000 classification results on test data with correct class labels ...
print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))
# prints out (for example, on test data) Result: 9244/10000 = 92.44%

########## THE END ##########
