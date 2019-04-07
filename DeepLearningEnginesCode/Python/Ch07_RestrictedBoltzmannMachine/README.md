# Chapter 7: Restricted Boltzmann Machine (RBM) 

* To run code, use main.py
* Author: Gabriel Bianconi 
* Date created: 2017
* License: MIT License, reproduced with permission
* Original Source: https://github.com/GabrielBianconi/pytorch-rbm
* Description: Applies a single RBM to the MNIST dataset of images of digits from 0-9. The trained model uses a SciPy-based logistic regression to classify outputs. It achieves 92.8% classification accuracy on the test set of images.
* Requirements: The following packages are required: sklearn, torch, torchvision. 
* Notes from the author (Gabriel Bianconi):
This project implements Restricted Boltzmann Machines (RBMs) using PyTorch (see `rbm.py`). Our implementation includes momentum, weight decay, L2 regularization, and CD-*k* contrastive divergence. We also provide support for CPU and GPU (CUDA) calculations. In addition, we provide an example file applying our model to the MNIST dataset (see `mnist_dataset.py`). The example trains an RBM, uses the trained model to extract features from the images, and finally uses a SciPy-based logistic regression for classification. It achieves 92.8% classification accuracy (this is obviously not a cutting-edge model).
