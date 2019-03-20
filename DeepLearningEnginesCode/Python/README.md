# Code For Artificial Intelligence Engines Book

THIS IS A DRAFT.
 	James V Stone

The online code implements minimal examples, and is intended to demonstrate the basic principles that underpin each neural network. 

A particularly insightful PyTorch tutorial can be found here:
	https://pytorch.org/tutorials/beginner/nn_tutorial.html
and
	https://github.com/pytorch/examples/

For a really simple example, see 
	https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html

The following is a list of online examples at
	https://github.com/jgvfwstone/DeepLearningEngines

**Chapter 2: Linear Associative Networks**
* Github directory: https://github.com/jgvfwstone/ArtificialIntellgenceEngines/tree/master/DeepLearningEnginesCode/Python/Ch02_LinearNetwork
* Author: Stuart Wilson, modified by JVStone.
* Date created: 
* Original Source: Personal communication.
* License: MIT.
* Description: 
* Network: Associative linear network learns to map 4 2D input vectors to 4 scalars, using gradient descent.

**Chapter 3: Perceptrons** 
* Github directory: https://github.com/jgvfwstone/DeepLearningEngines/tree/master/DeepLearningEnginesCode/Python/Ch03_Perceptron
* Original Source: https://github.com/Honghe/perceptron/tree/master/perceptron
* Author: Honghe, modified by JVS.
* Date created: 2013
* License: None, copied with permission from author.
* Description: Uses perceptron algorithm to classify two linearly separable classes, and produces graphical output during training.

Note also see: https://github.com/rasbt/stat479-deep-learning-ss19/blob/master/L09_mlp/code/xor-problem.ipynb

**Chapter 4: The Backpropagation Algorithm**
* Github directory: https://github.com/jgvfwstone/DeepLearningEngines/tree/master/DeepLearningEnginesCode/Python/Ch04_BackpropNetwork
* Author: Sunil Ray, modified by JVS.
* Date created: March 2017
* License: None, copied with permission from the author.
* Original Source: https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
* Description:  A three layer backprop network with 2 input units, 2 hidden units and 1 output unit that learns the exclusive-OR (XOR) problem.

**Chapter 5: Hopfield Nets**
* Github directory: https://github.com/jgvfwstone/DeepLearningEngines/tree/master/DeepLearningEnginesCode/Python/Ch05_HopfieldNet
* Author: Tom Stafford
* Date created: 2016
* License: MIT
* Original Source: https://github.com/tomstafford/emerge/blob/master/lecture4.ipynb
* Description: Learns binary images. Recalls perfect versions from noisy input images.

**Chapter 7: Restricted Boltzmann Machines**
* Github directory:  https://github.com/jgvfwstone/DeepLearningEngines/tree/master/DeepLearningEnginesCode/Python/Ch07_RestrictedBoltzmannMachine
* Author: Gabriel Bianconi 
* Date created: 2017
* License: MIT License
* Original Source: https://github.com/GabrielBianconi/pytorch-rbm
* Description: Applies a single RBM to the MNIST dataset of images of digits from 0-9. The trained model uses a SciPy-based logistic regression to classify outputs. It achieves 92.8% classification accuracy on the test set of images.

**Chapter 8: Variational Autoencoders**
* Github directory: https://github.com/jgvfwstone/DeepLearningEngines/tree/master/DeepLearningEnginesCode/Python/Ch08_VariationalAutoencoder
* Author: Kingma, Botha (cpbotha@vxlabs.com), Stone
* Date created: 2018
* License: MIT
* Original Source: https://github.com/dpkingma/examples/tree/master/vae
* Description: Data are MNIST images of digits. This is an improved implementation of the paper (http://arxiv.org/abs/1312.6114) by Kingma and Welling. It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster. JVS added graph of ELBO during training, plus reconstructed images ater training.
This is a combination of two vae main.py files, from
	https://github.com/pytorch/examples/tree/master/vae
and 
	https://github.com/dpkingma/examples/tree/master/vae

**Chapter 9: Deep Backprop Network**
* Github directory: https://github.com/jgvfwstone/DeepLearningEngines/tree/master/DeepLearningEnginesCode/Python/Ch09_ConvolutionalNetwork
* Author: Various
* Date created: 
* License: https://github.com/pytorch/examples/blob/master/LICENSE
* Original Source: https://github.com/pytorch/examples/blob/master/mnist
* Description: Convolutional backprop network trained to recogise digits 0-9 from the MNIST data set.

**Chapter 10: Reinforcement Learning**
* Github directory: https://github.com/jgvfwstone/DeepLearningEngines/tree/master/DeepLearningEnginesCode/Python/Ch10_ReinforcementLearning
* Author: Adam Paszke (adam.paszke@gmail.com)
* Date created: 2017.
* License: MIT
* Original Source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
* Description: This shows how to use PyTorch to train an agent on the cart pole task, with graphical output of the cart pole.
* Also see pytorch's own  example here
https://github.com/pytorch/examples/tree/master/reinforcement_learning

