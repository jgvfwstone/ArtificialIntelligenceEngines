# Code Examples For Deep Learning Engines Book

THIS IS A DRAFT.

21 Feb 2018 	James V Stone

**Things for JVS to do**
- []rename main files main.py
- [] check code runs well
- [] add comments to code
- [] add graph of output during training
- [] display final results as images


The online code implements minimal examples, and is intended to demonstrate the basic principles that underpin each neural network. 

A particularly insightful PyTorch tutorial can be found here:
	https://pytorch.org/tutorials/beginner/nn_tutorial.html
and
	https://github.com/pytorch/examples/

For a really simple example, see 
	https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html

The following is a list of online examples at
	https://github.com/jgvfwstone/DeepLearningEngines

—

**Chapter 2: Linear Associative Networks**
* Github directory: 
* Github files: Ch02_LinearNetwork
* Author: 
* Date created: 
* Original Source: https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html#sphx-glr-beginner-examples-nn-two-layer-net-nn-py
* License: 
* Description: 
* Network: Linear network using gradient descent to learn 1 association Linear network using gradient descent to learn 2 associations Linear network using gradient descent to learn photographs 
For linear network start with two_layer_net_nn.py
in /Users/JimStone/Documents/GitHub/DNNStoneBookCode/Python/PyTorchCode/tutorials

TTD Write code!

—

**Chapter 3: Perceptrons** 
* Github directory: https://github.com/jgvfwstone/DeepLearningEngines/tree/master/DeepLearningEnginesCode/Python/Ch03_Perceptron
* Original Source: https://github.com/Honghe/perceptron/tree/master/perceptron
* Author: Honghe
* Date created: 2013
* License: None, permission requested.
* Email: leopardsaga@gmail.com
* Description: Uses perceptron algorithm to classify linearly seperable classes, and produces graphical output after training is complete.
 
 - []  permission requested.

—

**Chapter 4: The Backpropagation Algorithm**
* Github directory: https://github.com/jgvfwstone/DeepLearningEngines/tree/master/DeepLearningEnginesCode/Python/Ch04_BackpropNetwork
* Author: Sunil Ray, modified by JVS
* Date created: 2016
* License: None (ask permission)
* Original Source: https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
* Description:  A three layer backprop network with 2 input units, 2 hidden units and 1 output unit that learns the exclusive-OR (XOR) problem.

TTD Permission requested from Ray.

—

**Chapter 5: Hopfield Nets**
* Github directory: https://github.com/jgvfwstone/DeepLearningEngines/tree/master/DeepLearningEnginesCode/Python/Ch05_HopfieldNet
* Author: Tom Stafford
* Date created: 2016
* License: MIT
* Original Source: https://github.com/tomstafford/emerge/blob/master/lecture4.ipynb
* Description: Learns binary images. Recalls perfect versions from noisy input images.

TTD

—

**Chapter 7: Restricted Boltzmann Machines**
* Github directory:  https://github.com/jgvfwstone/DeepLearningEngines/tree/master/DeepLearningEnginesCode/Python/Ch07_RestrictedBoltzmannMachine
* Author: Gabriel Bianconi 
* Date created: 2017
* License: MIT License
* Original Source: https://github.com/GabrielBianconi/pytorch-rbm
* Description: Applies a RBM to the MNIST dataset. The trained model uses a SciPy-based logistic regression for classification. It achieves 92.8% classification accuracy.

TTD Ask permission, see other RBMs.
- [] add initial comment
- [] plot lik during training
- [] show final images

—

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

TTD

—

**Chapter 9: Deep Backprop Network**
* Github directory: https://github.com/jgvfwstone/DeepLearningEngines/tree/master/DeepLearningEnginesCode/Python/Ch09_ConvolutionalNetwork
* Author: Various
* Date created: 
* License: https://github.com/pytorch/examples/blob/master/LICENSE
* Original Source: https://github.com/pytorch/examples/blob/master/mnist
* Description: Convolutional backprop network trained to recogise digits 0-9 from the MNIST data set.


- [] Add graphical output.



--

**Chapter 10: Reinforcement Learning**
* Github directory: https://github.com/jgvfwstone/DeepLearningEngines/tree/master/DeepLearningEnginesCode/Python/Ch10_ReinforcementLearning
* Author: Adam Paszke (adam.paszke@gmail.com)
* Date created: 2017.
* License: MIT
* Original Source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
* Description: This shows how to use PyTorch to train an agent on the cart pole task, with graphical output of the cart pole.
* Also see pytorch's own  example here
https://github.com/pytorch/examples/tree/master/reinforcement_learning
TTD OK - get permission.  Copyright 2017, PyTorch. Ask adam.paszke@gmail.com

—
