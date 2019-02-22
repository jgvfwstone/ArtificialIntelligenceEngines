THIS IS A DRAFT.

21 Feb 2018 	James V Stone


Code Examples Listed in Deep Learning Engines Book
===========================================

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

* Network: Linear Associative Network
* Github files: Ch02_LinearNetwork

Author: 
Date created: 
Original Source:: https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html#sphx-glr-beginner-examples-nn-two-layer-net-nn-py
License: 
Description: 
Network: Linear network using gradient descent to learn 1 association Linear network using gradient descent to learn 2 associations Linear network using gradient descent to learn photographs 
For linear network start with two_layer_net_nn.py
in /Users/JimStone/Documents/GitHub/DNNStoneBookCode/Python/PyTorchCode/tutorials

TTD Write code!

—

Network: Perceptron 
Github files: Ch03_Perceptron/test.py
Original Source: https://github.com/Honghe/perceptron/tree/master/perceptron
Author: Honghe??
Date created: 2013
License: None (ask permission)
Email: leopardsaga@gmail.com
Description: Gives nice graph.

TTD Ask permission.

—

Network: Backprop net
Github files: Ch04_BackpropNetwork/xor.py
Author: Sunil Ray with mods by JVS
Date created: 2016
License: None (ask permission)
Original Source: https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
Description:  A three layer backprop network with 2 hidden units that learns XOR.

TTD

—

Network: Hopfield net
Github files: Ch05_HopfieldNet/lecture4.py
Author: Tom Stafford
Date created: 
License: None
Original Source: https://github.com/tomstafford/emerge/blob/master/lecture4.ipynb
Description: Learns binary images. Recalls perfect versions from noisy input images.

TTD

—

Network: Boltzmann machine
Github files: 
Original Source::
Github files: 
Author: 
Date created: 
License: 
Description: 

TTD

—

Network: Restricted Boltzmann Machines
Github files: Ch07_RestrictedBoltzmannMachine/mnist_example.py
Author: Gabriel Bianconi 
Date created: 2017
License: MIT License
Original Source: https://github.com/GabrielBianconi/pytorch-rbm
Description: Applies a RBM to the MNIST dataset. The trained model uses a SciPy-based logistic regression for classification. It achieves 92.8% classification accuracy.

TTD Ask permission, see other RBMs.

—

Network: Variational autoencoder
Github files: Ch08_VariationalAutoencoder/main.py
Author: Kingma, Botha (cpbotha@vxlabs.com), Stone
Date created: 2018
License: MIT
Original Source: https://github.com/dpkingma/examples/tree/master/vae
Description: Data are MNIST images of digits. This is an improved implementation of the paper (http://arxiv.org/abs/1312.6114) by Kingma and Welling. It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster. JVS added graph of ELBO during training, plus reconstructed images ater training.
This is a combination of two vae main.py files, from
	https://github.com/pytorch/examples/tree/master/vae
and 
	https://github.com/dpkingma/examples/tree/master/vae

TTD

—

Network: Reinforcement learning
Github files: Ch10_ReinforcementLearning/reinforcement_q_learning.py
Author: Adam Paszke (adam.paszke@gmail.com)
Date created: 2017.
License: MIT
Original Source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Description: This shows how to use PyTorch to train an agent on the cart pole task, with graphical output of the cart pole.

TTD OK - get permission.  Copyright 2017, PyTorch. Ask adam.paszke@gmail.com

—
