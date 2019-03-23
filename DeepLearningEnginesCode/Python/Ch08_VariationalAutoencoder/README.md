# Variational Autoencoder

* To run, use main.py
* Author: Kingma, Botha (cpbotha@vxlabs.com), with modifcations by JVStone
* Date created: 2018
* License: MIT, reproduced with permission from the author
* Original Source: https://github.com/dpkingma/examples/tree/master/vae
* Description: Trained on MNIST images of digits. 
* Notes from the author: This is an improved implementation of the paper (http://arxiv.org/abs/1312.6114) by Kingma and Welling. It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster. 
* Notes from JVStone: JVS added graph of ELBO during training, plus reconstructed images after training.
This is a combination of two vae main.py files, from
	https://github.com/pytorch/examples/tree/master/vae
and 
	https://github.com/dpkingma/examples/tree/master/vae
