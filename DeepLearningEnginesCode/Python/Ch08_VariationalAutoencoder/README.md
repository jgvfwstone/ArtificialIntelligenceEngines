# Basic VAE Example

This is an improved implementation of the paper [Stochastic Gradient VB and the
Variational Auto-Encoder](http://arxiv.org/abs/1312.6114) by Kingma and Welling.
It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster. Changes to Kingma's code by Xingdong Zuo.

JVS added graph of ELBO during training, plus reconstructed images ater training.

This is a combination of two vae main.py files, from
	https://github.com/pytorch/examples/tree/master/vae
and 
	https://github.com/dpkingma/examples/tree/master/vae

```bash
pip install -r requirements.txt
python main.py
```
