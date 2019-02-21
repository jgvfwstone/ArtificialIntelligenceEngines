2/2/2018

==

Overall, make a folder on my mac with ack to each source, and uploade to Github, or just put on my web site as zip file.

https://github.com/PacktPublishing/Deep-Learning-with-PyTorch

See github online https://github.com/jgvfwstone/KingmaPyTorchExamples

Copy python book to tell how to set up anaconda.

# PLOTTING

 plt.imshow(lum_img, cmap="hot")

https://matplotlib.org/index.html

plotIncrementalJVS.py

Notes for Jim:

spyder set prefs/console/graphics to automatic then restart spyder.

To see weights:
 plt.imshow(model.fc1.weight.detach())

https://github.com/PacktPublishing/Deep-Learning-with-PyTorch

These two versions of python have different packages.

My IDLE is in Stone/Applications/Python3.7/IDLE
DO:  cd /Applications/Python\ 3.7

But Spyder is:
/Users/JimStone/anaconda3/python.app

Spyder seems more reliable.
Test all examples on spyder.

To see conda list
# packages in environment at /Users/JimStone/anaconda3:
#
so I have scikit-learn for sklearn

or 
$ pip show scikit-learn
Name: scikit-learn
Version: 0.20.1
Summary: A set of python modules for machine learning and data mining
Home-page: http://scikit-learn.org
Author: None
Author-email: None
License: new BSD
Location: /Users/JimStone/anaconda3/lib/python3.7/site-packages

==

Get PyTorch examples working first, do simple ones later.

Do Fork then Clone(download) to my mac.

ClaudeShannon1916

This file: /Users/JimStone/Documents/GitHub/DeepLearningEnginesCode/Python/README.txt

Online Code Examples Listed in Deep Learning Book (code below from various sources)
=========================================
The online code implements minimal examples, and is intended to demonstrate the basic principles that underpin each neural network. Code examples originally implemented in Python 2.7 have been updated to Python 3.

A particularly insightful PyTorch tutorial can be found here:
https://pytorch.org/tutorials/beginner/nn_tutorial.html

Creating a simple NN in Py Torch

PYTORCH EXAMPLES
https://github.com/pytorch/examples/

For a really simple eg, see https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html

—
TODO
Network: Linear Associative NetworkGithub files: Ch02_LinearNetwork
Author: 
Date created: 
Original Source:: https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html#sphx-glr-beginner-examples-nn-two-layer-net-nn-py
License: 
Description: 
Network: Linear network using gradient descent to learn 1 association Linear network using gradient descent to learn 2 associations Linear network using gradient descent to learn photographs 
For linear network start with two_layer_net_nn.py
in /Users/JimStone/Documents/GitHub/DNNStoneBookCode/Python/PyTorchCode/tutorials

TTD

—

Network: Perceptron 
Github files: Ch03_Perceptron/test.py
Original Source: https://github.com/Honghe/perceptron/tree/master/perceptronAuthor: Honghe
Date created: 2013
License: None (ask permission)
Email: leopardsaga@gmail.com
Description: Gives nice graph.

TTD Ask permission.

—

Network: Backprop net
Github files: Ch04_BackpropNetwork/backpropSuartWilson_JVS1.py + xor.py runs well in spyder.
Author: Wilson
Date created: 2016
License: None (ask permission)
Original Source:: 
Description:  A three layer backprop network learns XOR.
backpropSuartWilson_JVS1 = M = MLP(2,3,1) # JVS  works with 2 h units, but graphics not set up for that.

sklearn: plot_mnist_filters.py works under spyder but not IDLE
https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py

Network:  backprop      CreatingasimpleNNinPyTorch.py This is a backprop
Local Github files: Ch04_BackpropNetwork
Author: 
Date created: 2018
License: 
Original Source:: https://github.com/J-Yash/Creating-a-simple-NN-in-pytorch
Description: A neural network with one hidden layer and a single output unit.
A model that looks like input -> linear -> relu -> linear -> sigmoid. 

Full here: https://medium.com/coinmonks/create-a-neural-network-in-pytorch-and-make-your-life-simpler-ec5367895199

TTD

—

Network: Hopfield net
Github files: Ch05_HopfieldNet/lecture4.py
Author: 
Date created: 
License: None (ask permission)
Original Source:: https://github.com/tomstafford/emerge/blob/master/lecture4.ipynb
GitHub/DNNStoneBookCode/Python/ContributedExamples/emergeTomStafford
Description: Gives nice graph.

TTD

—

Network: Boltzmann machine
Original Source::Github files: 
Author: 
Date created: 
License: 
Description: 

matlab RBM here
https://github.com/kyunghyuncho/deepmat

TTD

—

Network: Restricted Boltzmann Machines
Author: Gabriel Bianconi
Github files: Ch07_RestrictedBoltzmannMachine
Author: 
Date created: 2017
License:   MIT License
Original Source::  https://github.com/GabrielBianconi/pytorch-rbm
Description: This project implements Restricted Boltzmann Machines (RBMs) using PyTorch (see `rbm.py`). Our implementation includes momentum, weight decay, L2 regularization, and CD-*k* contrastive divergence. We also provide support for CPU and GPU (CUDA) calculations.
In addition, we provide an example file applying our model to the MNIST dataset (see `mnist_dataset.py`). The example trains an RBM, uses the trained model to extract features from the images, and finally uses a SciPy-based logistic regression for classification. It achieves 92.8% classification accuracy (this is obviously not a cutting-edge model).

TTD

—

OK
Network: Variational autoencoder
Github files: Ch08_VariationalAutoencoder	README.txt
Author: Kingma
Date created: 
License: 
Original Source: https://github.com/dpkingma/examples/tree/master/vae
Description: This is  a forked version of Pytorch examples in Kingma’s github
Kingma’s readme file says: This is an improved implementation of the paper Stochastic Gradient VB and the Variational Auto-Encoder by Kingma and Welling. It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster. Results as images are stored in files.

Nice commented code from https://github.com/cpbotha
which is in vaecpbotha.py runs ok on spyder.
Need to add my plotting code.

TTD

—

Network: Convolutional neural networkGithub files: Ch09_ConvolutionalNetwork
Author: 
Date created: 
License: 
Original Source: https://github.com/pytorch/examples/tree/master/mnist
Description: 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

Test set: Average loss: 0.0505, Accuracy: 9841/10000 (98%)
Also See packt book for eg.

TTD

DenseNet
https://github.com/liuzhuang13/DenseNet

INCLUDE GAN?
GAN
https://github.com/pytorch/examples/tree/master/dcgan

—

Network: Reinforcement learningGithub files: Ch10_ReinforcementLearning/reinforcement_q_learning.py
Author: Adam Paszke adam.paszke@gmail.com
Date created: 
Original Source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
License: 
Description: 
 Reinforcement learning using TD(0) and TD(λ) Reinforcement learning using SARSA to balance a pole 

Includes graphical output of pole balancing. This tutorial shows how to use PyTorch to train a Deep Q Learning (DQN) agent on the CartPole-v0 task from the `OpenAI Gym.

TTD OK - get permission.  Copyright 2017, PyTorch. Ask adam.paszke@gmail.com

—

Notes delete later:
==============

vae.py is self contained and does its own gradients.

Variational autoencoder learning algorithm
https://github.com/cmudeeplearning11785/Spring2018-tutorials/tree/master/recitation-9

Graphics are in this notebook: THis shows how to display results as images:
https://github.com/cmudeeplearning11785/Spring2018-tutorials/blob/master/recitation-9/variational-autoencoders.ipynb

With nice graphics
variational autoencoder code
\url{https://github.com/altosaar/variational-autoencoder}

with lovely mpegs here:
\url{https://jaan.io/what-is-variational-autoencoder-vae-tutorial/}

Use VAE pytorch - this uses TF NBG
https://github.com/wiseodd/generative-models/blob/master/VAE/vanilla_vae/vae_pytorch.py
Nice pytorch code here: https://github.com/wiseodd/generative-models
+ good tutorial here 
https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/

—

MatLab VAE here
https://github.com/peiyunh/mat-vae

—

==

Examples Included in Pytorch Github
============================

Most examples listed here are from the PyTorch web site:
	https://github.com/pytorch/pytorch

MNIST Convnets

Word level Language Modeling using LSTM RNNs

Training Imagenet Classifiers with Residual Networks

Generative Adversarial Networks (DCGAN)

Variational Auto-Encoders

Superresolution using an efficient sub-pixel convolutional neural network

Hogwild training of shared ConvNets across multiple processes on MNIST

Training a CartPole to balance in OpenAI Gym with actor-critic

Natural Language Inference (SNLI) with GloVe vectors, LSTMs, and torchtext

Time sequence prediction - create an LSTM to learn Sine waves

Additionally, a list of good examples are hosted in their own repositories:
[Neural Machine Translation using sequence-to-sequence RNN with attention (OpenNMT)](https://github.com/OpenNMT/OpenNMT-py)

Contributed Python Code
===================

This refers to Python code that has been collected from various online resources.
Linear network using gradient descent to learn 1 association Linear network using gradient descent to learn 2 associations Linear network using gradient descent to learn photographs PerceptronSeveral nns here:
https://github.com/eriklindernoren/ML-From-Scratch
Hopfield net
Python:https://github.com/srowhani/hopfield-networkBoltzmann machine
https://github.com/tomstafford/emerge/blob/master/lecture4.ipynbTom Stafford lecture4.py needs ipython which is part of Spyder not IDLE
AND
https://github.com/yosukekatada/Hopfield_network
https://github.com/tailaijin/Hopfield-Network-by-Python
https://gist.github.com/anonymous/01df7e791c1b2cc46baf
Restricted Boltzmann machinehttps://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern


Convolutional Neural Networks in Pytorch
https://github.com/cmudeeplearning11785/Spring2018-tutorials/blob/master/recitation-4/Tutorial-pytorch-cnn.ipynb

GAN
https://github.com/cmudeeplearning11785/Spring2018-tutorials/tree/master/recitation-10
Reinforcement learning using TD(0) and TD(λ) Reinforcement learning using SARSA to balance a pole 

Contributed MatLab Code
====================

This refers to MatLab code that has been collected from various online resources.

V nice matlab code here: dustinstansbury 5 yrs old 

https://github. com/dustinstansbury/medal/tree/master/models
mlnn.m Multi-layer neural network 

mlcnn.m Multi-layer convolutional neural network 

rbm.m Restricted Boltzmann machine (RBM) 

mcrbm.m Mean-covariance (3-way Factored) RBM 

drbm.m Dynamic/conditional RBM
dbn.m Deep belief network
crbm.m Convolutional RBM
ae.m Shallow autoencoder
dae.m Deep autoencoder

Installing Software
==============

All instructions below are given to a unix shell in MacOS or Linux (no instructions here for PCs FIX)

I found this web site useful for setting PATH (And some text below is borrowed from that site):
https://github.com/landlab/landlab/wiki/Correcting-Install-Paths

To find out which shell you're running, open a terminal window and type echo $SHELL from the prompt. Anything other than /bin/bash being returned will require a little more manual set up as described below.

JVStoneMacBookAir:~ JimStone$ mv .bash_profile .bash_profile_OLD

JVStoneMacBookAir:~ JimStone$ rm -r ./.spyder*

Installation
========

Anaconda is the recommended package manager since it (should) install most dependencies. 

* Go to https://pytorch.org/

* Install anaconda

* Anaconda should have a good version of python, but to install a specific version of python do (for example):
$ conda install python=3.7.2

* Check python version:
$ python
Python 3.7.2 (default, Dec 29 2018, 00:00:04) 
>>> quit()

* Check that the python being used is in Anaconda:
$ which python
/anaconda3/bin/python

Removing Anaconda
================

See https://stackoverflow.com/questions/42182706/how-to-uninstall-anaconda-completely-from-macos
conda install anaconda-clean

anaconda-clean --yes

Once the configs are removed you can delete the anaconda install folder, which is usually under your home dir:

rm -rf ~/anaconda3

Installing Modules
==============

Install modules pytorch and torchvision:
$ anaconda/bin/conda install pytorch torchvision -c pytorch
The -c tells conda to look under pytorch online

Install gym into anaconda package directly??????
$ cd PATH  /anaconda3/bin
Use  pip from within anaconda’s own directory to install packages in correct place:
$ ./pip install gym 

$ pip show gym
Name: gym
Version: 0.10.11
Summary: The OpenAI Gym: A toolkit for developing and comparing your reinforcement learning agents.
Home-page: https://github.com/openai/gym
Author: OpenAI
Author-email: gym@openai.com
License: UNKNOWN
Location: /Users/JimStone/anaconda3/lib/python3.7/site-packages
Requires: scipy, numpy, six, requests, pyglet
Required-by: 
(base) va185098:anaconda3


environment location: /Users/JimStone/anaconda3

./pip install tensorboardX

THEN

$ bin/pip install torch

$ conda install pytorch torchvision -c pytorch

For installing other modules, need to set Python path:

PATH="/Library/Frameworks/Python.framework/Versions/3.7/bin:${PATH}"
export PATH

Python Environments
=================

https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c

Copying online code to home computer using git 
=====================================

$ cd # chage directory to home directory

$ mkdir pytorch_code # make a directory called pytorch_code

$ cd pytorch_code # go to directory pytorch_code

$ git clone https://github.com/pytorch/examples.git  # copy online directory called examples to pytorch_code

$ git clone https://github.com/openai/gym # copy online directory called gym to pytorch_code
cd gym
pip3 install -e .

git clone https://github.com/Honghe/perceptron

Saving Notebook as Python  Code
===========================

Sometimes an online example is presented as a Jupyter Notebook or Jupyter Lab.
You can save the Python code to run within a Python application (e.g. Spyder).

From within Anaconda’s Jupyter Notebook, find the menu item;
	Download as …
or from Jupyter Lab, find the menu item
	Export Notebook as …
and download to file *.py.

Running Code
===========

* Make directory in which to put example code from pytorch.
$ cd 
$ mkdir pytorch_code

* Start anaconda

* Select Jupyter pylab 

***Running code from Python IDLE

Pytorch has two classes of code: tutorials and examples.

Each tutorial comes with both python code and jupyter notebooks.
Each example has code only.
Python code can be downloaded and run from any Python application like IDLE.

***Running Notebooks

* Use jupyter to run the reinforcement tutorial, which has a graphical output (a cart balancing a pole).
Download notebook from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

***Running Python Code Alone

Anaconda has a standard interface for running Python called Spyder (or you can use IDLE).
To run code from within Python application like Spyder or IDLE:

* Get pytorch examples
$ cd pytorch_code
(The repository should have been cloned into the directory located in whichever directory you ran the git clone command from.)
$ git clone https://github.com/pytorch/examples.git
This creates a directory called pytorch_code/examples, which contains:

Installing Python
—————————————

To find out where the default version of python is stored:

$ which python3 # text after a hash is ignored
/Library/Frameworks/Python.framework/Versions/3.7/bin/python3

Python version:  python3.7.2

Installation: Basic python interface (IDLE), 

Either:

Warned not to use pip, or to use conda’s pip
$ pip install python 

or download from:

https://www.python.org/

==========================
==========================
==========================
==========================

OLD NOTES

NOTES

NO - get python3 via pip3 and install torch and torchvision
-> python3.7.2
Then clone pytorch examples.

cd  /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages

make dir
mkdir pytorch_code
cd pytorch_code
$git clone https://github.com/pytorch/examples.git
copies this to
pytorch_code/examples
git clone https://github.com/openai/gym
cd gym
pip3 install -e .

\section*{PyTorch}
\index{PyTorch}%
% https://www.youtube.com/watch?v=UWlFM0R_x6I
% https://colab.research.google.com/drive/1b522fQGUdFI8Wl840D_XsevYYdbIAlrl
% https://github.com/bayesgroup/deepbayes-2018

 %cd /anaconda3/lib/python3.7/site-packages
 
 https://github.com/landlab/landlab/wiki/Correcting-Install-Paths
 
 PATH="/Library/Frameworks/Python.framework/Versions/3.7/bin:${PATH}"
export PATH

PATH="/Applications/Anaconda2019B/anaconda3/bin:${PATH}"
export PATH

PATH="/anaconda3/bin:${PATH}"
export PATH

mnistExample worked after I changed the PATH and then did:

$ pip3 install sklearn
Collecting sklearn
Collecting scikit-learn (from sklearn)
  Downloading https://files.pythonhosted.org/packages/11/0f/e2279fee7f9834c63b24fe64515412fd21dd81e82adcf6c79dcc93bb8e6a/scikit_learn-0.20.2-cp37-cp37m-macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64.whl (7.8MB)
    100% |████████████████████████████████| 7.8MB 901kB/s 
Requirement already satisfied: scipy>=0.13.3 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from scikit-learn->sklearn) (1.2.0)
Requirement already satisfied: numpy>=1.8.2 in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (from scikit-learn->sklearn) (1.15.4)
Installing collected packages: scikit-learn, sklearn
Successfully installed scikit-learn-0.20.2 sklearn-0.0
You are using pip version 18.1, however version 19.0.1 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.

Run from IDLE 3.7
/Users/JimStone/Documents/GitHub/pytorch-rbm/mnist_example.py

 $ pip3 install torch 
Requirement already satisfied: torch in /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages (1.0.0)
You are using pip version 18.1, however version 19.0.1 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.
==

# PyTorch Examples

A repository showcasing examples of using [PyTorch](https://github.com/pytorch/pytorch)

- MNIST Convnets
- Word level Language Modeling using LSTM RNNs
- Training Imagenet Classifiers with Residual Networks
- Generative Adversarial Networks (DCGAN)
- Variational Auto-Encoders
- Superresolution using an efficient sub-pixel convolutional neural network
- Hogwild training of shared ConvNets across multiple processes on MNIST
- Training a CartPole to balance in OpenAI Gym with actor-critic
- Natural Language Inference (SNLI) with GloVe vectors, LSTMs, and torchtext
- Time sequence prediction - create an LSTM to learn Sine waves

Additionally, a list of good examples hosted in their own repositories:

- [Neural Machine Translation using sequence-to-sequence RNN with attention (OpenNMT)](https://github.com/OpenNMT/OpenNMT-py)

==

conda list

 $ conda info

     active environment : base
    active env location : /anaconda3
            shell level : 1
       user config file : /Users/JimStone/.condarc
 populated config files : /Users/JimStone/.condarc
          conda version : 4.6.2
    conda-build version : 3.17.6
         python version : 3.7.2.final.0
       base environment : /anaconda3  (writable)
           channel URLs : https://repo.anaconda.com/pkgs/main/osx-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/free/osx-64
                          https://repo.anaconda.com/pkgs/free/noarch
                          https://repo.anaconda.com/pkgs/r/osx-64
                          https://repo.anaconda.com/pkgs/r/noarch
          package cache : /anaconda3/pkgs
                          /Users/JimStone/.conda/pkgs
       envs directories : /anaconda3/envs
                          /Users/JimStone/.conda/envs
               platform : osx-64
             user-agent : conda/4.6.2 requests/2.21.0 CPython/3.7.2 Darwin/15.3.0 OSX/10.11.3
                UID:GID : 504:20
             netrc file : None
           offline mode : False

Instructions for using Pytorch




Graphics better in pylab for
reinforcement_q_learning.ipynb

The repository should have been cloned into a directory named "foo" located in whichever directory you ran the git clone command from.

pip install gym -t any/path/i/like

cd /anaconda3/bin
./pip install gym = tha tworked so that python notebook works
Graphics better in pylab for
reinforcement_q_learning.ipynb

But Reinforcement Learning (DQN) Tutorial does work with gym and showns graphics ok of cart. reinforcement_q_learning.py
reinforce.py has no graphics but it works.

Get started
https://pytorch.org/get-started/locally/

https://pytorch.org/

Anaconda is our recommended package manager since it installs all dependencies. 

cd

conda update -n base -c defaults conda

conda install pytorch torchvision -c pytorch

  environment location: /anaconda3

JVStoneMacBookAir:~ JimStone$ which python
/anaconda3/bin/python
(base) JVStoneMacBookAir:~ JimStone$ python
Python 3.7.1 (default, Dec 14 2018, 13:28:58) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin

conda install python=3.7.2

$ python
Python 3.7.2 (default, Dec 29 2018, 00:00:04) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
>>> quit()
(base) JVStoneMacBookAir:~ JimStone$ which python
/anaconda3/bin/python

Then 
https://pytorch.org/tutorials/

If you would like to do the tutorials interactively via IPython / Jupyter, each tutorial has a download link for a Jupyter Notebook and Python source code.

Can download code or Jupyer nb from within each tutoria.

BUT examples only have code.

==


mkdir pytorch_code

cd pytorch_code/

pip3 install torch torchvision

git clone https://github.com/pytorch/examples.git

history
 pip3 install gym

git clone https://github.com/openai/gym

 pip3 install gym

cd gym
pip install -e .

python3.7 -m pip install gym

/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages

>>> help> modules torch
help> modules torch


help> modules gym

Here is a list of modules whose name or summary contains 'gym'.
If there are any, enter a module name to get more help.

gym 
gym.core 
gym.envs 
… etc

pip3 install matplotlib

pip3 installs to python3

Used above to get this to work
Reinforcement Learning (DQN) Tutorial
from
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
which has .py and .ipynb BUT can only get .py to work using IDLE with python3,
but .pynb not work in jupyter lab 

Also have:
reinforce.py from
/Users/JimStone/Documents/pytorch_code/examples
which was created using
git clone https://github.com/pytorch/examples.git

/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages


Ensure cd to correct dir before using pip3:

cd /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages

[]: !ipython profile locate default
/Users/JimStone/.ipython/profile_default

Since the release of macOS Sierra, when in Finder, it is now possible to use the shortcut:

 CMD + SHIFT + .


\fi

cp -R torch /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages  /anaconda3/pkgs
 IDLE: Examples/regression and VAE works.
 
 BUT Spyder: ModuleNotFoundError: No module named 'torch'
 
conda list

To install pytorch modules do: 
conda install pytorch torchvision -c pytorch

image classification:
\url{https://github.com/pytorch/examples/blob/master/mnist/main.py}

See \url{http://deeplizard.com/learn/video/v5cngxo4mIg}

Facebook created PyTorch
d it?s best for prototyping or small-scale projects. 

  conda install pytorch torchvision -c pytorch
 
 check using
 
  conda list pytorch
 
% JVStoneMacBookAir:~ JimStone$ conda list pytorch
%# packages in environment at /Users/JimStone/anaconda3:
%#
%# Name                    Version                   Build  Channel
%pytorch                   0.4.1           py36_cuda0.0_cudnn0.0_1    pytorch

Book reviews

https://www.kdnuggets.com/2018/09/aggarwal-neural-networks-textbook.html
https://insidebigdata.com/category/book-review/
https://medium.com/datadriveninvestor/a-curated-list-of-artificial-intelligence-ai-courses-books-video-lectures-and-papers-d2f584ca14f8

===============

TO get visdom into conda packages

$ git clone https://github.com/facebookresearch/visdom
$ cd /Users/JimStone/anaconda3/bin
$  ./pip install visdom

Collecting visdom
  Using cached https://files.pythonhosted.org/packages/97/c4/5f5356fd57ae3c269e0e31601ea6487e0622fedc6756a591e4a5fd66cc7a/visdom-0.1.8.8.tar.gz
Requirement already satisfied: numpy>=1.8 in /Users/JimStone/anaconda3/lib/python3.7/site-packages (from visdom) (1.15.4)
Requirement already satisfied: scipy in /Users/JimStone/anaconda3/lib/python3.7/site-packages (from visdom) (1.1.0)
Requirement already satisfied: requests in /Users/JimStone/anaconda3/lib/python3.7/site-packages (from visdom) (2.21.0)
Requirement already satisfied: tornado in /Users/JimStone/anaconda3/lib/python3.7/site-packages (from visdom) (5.1.1)
Requirement already satisfied: pyzmq in /Users/JimStone/anaconda3/lib/python3.7/site-packages (from visdom) (17.1.2)
Requirement already satisfied: six in /Users/JimStone/anaconda3/lib/python3.7/site-packages (from visdom) (1.12.0)
Collecting torchfile (from visdom)
  Using cached https://files.pythonhosted.org/packages/91/af/5b305f86f2d218091af657ddb53f984ecbd9518ca9fe8ef4103a007252c9/torchfile-0.1.0.tar.gz
Collecting websocket-client (from visdom)
  Using cached https://files.pythonhosted.org/packages/26/2d/f749a5c82f6192d77ed061a38e02001afcba55fe8477336d26a950ab17ce/websocket_client-0.54.0-py2.py3-none-any.whl
Requirement already satisfied: pillow in /Users/JimStone/anaconda3/lib/python3.7/site-packages (from visdom) (5.3.0)
Requirement already satisfied: idna<2.9,>=2.5 in /Users/JimStone/anaconda3/lib/python3.7/site-packages (from requests->visdom) (2.8)
Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /Users/JimStone/anaconda3/lib/python3.7/site-packages (from requests->visdom) (3.0.4)
Requirement already satisfied: urllib3<1.25,>=1.21.1 in /Users/JimStone/anaconda3/lib/python3.7/site-packages (from requests->visdom) (1.24.1)
Requirement already satisfied: certifi>=2017.4.17 in /Users/JimStone/anaconda3/lib/python3.7/site-packages (from requests->visdom) (2018.11.29)
Building wheels for collected packages: visdom, torchfile
  Running setup.py bdist_wheel for visdom ... done
  Stored in directory: /Users/JimStone/Library/Caches/pip/wheels/ee/87/ce/a5023722374ca73b57fc8d4284ba6f973c01219b3c385a07e0
  Running setup.py bdist_wheel for torchfile ... done
  Stored in directory: /Users/JimStone/Library/Caches/pip/wheels/b1/c3/d6/9a1cc8f3a99a0fc1124cae20153f36af59a6e683daca0a0814
Successfully built visdom torchfile
Installing collected packages: torchfile, websocket-client, visdom
Successfully installed torchfile-0.1.0 visdom-0.1.8.8 websocket-client-0.54.0
