
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
