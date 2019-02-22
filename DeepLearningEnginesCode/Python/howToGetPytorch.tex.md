
Open the file howToGetPytorch.md to see latex rendered properly (e.g. $y=mx$).

Note: I am not expert in installing software. The text below is intended as a general guide, based on my own experience with Python. The book Deep Learning Engines is mainly about the mathematics of neural networks, so this code is really an added extra. However, if you spot any errors here then please let me know. 

My own machine is a mac, so most information refers to Python on macs.

Installing Python on a PC
=================

See Pythonxy or WinPython.

Installing Python on Unix (eg Mac/linux)
=================

WARNING: DO NOT DELETE THE NATIVE VERSION OF PYTHON ON THE MAC OR LINUX BECAUSE THE OS NEEDS IT. JUST INSTALL A DIFFERENT (NEWER) VERSION AS BELOW.

The simplest and most complete Python can be downloaded from Anaconda:

https://www.anaconda.com/distribution/

To find out where the default version of python is stored, type this to unix shell:

$ which python3

/Users/JimStone/anaconda3/bin/python3

Or, to download the basic Python language go to:

https://www.python.org/

Assuming you have Anaconda, it offers various programs to use with Python, including Python Notebooks.
I use the IPython application called Spyder which can be launched from within Anaconda.

In order to use standard neural network packages, you need to install (usually just one) software platforms:

Pytorch (owned by Facebook)

scikit-learn (funded by various public bodies)

TensorFlow (owned by Google)

Most examples in this book rely on Pytorch.
Both Pytorch and scikit-learn allow fairly easy access to the underlying Python code.

Installing Pytorch
=================

Install modules pytorch and torchvision:

$ conda install pytorch torchvision -c pytorch
(The -c tells conda to look in pytorch online)

Other usefull commands:

$ conda list

$ conda info

Installing Modules Using pip
=======================

For some reason conda cannot find all modules, so use pip for those. 

To use some graphical examples involving computer games you will need gym:

Install gym into anaconda package

$ pip install gym

To install scikit-learn do

$ pip install sklearn

To make sure the files get put in the right place (ie in anaconda's domain), 
you may need to ensure you are using the pip that is part of anaconda:

$ which pip
/Users/JimStone/anaconda3/bin/pip

Do this by running anaconda's pip, for example:

$ cd /Users/JimStone/anaconda3/bin

$ ./pip install gym

Requirement already satisfied: gym in /Users/JimStone/anaconda3/lib/python3.7/site-packages (0.10.11)
etc

If you need to tell pip where to put downloaded files then use the -t flag:

$ pip3 -t, --target adirectory

Python, IPython, Notebooks and Jupyter Labs
=================

These are the different interfaces for using Python, listed from most primitive to least.

Python is often run from a command line using an IDLE; this the basic python.

IPython is a more sophisticated interface that has integrated windows and displays variables (Spyder under Anaconda is an IPython application).

Notebooks are great for teaching because they display code, comments and outputs in one graphical display.

Jupyter Labs are an upgraded form of Notebook.

You can export the code from a Notebook or Lab to run on its own (see below).
 
Saving Notebook as Python Code
===========================

Sometimes an online example is presented as a Jupyter Notebook or Jupyter Lab.
You can save the Python code to run within a Python application (e.g. Spyder).

From within Anaconda’s Jupyter Notebook, find the menu item:

	Download as …

or from Jupyter Lab, find the menu item

	Export Notebook as …

and download to file *.py.


