
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

<img src="/DeepLearningEnginesCode/Python/tex/24be74da2f8b6c84350a80f0f0aea3c4.svg?invert_in_darkmode&sanitize=true" align=middle width=700.5028387499999pt height=598.1735232000001pt/> conda install pytorch torchvision -c pytorch
(The -c tells conda to look in pytorch online)

Other usefull commands:

<img src="/DeepLearningEnginesCode/Python/tex/8eb8508fe6faf08a760f0850ee76fc03.svg?invert_in_darkmode&sanitize=true" align=middle width=66.72699659999998pt height=22.831056599999986pt/> conda info

Installing Modules Using pip
=======================

For some reason conda cannot find all modules, so use pip for those. 

To use some graphical examples involving computer games you will need gym:

Install gym into anaconda package

<img src="/DeepLearningEnginesCode/Python/tex/68b61e54061c3878c30181b9458cd6ef.svg?invert_in_darkmode&sanitize=true" align=middle width=167.3520486pt height=39.45205440000001pt/> pip install sklearn

To make sure the files get put in the right place (ie in anaconda's domain), 
you may need to ensure you are using the pip that is part of anaconda:

<img src="/DeepLearningEnginesCode/Python/tex/b1a0230d1f59a2608a513b0f56650ee3.svg?invert_in_darkmode&sanitize=true" align=middle width=346.8772295999999pt height=45.84475500000001pt/> cd /Users/JimStone/anaconda3/bin

<img src="/DeepLearningEnginesCode/Python/tex/3e335f12ff462788f309ddef63e7ae01.svg?invert_in_darkmode&sanitize=true" align=middle width=709.36281075pt height=124.74886710000001pt/> pip3 -t, --target adirectory

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


