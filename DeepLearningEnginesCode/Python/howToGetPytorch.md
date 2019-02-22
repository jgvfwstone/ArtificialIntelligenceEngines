
Open the file howToGetPytorch.md to see latex rendered properly (e.g. <img src="/DeepLearningEnginesCode/Python/tex/1cbdf1c3ec380d9a41eb881ea74212cb.svg?invert_in_darkmode&sanitize=true" align=middle width=54.39492464999999pt height=14.15524440000002pt/>).

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

<img src="/DeepLearningEnginesCode/Python/tex/4b09e1c80baeb7726ec2a0c26bc0f7c6.svg?invert_in_darkmode&sanitize=true" align=middle width=700.5028387499999pt height=633.0593544pt/> conda install pytorch torchvision -c pytorch
(The -c tells conda to look in pytorch online)

Other usefull commands:

\$ conda list

<img src="/DeepLearningEnginesCode/Python/tex/aed827aaffba4b207537cec436f2447d.svg?invert_in_darkmode&sanitize=true" align=middle width=558.4490175pt height=164.20092150000002pt/> pip install gym

To install scikit-learn do

<img src="/DeepLearningEnginesCode/Python/tex/ccc47f81c25f969de60a446f1e4439cc.svg?invert_in_darkmode&sanitize=true" align=middle width=700.2744836999999pt height=85.29681270000002pt/> which pip
/Users/JimStone/anaconda3/bin/pip

Do this by running anaconda's pip, for example:

<img src="/DeepLearningEnginesCode/Python/tex/48bed7afb24ae234d519a74f1c4c036e.svg?invert_in_darkmode&sanitize=true" align=middle width=265.98896114999997pt height=24.65753399999998pt/> ./pip install gym

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


