# Computer Code

> custom computer code that allows readers to repeat the results.

## About

There are three code files: common.py, Main.ipynb, and SI.ipynb.

The notations in the code are consistent with those in the manuscript and the supplementary information.

The order of the code is aligned with that of the manuscript (Main.ipynb) and the supplementary information (SI.ipynb).

### common.py
> shared variables and functions for Main.ipynb and SI.ipynb

* symbols 
* plotting and graphing

### Main.ipynb
> figures in the manuscript and the rebuttal letter

* plotting and graphing


### SI.ipynb
> results and figures in the supplementary information

* expressions: equations, inequalities, solutions, ...
* plotting and graphing

## Instructions

The code is organized into python files or ipython notebooks. 

The software, module and hardware list is given below.

* Software

> Python 3.8.8 and above

* Module

name | version | build | channel 
------------ | ------------- | ------------- | -------------
numpy | 1.19.5 | pypi_0 | pypi
sympy | 1.8 | py38hecd8cb5_0 | 
matplotlib | 3.3.4 | py38hecd8cb5_0 |
seaborn | 0.11.1 | pyhd3eb1b0_0 | 
mpl_toolkits 
warnings 

> If we have matplotlib installed, we would be able to import mpl_toolkits directly.

* OS

> Mac OS X, Windows, or Linux


We use Anaconda GUI to run our code (which comes with the packages automatically installed). Otherwise, we may run `pip install` with the name and version for every candidate item.


### Get Started

The figures are saved in the two folders below.
```python
'./figures/main/'
```
```python
'./figures/SI/'
```

We can change the corresponding 
```python
_Figure_PATH_
```
if needed.

### How to Obtain the Expressions and Figures

More detailed explanations are given in the comments (for python files and ipython notebooks) and markdown notes (for ipython notebooks).

In particular, for every user-defined function, the corresponding docstring summarize its behavior and document its arguments.
