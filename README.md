# pyDPP

[![PyPI version shields.io](https://img.shields.io/pypi/v/pydpp.svg)](https://pypi.python.org/pypi/pydpp)  [![Build Status](https://travis-ci.org/satwik77/pyDPP.svg?branch=master)](https://travis-ci.org/satwik77/pyDPP) [![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)

A python package for sampling from determinantal point processes. Below are instances of sampling from a bicluster and from a random set of points using pyDPP. Refer to examples and references for more information.

![](https://raw.githubusercontent.com/satwik77/pyDPP/master/example/random_pt.png)

![](https://raw.githubusercontent.com/satwik77/pyDPP/master/example/bicluster.png)



## Usage:

```python
>>> from pydpp.dpp import DPP
>>> import numpy as np
>>> X = np.random.random((10,10))
>>> dpp = DPP(X)
>>> dpp.compute_kernel(kernel_type = 'rbf', sigma= 0.4)   # use 'cos-sim' for cosine similarity
>>> samples = dpp.samples()                   # samples := [1,7,2,5]
>>> ksamlpes = dpp.sample_k(3)                # ksamples := [5,8,0]
```

Refer to examples/test-dpp.ipynb for more on usage.

## Installation:

#### Stable:

```shell
$ pip install -U pydpp
```

#### Dev:

To get the project's source code, clone the github repository:

```shell
$ git clone https://github.com/satwik77/pyDPP.git
$ cd pyDPP
```

Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv venv
$ source venv/bin/activate
```

Install all the required packages:

```shell
$ pip install -r requirements.txt
```

Install the package by running the following command from the root directory of the repository:

```shell
$ python setup.py install	
```



### Requirements

- Numpy
- Scipy

### Compatibility

The package has been tested with python 2.7 and python 3.5.2

## References

- Kulesza, A. and Taskar, B., 2011. k-DPPs: Fixed-size determinantal point processes. In Proceedings of the 28th International Conference on Machine Learning (ICML-11) (pp. 1193-1200). [[paper](https://homes.cs.washington.edu/~taskar/pubs/kdpps_icml11.pdf)]

- Kulesza, A. and Taskar, B., 2012. Determinantal point processes for machine learning. Foundations and Trends® in Machine Learning, 5(2–3), pp.123-286. [[paper](http://www.alexkulesza.com/pubs/dpps_fnt12.pdf)]
