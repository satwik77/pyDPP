=====
pyDPP
=====

A python package for sampling from determinantal point processes. Below are instances of sampling from a bicluster and from a random set of points using pyDPP. Refer to examples for more information.

.. image:: https://raw.githubusercontent.com/satwik77/pyDPP/master/example/original_points_bicluster.png?token=AKhAbafMvTsLS2qPVEJJUgrKxvOA-X0Uks5b0beYwA%3D%3D 
    :width: 150pt

.. image:: https://raw.githubusercontent.com/satwik77/pyDPP/master/example/dpp_biselection_k6.png?token=AKhAbcyy19XZFqzPq1LAsQne25brrMnNks5b0baVwA%3D%3D 
    :width: 150pt

.. image:: https://raw.githubusercontent.com/satwik77/pyDPP/master/example/dpp_biselection_k8.png?token=AKhAbYorZxjSQFvScnSLS4wSS3K2MnMwks5b0bfHwA%3D%3D
    :width: 150pt


.. image:: https://raw.githubusercontent.com/satwik77/pyDPP/master/example/original_points_random.png?token=AKhAbe5GRE4Xunr27vH3yZuhwV_VyZmZks5b0bhDwA%3D%3D 
    :width: 150pt

.. image:: https://raw.githubusercontent.com/satwik77/pyDPP/master/example/random_selection_k12.png?token=AKhAbfUxxoDHR1K7lgM_viUaqUHzar31ks5b0bhPwA%3D%3D 
    :width: 150pt

.. image:: https://raw.githubusercontent.com/satwik77/pyDPP/master/example/dpp_selection_k12.png?token=AKhAbS05A3CBgKfXR9P7i4adhlM7Q-whks5b0bhYwA%3D%3D
    :width: 150pt



Usage
-----

Usage example:

::

  >>> from pydpp.dpp import DPP
  >>> import numpy as np
  >>> X = np.random.random((10,10))
  >>> dpp = DPP(X)
  >>> dpp.compute_kernel(kernel_type = 'rbf', sigma= 0.4)		# use 'cos-sim' for cosine similarity
  >>> samples = dpp.samples()			# samples := [1,7,2,5] 
  >>> ksamlpes = dpp.sample_k(3)		# ksamples := [5,8,0]

Installation
------------

To get the project's source code, clone the github repository:

::

  $ git clone https://github.com/satwik77/pyDPP.git
  $ cd pyDPP

Create a virtual environment and activate it. [optional]

::

  $ [sudo] pip install virtualenv
  $ virtualenv -p python3 venv
  $ source venv/bin/activate
  (venv)$ 

Next, install all the dependencies in the environment.

::

  (venv)$ pip install -r requirements.txt


Install the package into the virtual environment.

::

  (venv)$ python setup.py install

Requirements
^^^^^^^^^^^^
- Numpy 
- Scipy

To run the example jupyter notebook you need install jupyter notebook, sklearn and matplotlib.

Compatibility
^^^^^^^^^^^^^
The package has been test with python 2.7 and python 3.5.2