[![build status](https://iffgit.fz-juelich.de/unger/jutil/badges/master/build.svg)](https://iffgit.fz-juelich.de/unger/jutil/commits/master) [![coverage report](https://iffgit.fz-juelich.de/unger/jutil/badges/master/coverage.svg)](https://iffgit.fz-juelich.de/unger/jutil/commits/master)

# JUTIL

This package of python utilities is designed to facilitate the solving of medium-sized
ill-posed problems by means of iterative techniques.

An example of how to use it is given in scripts/example.py. The use of further functions
can be derived from the testcases.

Please contact <j.ungermann@fz-juelich.de> if you require assistance.


# Installation

There are other ways to install this software package, but our prefered way is described below.

* Install Anaconda.

  1. Download of miniconda Python3 from https://conda.io/miniconda.html

  2. Install it to home or private drive

  3. Allow the PATH to be extended and re-login or restart your SHELL such that your PATH contains miniconda.


* Configure Anaconda to use the conda-forge channel for packets not present in default channel:

  1. Execute “conda config --add channels conda-forge”


* Retrieve jutil package:

  1. Clone jutil source code (“git clone <URL>")


* Install jutil and its dependencies:

  1. Change into jutil directory

  2. Execute “conda install --file requirements.txt”

  3. Execute “python setup.py install”


# First steps

The basic idea behind this library is to offer tools to simplify the creation
and maintenance of cost function for large konvex minimization problems,
i.e. problems that can be solved via Newton-type solvers. 

A CostFunction class offers necessary methods for the solver to work and
allows an efficient implementation for, e.g., the computation of first
or second derivatives. An explicit aim of the design is to avoid
matrix-matrix-multiplications, which is facilitated by a series of operator
classes. 

Two examples are provided in the scripts directory. scripts/radtrans.py gives
an example for a potentially complicated non-linear forward model wrapped by
a cost function with regularization. scripts/image_reconstruction.py shows
how linear models can be used for a tomographic image reconstruction and denoising
using different norms.

