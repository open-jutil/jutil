[![build status](https://iffgit.fz-juelich.de/unger/jutil/badges/master/build.svg)](https://iffgit.fz-juelich.de/unger/jutil/commits/master) [![coverage report](https://iffgit.fz-juelich.de/unger/jutil/badges/master/coverage.svg)](https://iffgit.fz-juelich.de/unger/jutil/commits/master)

JUTIL
=====

This package of python utilities is designed to facilitate the solving of medium-sized
ill-posed problems by means of iterative techniques.

An example of how to use it is given in scripts/example.py. The use of further functions
can be derived from the testcases.

Please contact <j.ungermann@fz-juelich.de> if you require assistance.


Installation
------------

There are other ways to install this software package, but our prefered way is described below.

* Install Anaconda.

  1. Download of miniconda Python2.7 from https://conda.io/miniconda.html

  2. Install it to home or private drive

  3. Allow the PATH to be extended and re-login or restart your SHELL such that your PATH contains miniconda.


* Configure Anaconda to use the conda-forge channel for packets not present in default channel:

  1. Execute “conda config --add channels conda-forge”

  2. Execute “conda config --add channels defaults”


* Retrieve jutil package:

  1. Clone jutil source code (“git clone git@software.icg.kfa-juelich.de:/joernu76/jutil")


* Install jutil and its dependencies:

  1. Change into jutil directory

  2. Execute “conda install --file requirements.txt”

  3. Execute “python setup.py install”

