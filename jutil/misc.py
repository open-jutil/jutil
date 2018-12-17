#
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Ungermann
#

import numpy as np
import logging


try:
    try:
        if type(get_ipython()).__name__ == 'ZMQInteractiveShell':  # IPython Notebook!
            from tqdm import tqdm_notebook as tqdm
        else:  # IPython, but not a Notebook (e.g. terminal)
            from tqdm import tqdm
    except NameError:
        from tqdm import tqdm
except ImportError:
    class tqdm(object):
        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable

        def __enter__(self, **kwargs):
            return self

        def __iter__(self):
            for obj in self._iterable:
                yield obj

        def update(self, **kwargs):
            pass

        def __exit__(self, *args):
            return False

        def sp(self, **kwargs):
            pass


def get_lena_256():
    """
    Returns the center of the lena image.
    """
    import scipy.misc
    return scipy.misc.ascent()[384:128:-1, 128:384]


def get_phantom_1():
    """
    Returns a 256x256 test phantom.
    """
    image = 50 * np.ones((256, 256))
    image[:] += (np.arange(256) / 10.)[:, np.newaxis]
    image[:] += (np.arange(256) / 10.)[np.newaxis, :]
    image[100:200, 100:200] = 255.
    image[50:150, 50:150] = 200.
    image[55:145, 55:145] = 0.
    for i, j in [(i, j) for i in range(-25, 25) for j in range(-25, 25)]:
        if abs(i) + abs(j) < 25:
            image[50 + i, 50 + j] = 150
    for i, j in [(i, j) for i in range(-25, 25) for j in range(-25, 25)]:
        if np.hypot(i, j) < 25:
            image[200 + i, 50 + j] = 60
        if np.hypot(i, j) < 10:
            image[50 + i, 200 + j] = 200
    return image


def setup_logging(logfile="jutil.log"):
    logger = logging.getLogger('jutil')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    cf = logging.Formatter('jutil:%(message)s')
    ch.setFormatter(cf)
    logger.addHandler(ch)
    if logfile:
        fh = logging.FileHandler(logfile)
        ff = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(ff)
        logger.addHandler(fh)
