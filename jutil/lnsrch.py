import scipy.optimize

import logging

LOG = logging.getLogger(__name__)


def lnsrch(x_0, f_0, g, p, func, stpmax=None):
    """
    Wrapper around scipy linesearch.
    """
    alpha_i, _, f_i = scipy.optimize.linesearch.line_search_armijo(
        func, x_0, p, g, f_0)
    return False, f_i, x_0 + alpha_i * p
