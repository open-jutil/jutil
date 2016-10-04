#
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Ungermann
#

import numpy as np


def _safe_inverse(vec):
    result = np.array(vec, copy=True)
    nonzero = (result != 0)
    result[nonzero] = 1. / result[nonzero]
    return result


class JacobiPreconditioner(object):
    def __init__(self, A):
        assert A.shape[0] == A.shape[1]
        assert hasattr(A, "diagonal")
        self._A = A
        self._diagonal = _safe_inverse(A.diagonal())

    def dot(self, x):
        return (self._diagonal * x.T).T

    @property
    def shape(self):
        return self._A.shape


class CostFunctionPreconditioner(object):
    def __init__(self, J, x_i):
        self._J = J
        self._x_i = x_i
        if hasattr(J, "hess_diag"):
            self._diagonal = _safe_inverse(J.hess_diag(self._x_i))
            self.dot = self._dot_diag
        else:
            self.dot = lambda x: np.array(x, copy=True)

    def _dot_diag(self, x):
        return (self._diagonal * x.T).T

    @property
    def shape(self):
        return (self._J.n, self._J.n)
