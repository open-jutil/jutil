import numpy as np


class JacobiMatrixOperator(object):
    def __init__(self, A):
        assert A.shape[0] == A.shape[1]
        self._A = A
        self._diagonal = np.diag(A)
        nonzero = self._diagonal != 0
        self._diagonal[nonzero] = 1. / self._diagonal[nonzero]

    def dot(self, x):
        return self._A.dot(x)

    def cond(self, x):
        return (self._diagonal * x.T).T

    @property
    def shape(self):
        return self._A.shape


class CostFunctionOperator(object):
    def __init__(self, J, x, lmpar=0):
        self._J = J
        self._x = x
        self._lmpar = lmpar
        self.dot = self._dot if lmpar == 0 else self._dot_lam
        self.cond = self._cond if lmpar == 0 else self._cond_lam

    def _dot(self, vec):
        return self._J.hess_dot(self._x, vec)

    def _cond(self, vec):
        return vec.copy()

    def _dot_lam(self, vec):
        return self._J.hess_dot(self._x, vec) + self._lmpar * vec

    def _cond_lam(self, vec):
        return vec.copy() * (1. + self._lmpar)

    @property
    def shape(self):
        return (self._J.n, self._J.n)


class Dot(object):
    def __init__(self, A, B, a=1):
        self._A, self._B, self._a = A, B, a
        self.dot = self._dot if a == 1 else self._dot_a
        assert self._A.shape[1] == self._B.shape[0]

    def _dot(self, x):
        return self._A.dot(self._B.dot(x))

    def _dot_a(self, x):
        return self._a * self._A.dot(self._B.dot(x))

    @property
    def shape(self):
        return (self._A.shape[0], self._B.shape[1])


class Plus(object):
    def __init__(self, A, B):
        self._A, self._B = A, B
        assert self._A.shape == self._B.shape

    def dot(self, x):
        return self._A.dot(x) + self._B.dot(x)

    @property
    def shape(self):
        return (self._A.shape[0], self._A.shape[1])


class Function(object):
    def __init__(self, F, (n, m), a=1):
        self._F = F
        self._shape = (n, m)
        self._a = a
        self.dot = self._dot if a == 1 else self._dot_a

    def _dot_a(self, x):
        return self._a * self._F(x)

    def _dot(self, x):
        return self._F(x)

    @property
    def shape(self):
        return self._shape
