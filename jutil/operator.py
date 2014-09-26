import numpy as np
import numpy.linalg as la



class JacobiCGWrapper(object):
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


class CostFunctionCGWrapper(object):
    def __init__(self, J, x, lmpar=0):
        self._J = J
        self._x = x
        self._lmpar = lmpar

    def dot(self, vec):
        return self._J.hess_dot(self._x, vec) + self._lmpar * vec

    def cond(self, vec):
        return vec.copy() * (1. + self._lmpar)

    @property
    def shape(self):
        return (self._J.n, self._J.n)


class AT_dot_A_plus_lambda_I_CGWrapper(object):
    def __init__(self, A, lambd):
        self._A = A
        self._lambda = lambd

    def dot(self, x):
        return self._A.T.dot(self._A.dot(x)) + self._lambda * x

    def cond(self, x):
        return x.copy()

    @property
    def shape(self):
        return (self._A.shape[1], self._A.shape[1])


class TTWrapper(object):
    def __init__(self, A, B, lambd):
        self._A = A
        self._B = B
        self._lambda = lambd

    def dot(self, x):
        return self._A.T.dot(self._A.dot(x)) + self._lambda * self._B.T.dot(self._B.dot(x))

    def cond(self, x):
        return x.copy()

    @property
    def shape(self):
        return (self._A.shape[1], self._A.shape[1])


class Dot(object):
    def __init__(self, A, B, a=1):
        self._A, self._B, self._a = A, B, a
        assert self._A.shape[1] == self._B.shape[0]

    def dot(self, x):
        return self._a * self._A.dot(self._B.dot(x))

    @property
    def shape(self):
        return (self._A.shape[0], self._B.shape[1])


class Plus(object):
    def __init__(self, A, B, a=1, b=1):
        self._A, self._B, self._a, self._b = A, B, a, b
        assert self._A.shape == self._B.shape

    def dot(self, x):
        return self._a * self._A.dot(x) + self._b * self._B.dot(x)

    @property
    def shape(self):
        return (self._A.shape[0], self._A.shape[1])


class Function(object):
    def __init__(self, F, (n, m), a=1):
        self._F = F
        self._shape = (n, m)
        self._a = a

    def dot(self, x):
        return self._a * self._F(x)

    @property
    def shape(self):
        return self._shape

