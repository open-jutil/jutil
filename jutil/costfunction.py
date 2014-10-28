import numpy as np
import jutil.linalg

class ScaledCostFunction(object):
    def __init__(self, D, J):
        self._D, self._J = D, J
        assert len(D.shape) == 2
        assert D.shape[0] == D.shape[1]
        assert D.shape[1] == J.n
        assert D.shape[0] > 1
        self.m, self.n = J.M, J.n

    def init(self, x_i):
        self.__call__(x_i)

    def __call__(self, x):
        return self._J(D.dot(x))

    def jac(self, x):
        return self._J.jac(D.dot(x)).dot(self._D)

    def hess(self, x):
        return self._D.T.dot(self._J.hess(self._D.dot(x)).dot(self._D))

    def hess_dot(self, x, vec):
        return self._D.T.dot(self._J.hess_dot(self._D.dot(x), self._D.dot(vec)))

    def hess_diag(self, x):
        return jutil.linalg.quick_diagonal_product(self.D, J.hess_diag(x))

    @property
    def chisq(self):
        return self._J.chisq
    @property
    def chisq_m(self):
        return self._J.chisq_m
    @property
    def chisq_a(self):
        return self._J.chisq_a


class CountingCostFunction(object):
    def __init__(self, J):
        self._J = J
        self.m, self.n = J.m, J.n
        self.cnt_call, self.cnt_jac, self.cnt_hess_dot, self.cnt_hess, self.cnt_hess_diag = 0, 0, 0, 0, 0

    def init(self, x_i):
        self.__call__(x_i)

    def __call__(self, x):
        self.cnt_call += 1
        return self._J(x)

    def jac(self, x):
        self.cnt_jac += 1
        return self._J.jac(x)

    def hess(self, x):
        self.cnt_hess += 1
        return self._J.hess(x)

    def hess_dot(self, x, vec):
        self.cnt_hess_dot += 1
        return self._J.hess_dot(x, vec)

    def hess_diag(self, x):
        self.cnt_hess_diag += 1
        return J.hess_diag(x)

    @property
    def chisq(self):
        return self._J.chisq
    @property
    def chisq_m(self):
        return self._J.chisq_m
    @property
    def chisq_a(self):
        return self._J.chisq_a

