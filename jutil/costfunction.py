import numpy as np
import jutil.linalg

class ScaledCostFunction(object):
    def __init__(self, D, J):
        self._D, self._J = D, J
        assert len(D.shape) == 2
        assert D.shape[0] == D.shape[1]
        assert D.shape[1] = J.n
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

