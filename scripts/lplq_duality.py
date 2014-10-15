import jutil.norms as norms
import jutil.lnsrch as lnsrch
from jutil.minimizer import minimize
import numpy as np
import numpy.linalg as la

n = 100
p = 1.1
q = 1. / (1. - (1. / p))
print p, q


class CostFunction(object):
    def __init__(self, norm, y):
        self._norm, self._y = norm, y
        self.m, self.n = len(y), len(y)

    def init(self, x_i):
        self.__call__(x_i)

    def __call__(self, x):
        self._chisqm = self._norm(x - self._y) #/ self.m
        self._chisqa = 0
        self._chisq = self._chisqm
        return self._chisq

    def jac(self, x):
        return self._norm.jac(x - self._y) #/ self.m

    def hess_dot(self, x, vec):
        return self._norm.hess_dot(x - self._y, vec)# / self.m

    @property
    def chisq(self):
        return self._chisq
    @property
    def chisq_m(self):
        return self._chisqm
    @property
    def chisq_a(self):
        return self._chisqa

A = np.diag(np.random.random(n))
AI = np.asarray(np.asmatrix(A).I)
lp = norms.NormLPPow(p, 1e-20)
lq = norms.NormLPPow(q, 1e-20)

lAp = norms.WeightedNorm(lp, A)
lAq = norms.WeightedNorm(lq, AI.T)

x = np.random.random(n)
J = CostFunction(lAp, x)

P = lambda x: (1. / q) * lAq.jac(x)

tol = {"max_iteration": 10}

minimize(J, np.zeros(len(x)), method="SteepestDescent", tol=tol)
minimize(J, np.zeros(len(x)), tol=tol)
minimize(J, np.zeros(len(x)), method="SteepestDescent", options={"preconditioner": P}, tol=tol)

print J(J._y)
