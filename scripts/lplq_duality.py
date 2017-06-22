from __future__ import print_function

import jutil
import jutil.norms as norms
from jutil.minimizer import minimize
import numpy as np

jutil.misc.setup_logging()

n = 100
p = 2  # 1.1
q = 1. / (1. - (1. / p))


class CostFunction(object):
    def __init__(self, norm, y):
        self._norm, self._y = norm, y
        self.m, self.n = len(y), len(y)

    def init(self, x_i):
        self.__call__(x_i)

    def __call__(self, x):
        self._chisq = self._norm(x - self._y) / self.m
        return self._chisq

    def jac(self, x):
        return self._norm.jac(x - self._y) / self.m

    def hess_dot(self, x, vec):
        return self._norm.hess_dot(x - self._y, vec) / self.m

    def hess_diag(self, x):
        return np.ones_like(x)

    @property
    def chisq(self):
        return self._chisq

A = np.diag(np.random.random(n))
AI = np.asarray(np.asmatrix(A).I)
lp = norms.LPPow(p, 1e-20)
lq = norms.LPPow(q, 1e-20)

lAp = norms.WeightedNorm(lp, A)
lAq = norms.WeightedNorm(lq, AI.T * n)  # n for division by m in J

x = np.random.random(n)
J = CostFunction(lAp, x)


def P(_):
    return (1. / q) * lAq.jac(x)

print(AI ** 2 * n ** 2)
print(P(None, np.ones(n)))
tol = {"max_iteration": 10}

minimize(J, np.zeros(len(x)), method="SteepestDescent", tol=tol)
minimize(J, np.zeros(len(x)), tol=tol)
minimize(J, np.zeros(len(x)), method="SteepestDescent", options={"preconditioner": P}, tol=tol)

print(J(J._y))
