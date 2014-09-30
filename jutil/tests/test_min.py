import jutil.minimizer as mini
import numpy as np
import numpy.linalg as la
from numpy.testing import assert_almost_equal
import time


class CostFunction(object):
    def __init__(self):
        self._A = np.random.rand(4, 4)
        for i in range(4):
            self._A[i, i] += 1
        self._x_t = np.ones(4)
        self._lambda = 0
        self._y = self._A.dot(self._x_t)
        self.m = len(self._y)
        self.n = len(self._x_t)

    def init(self, x_i):
        self.__call__(x_i)

    def __call__(self, x):
        dy = self._A.dot(x) - self._y
        self._chisqm = np.dot(dy, dy) / self.m
        self._chisqa = self._lambda * np.dot(x, x) / self.m
        self._chisq = self._chisqm + self._chisqa
        return self._chisq

    def jac(self, x):
        return (2. * self._A.T.dot(self._A.dot(x) - self._y) + 2. * self._lambda * x) / self.m

    def hess_dot(self, _, vec):
        return (2. * self._A.T.dot(self._A.dot(vec)) + 2. * self._lambda * vec) / self.m

    @property
    def chisq(self):
        return self._chisq

    @property
    def chisq_m(self):
        return self._chisqm

    @property
    def chisq_a(self):
        return self._chisqa


def test_minimizer():
    J = CostFunction()

    for maxit, stepper in [
            (1000, mini.SteepestDescentStepper()),
            (1000, mini.ScaledSteepestDescentStepper()),
            (10, mini.LevenbergMarquardtStepper(1e-4, 100)),
            (10, mini.GaussNewtonStepper()),
            (10, mini.TruncatedQuasiNewtonStepper(1e-4, 10))]:
        optimize = mini.Minimizer(stepper)
        optimize.conv_max_iteration = maxit
        optimize(J, 0.5 * np.ones(J.n)),

        assert_almost_equal(optimize(J, np.zeros(J.n)), J._x_t)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
