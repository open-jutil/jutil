import jutil.minimizer as mini
import numpy as np
from numpy.testing import assert_almost_equal


class CostFunction(object):
    def __init__(self):
        # self._A = np.random.rand(4, 4)
        self._A = np.array(
            [[0.88494541, 0.60670887, 0.74840765, 0.11873982],
             [0.13653893, 0.62102802, 0.80779039, 0.90103182],
             [0.80150651, 0.57204279, 0.41576394, 0.33573904],
             [0.65005687, 0.06166340, 0.01514931, 0.91255390]])
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

    def hess_diag(self, x):
        return np.ones(self.n)

    @property
    def chisq(self):
        return self._chisq

    @property
    def chisq_m(self):
        return self._chisqm

    @property
    def chisq_a(self):
        return self._chisqa


def execute_minimizer(max_it, stepper):
    J = CostFunction()
    optimize = mini.Minimizer(stepper)
    optimize.update_tolerances({"max_iteration": max_it})
    assert_almost_equal(optimize(J, 0.5 * np.ones(J.n)).x, J._x_t, decimal=5)


def execute_minimize(max_it, method, options):
    J = CostFunction()
    x0 = 0.5 * np.ones(J.n)
    result = mini.minimize(J, x0, method=method, options=options, tol={"max_iteration": max_it})["x"]
    assert_almost_equal(result, J._x_t, decimal=5)


def execute_scipy(method):
    J = CostFunction()
    res = mini.scipy_minimize(J, 0.5 * np.ones(J.n), tol=1e-12, method=method)["x"]
    assert_almost_equal(res, J._x_t, decimal=5)


for maxit, stepper, options in [
        (1000, "SteepestDescent", {}),
        (1000, "CauchyPointSteepestDescent", {}),
        (10, "LevenbergMarquardtReduction", {"lmpar": 1e-4, "factor": 100}),
        (10, "LevenbergMarquardtPredictor", {"lmpar": 10., "factor": 100.}),
        (10, "GaussNewton", {}),
        (10, "TruncatedCGQuasiNewton", {}),
        (10, "TrustRegionTruncatedCGQuasiNewton", {})
]:
    current_module = __import__(__name__)
    test_function = (lambda y: lambda: execute_minimize(*y))((maxit, stepper, options))
    test_function.__name__ = "test_minimize_" + stepper
    setattr(current_module, test_function.__name__, test_function)

    instance = getattr(mini, stepper + "Stepper")(**options)
    test_function2 = (lambda y: lambda: execute_minimizer(*y))((maxit, instance))
    test_function2.__name__ = "test_minimizer_" + type(instance).__name__
    setattr(current_module, test_function2.__name__, test_function2)


for method in [
        "BFGS",
        "Newton-CG",
        "CG",
        "trust-ncg",
]:
    current_module = __import__(__name__)
    test_function = (lambda y: lambda: execute_scipy(y))(method)
    test_function.__name__ = "test_scipy_" + method
    setattr(current_module, test_function.__name__, test_function)

if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
