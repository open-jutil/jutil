import numpy as np
import scipy.sparse
from jutil.norms import *
from numpy.testing import assert_almost_equal

h = 1e-6


def fd_jac(norm, x):
    return np.asarray([(norm(x + h * np.eye(len(x))[:, i]) - norm(x)) / h
                       for i in range(len(x))])


def fd_hess(norm, x):
    return np.asarray([(norm.jac(x + h * np.eye(len(x))[:, i]) - norm.jac(x)) / h
                       for i in range(len(x))])


def hess(norm, x):
    return np.asarray([norm.hess_dot(x, np.eye(len(x))[:, i])
                       for i in range(len(x))])


def execute_norm(norm):
    x = np.arange(1., 6.)[::-1]
    assert_almost_equal(fd_jac(norm, x), norm.jac(x), decimal=5)
    assert_almost_equal(fd_hess(norm, x), hess(norm, x), decimal=5)
    assert_almost_equal(np.diag(hess(norm, x)), norm.hess_diag(x), decimal=5)


def test_tv():
    W = np.zeros((6, 4))
    for i in range(4):
        W[i, i] = 1 + i
    W[4, 3] = 1
    W[4, 2] = -1
    W[5, 3] = 1
    norm = WeightedTV(LPPow(1., 0), W, [2, 4])

    x = 1 + np.arange(4.)

    assert_almost_equal(fd_jac(norm, x), norm.jac(x))
    assert_almost_equal(fd_hess(norm, x), hess(norm, x))

    S1 = np.zeros((4, 4))
    for i in range(4):
        base = np.zeros(4)
        base[i] = 1
        S1[:, i] = norm._map_jac_dot(x, base)
    S2 = np.zeros((4, 4))
    for i in range(4):
        base = np.zeros(4)
        base[i] = 1
        S2[i, :] = norm._map_jacT_dot(x, base)
    assert_almost_equal(S1, S2)


weight = scipy.sparse.csr_matrix(np.arange(25).reshape(5, 5) / 25.)
for name, norm in [
        ("L1", L1()),
        ("L2Square", L2Square()),
        ("WeightedL2Square", WeightedL2Square(weight)),
        ("WeightedNorm_L2Square", WeightedNorm(L2Square(), weight)),
        ("WeightedNormLPPow", WeightedNorm(LPPow(1.1, 0), weight)),
        ("Huber", Huber(1.5)),
        ("Ekblom", Ekblom(1.5, 2.)),
        ("BiSquared_4", BiSquared(4.)),
        ("BiSquared_22", BiSquared(22.)),
        ("WeightedNorm_Huber", WeightedNorm(Huber(1.5), weight))
        ]:
    current_module = __import__(__name__)
    test_function = (lambda y: lambda: execute_norm(y))(norm)
    test_function.__name__ = "test_" + name
    setattr(current_module, test_function.__name__, test_function)

if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
