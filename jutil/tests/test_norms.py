import numpy as np
import scipy.sparse as sp
from jutil.norms import *
from numpy.testing import assert_almost_equal


def test_jacobian():
    weight = np.arange(25).reshape(5, 5) / 25.
    x = np.arange(1, 6)[::-1]
    h = 1e-6
    for norm in [NormL2Square(),
                 WeightedL2Square(weight),
                 WeightedNorm(NormL2Square(), weight),
                 WeightedNorm(NormLPPow(1.1, 0), weight),
                 Huber(1.5),
                 Ekblom(1.5, 2.),
                 BiSquared(4.),
                 BiSquared(22.),
                 WeightedNorm(Huber(1.5), weight)]:
        fd_jac = np.asarray([(norm(x + h * np.eye(5)[:, i]) - norm(x)) / h for i in range(len(x))])
        assert_almost_equal(
            fd_jac, norm.jac(x), decimal=5)


def test_tv():
    W = np.zeros((6, 4))
    for i in range(4):
        W[i, i] = 1 + i
    W[4, 3] = 1
    W[4, 2] = -1
    W[5, 3] = 1
    norm = WeightedTVNorm(NormLPPow(1., 0), W, [2, 4])

    x = 1 + np.arange(4)
    h = 1e-6

    def fd_jac(norm, x):
        return np.asarray([(norm(x + h * np.eye(4)[:, i]) - norm(x)) / h
                           for i in range(len(x))])

    def fd_hess(norm, x):
        return np.asarray([(norm.jac(x + h * np.eye(4)[:, i]) - norm.jac(x)) / h
                           for i in range(len(x))])

    assert_almost_equal(
        fd_hess(norm, x),
        np.asarray([norm.hess_dot(x, np.eye(4)[:, i])
                    for i in range(len(x))]))

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

if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
