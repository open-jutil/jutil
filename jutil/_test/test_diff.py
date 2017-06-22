import scipy.sparse
import numpy as np
import jutil.diff
from numpy.testing import assert_equal, assert_almost_equal


def test_fd_jac():
    def f(x):
        return x ** 3

    assert_almost_equal(
        jutil.diff.fd_jac(f, np.asarray([2, 3])),
        np.asarray([[12, 0], [0, 27]]),
        decimal=4)

    assert_almost_equal(
        jutil.diff.fd_jac_dot(f, np.asarray([2, 3]), np.asarray([1., 0.5])),
        np.asarray([12, 13.5]),
        decimal=4)


def test_fd_hess():
    def f(x):
        return x[0] ** 2 + x[1] ** 3

    assert_almost_equal(
        jutil.diff.fd_hess(f, np.asarray([2, 3])),
        np.asarray([[2, 0], [0, 6 * 3]]),
        decimal=2)

    assert_almost_equal(
        jutil.diff.fd_hess_dot(f, np.asarray([2, 3]), np.asarray([1., 0.5])),
        np.asarray([2, 9]),
        decimal=2)


def get_diff_op_old(mask):
    n = mask.sum() * 3
    count = n / 3

    DiffOp = scipy.sparse.lil_matrix((2 * n, n))
    mask1 = mask.copy().squeeze()
    mask1[1:, :] = mask1[1:, :] & mask1[:-1, :]
    mask1[0, :] = False
    mask2 = mask.copy().squeeze()
    mask2[:, 1:] = mask2[:, 1:] & mask2[:, :-1]
    mask2[:, 0] = False

    indice = np.arange(len(mask.reshape(-1)))[mask.reshape(-1)]
    indmap = dict([(indice[i], i) for i in range(len(indice))])
    for i, j in enumerate(np.where(mask1.reshape(-1))[0]):
        m1 = indmap[j - mask1.shape[0]]
        p1 = indmap[j]
        DiffOp[i, m1] = -1
        DiffOp[i, p1] = 1
        DiffOp[i + count, count + m1] = -1
        DiffOp[i + count, count + p1] = 1
        DiffOp[i + 2 * count, 2 * count + m1] = -1
        DiffOp[i + 2 * count, 2 * count + p1] = 1
    for i, j in enumerate(np.where(mask2.reshape(-1))[0]):
        m1 = indmap[j - 1]
        p1 = indmap[j]
        DiffOp[i + 3 * count, m1] = -1
        DiffOp[i + 3 * count, p1] = 1
        DiffOp[i + 4 * count, count + m1] = -1
        DiffOp[i + 4 * count, count + p1] = 1
        DiffOp[i + 5 * count, 2 * count + m1] = -1
        DiffOp[i + 5 * count, 2 * count + p1] = 1
    DiffOp = DiffOp.tocsr()
    return DiffOp


mask = np.zeros((50, 50), dtype=bool)
for i in range(1, 40):
    for j in range(1, 40):
        mask[i, j] = True


def test_diff():
    import jutil.operator as op
    #  import timeit
    #  print timeit.timeit("get_diff_op(mask, 0, factor=3)", setup="from __main__ import *", number=10)

    A = op.VStack([jutil.diff.get_diff_operator(mask, i, factor=3) for i in [0, 1]])
    B = get_diff_op_old(mask)
    x = np.random.rand(3 * mask.sum())
    assert_equal(A.dot(x), B.dot(x))

if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
