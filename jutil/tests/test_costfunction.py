import numpy as np
from numpy.testing import assert_almost_equal
import jutil.costfunction
import jutil.minimizer


def test_leastsq_costfunction():
    def f(x):
        return np.asarray([x[0] - 2, 2 * (x[1] + 3)])
    J = jutil.costfunction.LeastSquaresCostFunction(f, 2, m=1, epsilon=1e-4)
    x = np.asarray([1., 2.])
    J.init(x)
    assert_almost_equal(J(x), 101)
    assert_almost_equal(J.jac(x), np.asarray([-2, 40]), decimal=3)
    assert_almost_equal(J.hess(x), np.asarray([[2., 0.], [0., 8.]]),decimal=3)
    assert_almost_equal(J.hess_dot(x, x), np.asarray([2., 16]),decimal=3)

    assert_almost_equal(jutil.minimizer.minimize(J, x).x, np.asarray([2, -3]))


def test_wrap_costfunction():
    def f(x):
        return (x[0] - 2.) ** 4 + 4 * (x[1] + 3.) ** 2

    def j(x):
        return np.asarray([4 * (x[0] - 2.) ** 3, 8 * (x[1] + 3.)])
    def h_diag(x):
        return np.asarray([12 * (x[0] - 2.) ** 2, 8])
    def h(x):
        return np.diag(h_diag(x))
    def h_dot(x, v):
        return h_diag(x) * np.asarray(v)

    J = jutil.costfunction.WrapperCostFunction(f, 2, m=1, jac=j, hess_diag=h_diag, epsilon=1e-4)
    x = np.asarray([1., 2.])
    J.init(x)
    assert_almost_equal(J(x), 101)
    assert_almost_equal(J.jac(x), np.asarray([-4, 40]), decimal=2)
    assert_almost_equal(J.hess(x), np.asarray([[12., 0.], [0., 8.]]),decimal=2)
    assert_almost_equal(J.hess_dot(x, x), np.asarray([12., 16]),decimal=2)

    assert_almost_equal(jutil.minimizer.minimize(J, x, tol={"max_iteration":50}).x,
                        np.asarray([2, -3]))


def test_scaled_costfunction():
    def f(x):
        return np.asarray([x[0] - 2, 2 * (x[1] + 3)])
    J_p = jutil.costfunction.LeastSquaresCostFunction(f, 2, m=1, epsilon=1e-4)
    D = np.arange(4).reshape(2, 2)
    DI = np.asarray(np.linalg.inv(D))
    J = jutil.costfunction.ScaledCostFunction(np.arange(4).reshape(2, 2), J_p)
    x = np.asarray([1., 2.])
    J.init(x)
    assert_almost_equal(J(x), J_p(D.dot(x)))
    assert_almost_equal(J.jac(x), jutil.diff.fd_jac(J.__call__, x), decimal=3)
    assert_almost_equal(J.hess(x), jutil.diff.fd_jac(J.jac, x), decimal=3)
    assert_almost_equal(J.hess_dot(x, x), jutil.diff.fd_jac(J.jac, x).dot(x), decimal=3)
    assert_almost_equal(J.hess_diag(x), np.diag(jutil.diff.fd_jac(J.jac, x)), decimal=3)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
