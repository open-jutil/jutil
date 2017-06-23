import numpy as np
from numpy.testing import assert_almost_equal
from jutil.cg import conj_grad_solve, conj_grad_tall_solve
from jutil.preconditioner import JacobiPreconditioner


def test_cg():
    n, m = 7, 4
    A = np.random.rand(n, n)
    A = A.T.dot(A)
    for i in range(n):
        A[i, i] += 1  # + i ** 15
    x_tall = np.random.rand(n, m)
    x = x_tall[:, 0]
    b, b_tall = A.dot(x), A.dot(x_tall)

    assert_almost_equal(conj_grad_solve(A, b,
                                        max_iter=100, abs_tol=0, rel_tol=1e-10), x)
    assert_almost_equal(conj_grad_solve(A, b, P=JacobiPreconditioner(A),
                                        max_iter=100, abs_tol=0, rel_tol=1e-10), x)
    assert_almost_equal(conj_grad_tall_solve(A, b_tall,
                                             max_iter=100, abs_tol=0, rel_tol=1e-10), x_tall)
    assert_almost_equal(conj_grad_tall_solve(A, b_tall, P=JacobiPreconditioner(A),
                                             max_iter=100, abs_tol=0, rel_tol=1e-10), x_tall)
