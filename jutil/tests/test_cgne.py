import numpy as np
from numpy.testing import assert_almost_equal
from jutil.cg import *
from jutil.lsqr import *
from jutil.cgne import *


def test_cgne():
    import jutil.operator
    n, m = 7, 4
    A = np.random.rand(n, m)
    ATA = A.T.dot(A)
    b = np.random.rand(7)
    x = conj_grad_solve(ATA, A.T.dot(b), 100, 1e-10, 1e-10)
    x2 = cgne_solve(A, b, 100, 1e-10, 1e-10, verbose=True)
    assert_almost_equal(x, x2)

if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
