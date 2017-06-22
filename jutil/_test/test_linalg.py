import jutil.linalg
import scipy.sparse as sp
import numpy as np
from numpy.testing import assert_almost_equal


def execute_matrix(A, diag=None):
    if diag is not None:
        B = A
        if hasattr(B, "todense"):
            B = np.asarray(B.todense())
        ATA = B.T.dot(np.diag(diag).dot(B))
    else:
        ATA = A.T.dot(A)
    assert_almost_equal(
        ATA.diagonal().squeeze(),
        jutil.linalg.quick_diagonal_product(A, diag))


def test_sparse_csr():
    A = sp.csr_matrix(np.arange(25).reshape(5, 5) / 25.)
    execute_matrix(A)
    execute_matrix(A, np.arange(5.))


def test_sparse_csc():
    A = sp.csc_matrix(np.arange(25).reshape(5, 5) / 25.)
    execute_matrix(A)
    execute_matrix(A, np.arange(5.))


def test_array():
    A = np.arange(25).reshape(5, 5) / 25.
    execute_matrix(A)
    execute_matrix(A, np.arange(5.))


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
