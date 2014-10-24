import jutil.linalg
import scipy.sparse
import numpy as np
from numpy.testing import assert_almost_equal


def execute_matrix(A, diag):
    ATA = A.T.dot(A)
    assert_almost_equal(
            np.diag(ATA),
            jutil.linalg.quick_diagonal_product(A, diag))


def test_sparse_csr():
    A = sp.csr_matrix(np.arange(25).reshape(5, 5) / 25.)
    execute_matrix(A)
    execute_matrix(A, np.arange(5))


def test_sparse_csc():
    A = sp.csc_matrix(np.arange(25).reshape(5, 5) / 25.)
    execute_matrix(A)
    execute_matrix(A, np.arange(5))


def test_array():
    A = np.arange(25).reshape(5, 5) / 25.
    execute_matrix(A)
    execute_matrix(A, np.arange(5))
