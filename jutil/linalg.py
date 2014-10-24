import numpy as np
import scipy.sparse as sp
import logging

LOG = logging.getLogger(__name__)


def quick_diagonal_product(matrix, diagonal=None):
    if diagonal is None:
        diagonal = np.ones(matrix.shape[0])
    if type(matrix) is sp.csr_matrix:
        result = np.zeros(x.shape)
        for row_idx in xrange(matrix.shape[0]):
            row = matrix.getrow(row_idx)
            data = diagonal[row_idx] * (row.data ** 2)
            result[row.indices] += data
        return result
    elif type(matrix) is sp.csc_matrix:
        result = np.zeros(x.shape)
        for col_idx in xrange(matrix.shape[1]):   
            col = matrix.getcol(col_idx)
            result[col_idx] = np.dot(diagonal[col.indices], col.data ** 2)
        return result
    else:
        result = np.zeros(x.shape)
        for col_idx in xrange(matrix.shape[1]):   
            result[col_idx] = np.dot(diagonal, matrix[:, col_idx] ** 2)
        return result
 

