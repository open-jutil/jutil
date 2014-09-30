import scipy.sparse
import numpy as np


def get_diff_op(mask, axis, factor=1):
    """
    Returns a difference operator for a given mask indicating which elements are "active".

    Parameters
    ----------
    mask : n-dimensional array indicating if an element shall be part of the difference
    axis : Indicates along which axis the difference shall be computed
    factor : Indicates that another, unmentioned dimension exists and that the true
        vector is that much longer. Not recommended.
    Returns
    -------
    Sparse Matrix containing the difference operator
    """

    n = len(mask.reshape(-1))

    # Identify elements with valid neighbour
    mask1 = mask.copy().squeeze().swapaxes(axis, -1)
    mask1[:, 1:] = mask1[..., 1:] & mask1[..., :-1]
    mask1[:, 0] = False
    mask1 = mask1.swapaxes(axis, -1)

    # shift for left neighbour
    offset = mask.strides[axis] / mask.dtype.itemsize
    indice = np.where(mask.reshape(-1))[0]
    indmap = dict([(indice[i], i) for i in np.arange(len(indice))])

    ks = range(0, factor * n, n)

    val_indice = np.where(mask1.reshape(-1))[0]
    p1s = np.asarray([indmap[x] for x in val_indice])
    m1s = np.asarray([indmap[x] for x in val_indice - offset])
    iis = np.arange(len(m1s), dtype=int)

    cols = np.concatenate([m1s + k for k in ks] + [p1s + k for k in ks])
    rows = np.concatenate([iis + k for k in ks] * 2)
    vals = np.concatenate([-np.ones(factor * len(m1s)), np.ones(factor * len(m1s))])

    return scipy.sparse.coo_matrix((vals, (rows, cols)), (factor * n,factor * n)).tocsr()


def get_diff_op_old(mask):
    n = len(mask.reshape(-1)) * 3
    count = n / 3

    DiffOp = scipy.sparse.lil_matrix((2 * n, n))
    mask1 = mask.copy().squeeze()
    mask1[1:, :] = mask1[1:, :] & mask1[:-1,:]
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


mask = np.zeros((50,50), dtype=bool)
for i in range(1, 40):
    for j in range(1, 40):
        mask[i, j] = True
def test():
    import jutil.operator as op
    #import timeit
    #print timeit.timeit("get_diff_op(mask, 0, factor=3)", setup="from __main__ import *", number=10)

    A = op.VStack([get_diff_op(mask, i, factor=3) for i in [0, 1]])
    B = get_diff_op_old(mask)
    x = np.random.rand(3 * 50 * 50)
    print np.linalg.norm(A.dot(x) - B.dot(x))

test()
