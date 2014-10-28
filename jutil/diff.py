import scipy.sparse
import numpy as np


def fd_jac(fun, x, epsilon=1e-6):
    f0 = fun(x)
    return np.asarray([
        (fun(x + epsilon * np.eye(len(x), 1, -i).squeeze()) - f0)
         for i in xrange(len(x))]) / epsilon


def fd_jac_dot(fun, x, vec, epsilon=1e-6):
    f0 = fun(x)
    f1 = fun(x + epsilon * vec)
    return (f1 - f0) / epsilon


def fd_hess(fun, x, epsilon=1e-6):
    return fd_jac(lambda x: fd_jac(fun, x, epsilon), x, epsilon=1e-6)


def fd_hess_dot(fun, x, vec, epsilon=1e-6):
    return fd_jac_dot(lambda x: fd_jac(fun, x, epsilon), x, vec, epsilon=1e-6)


def get_diff_operator(mask, axis, factor=1):
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
    mask = mask.squeeze()
    n = mask.reshape(-1).sum()

    # Identify elements with valid neighbour
    mask1 = mask.copy().swapaxes(axis, -1)
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
