#
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Ungermann
#

import scipy.sparse
import numpy as np


def fd_jac(fun, x, epsilon=1e-6):
    """
    Computes Jacobian by finite differences.

    Parameters
    ----------
    fun : callable
    x : array_like
    epsilon : float, optional
        Delta for the finite difference computation. Default is 1e-6

    Returns
    -------
    Matrix containing the Jacobian
    """
    f0 = fun(x)
    return np.asarray([
        (fun(x + epsilon * np.eye(len(x), 1, -i).squeeze()) - f0)
        for i in xrange(len(x))]).T / epsilon


def fd_jac_dot(fun, x, vec, epsilon=1e-6):
    """
    Computes the product of Jacobian with a vector by finite differences.

    Parameters
    ----------
    fun : callable
    x : array_like
    vec : array_like
    epsilon : float, optional
        Delta for the finite difference computation. Default is 1e-6

    Returns
    -------
    Vector containing the Jacobian-vector product
    """
    f0 = fun(x)
    f1 = fun(x + epsilon * vec)
    return (f1 - f0) / epsilon


def fd_hess(fun, x, epsilon=1e-6):
    """
    Computes the Hessian by finite differences. Usually a very bad idea.
    Acceptable results can be achieved by finite-differencing an analytic
    Jacobian (that is apply fd_jac on the analytic Jacobian).
    """
    return fd_jac(lambda x: fd_jac(fun, x, epsilon), x, epsilon=1e-6)


def fd_hess_dot(fun, x, vec, epsilon=1e-6):
    """
    Computes a Hessian-vector product by finite differences. Usually a very bad idea.
    Acceptable results can be achieved by finite-differencing an analytic
    Jacobian (that is apply fd_jac_dot on the analytic Jacobian).
    """
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

    return scipy.sparse.coo_matrix((vals, (rows, cols)), (factor * n, factor * n)).tocsr()



def get_mass(shape):
    n = np.prod(shape)
    cols, rows, vals_m, vals_l = range(n), range(n), ([4. / 9.] * n), ([8. / 3.] * n)

    for axis in range(len(shape)):
        mask = np.ones(shape, dtype=bool)

        mask1 = mask.copy().swapaxes(axis, -1)
        mask1[:, 1:] = mask1[..., 1:] & mask1[..., :-1]
        mask1[:, 0] = False
        mask1 = mask1.swapaxes(axis, -1)

        mask2 = mask.copy().swapaxes(axis, -1)
        mask2[:, :-1] = mask2[..., 1:] & mask2[..., :-1]
        mask2[:, -1] = False
        mask2 = mask2.swapaxes(axis, -1)

        offset = mask.strides[axis] / mask.dtype.itemsize
        indice = np.where(mask.reshape(-1))[0]
        indmap = dict([(indice[i], i) for i in np.arange(len(indice))])

        val_indice1 = np.where(mask1.reshape(-1))[0]
        ms1 = [indmap[x] for x in val_indice1 - offset]
        iis1 = [indmap[x] for x in val_indice1]

        val_indice2 = np.where(mask2.reshape(-1))[0]
        ms2 = [indmap[x] for x in val_indice2 + offset]
        iis2 = [indmap[x] for x in val_indice2]

        cols += ms1 + ms2
        rows += iis1 + iis2
        vals_m += [1. / 9.] * (len(ms1) + len(ms2))
        vals_l += [-1. / 3.] * (len(ms1) + len(ms2))

        # ecken: 1. / 36., -1. / 3.

    M = scipy.sparse.coo_matrix((vals_m, (rows, cols)), (n, n)).tocsr()
    L = scipy.sparse.coo_matrix((vals_l, (rows, cols)), (n, n)).tocsr()
    return M, L

def f00(x,y):
    r1 = 1 - abs(x)
    r2 = 1 - abs(y)
    r1, r2 = [np.where(x > 0, x, np.zeros_like(x)) for x in [r1, r2]]
    return r1 * r2

def f00_jac(x, y):
    r1 = 1 - abs(x)
    r1_x = -np.sign(x)
    r2 = 1 - abs(y)
    r2_y = -np.sign(y)
    r1, r2 = [np.where(x > 0, x, np.zeros_like(x)) for x in [r1, r2]]
    r1_x = np.where(r1 > 0, r1_x, np.zeros_like(r1))
    r2_y = np.where(r2 > 0, r2_y, np.zeros_like(r2))
    return [r1_x * r2, r1 * r2_y]


def f11(x,y):
    r1 = 1 - abs(1 - x)
    r2 = 1 - abs(1 - y)
    r1, r2 = [np.where(x > 0, x, np.zeros_like(x)) for x in [r1, r2]]
    return r1 * r2

def f11_jac(x, y):
    r1 = 1 - abs(1 - x)
    r1_x = np.sign(1 - x)
    r2 = 1 - abs(1 - y)
    r2_y = np.sign(1 - y)
    r1, r2 = [np.where(x > 0, x, np.zeros_like(x)) for x in [r1, r2]]
    r1_x = np.where(r1 > 0, r1_x, np.zeros_like(r1))
    r2_y = np.where(r2 > 0, r2_y, np.zeros_like(r2))
    return [r1_x * r2, r1 * r2_y]


def f01(x,y):
    r1 = 1 - abs(x)
    r2 = 1 - abs(1 - y)
    r1, r2 = [np.where(x > 0, x, np.zeros_like(x)) for x in [r1, r2]]
    return r1 * r2

def f01_jac(x, y):
    r1 = 1 - abs(x)
    r1_x = -np.sign(x)
    r2 = 1 - abs(1 - y)
    r2_y = np.sign(1 - y)
    r1, r2 = [np.where(x > 0, x, np.zeros_like(x)) for x in [r1, r2]]
    r1_x = np.where(r1 > 0, r1_x, np.zeros_like(r1))
    r2_y = np.where(r2 > 0, r2_y, np.zeros_like(r2))
    return [r1_x * r2, r1 * r2_y]

"""
n = 2000
x, y = np.meshgrid(np.linspace(-3,3,n + 1), np.linspace(-3, 3, n + 1))
dx = 6. / n
a, b, h = 0.4, 0.4, 1e-8
print (f01(a + h, b) - f01(a, b)) / h, (f01(a, b + h) - f01(a, b)) / h
print f01_jac(a, b)
print
print np.trapz(np.trapz(f00(x, y) * f00(x, y), dx=dx), dx=dx)
print np.trapz(np.trapz(f00(x, y) * f01(x, y), dx=dx), dx=dx)
print np.trapz(np.trapz(f00(x, y) * f11(x, y), dx=dx), dx=dx)

aj, bj, cj = f00_jac(x, y), f01_jac(x, y), f11_jac(x, y)
print
print np.trapz(np.trapz(aj[0] * aj[0] + aj[1] * aj[1], dx=dx), dx=dx)
print np.trapz(np.trapz(aj[0] * bj[0] + aj[1] * bj[1], dx=dx), dx=dx)
print np.trapz(np.trapz(aj[0] * cj[0] + aj[1] * cj[1], dx=dx), dx=dx)
print np.trapz(np.trapz(bj[0] * cj[0] + bj[1] * cj[1], dx=dx), dx=dx)
print np.trapz(np.trapz(cj[0] * cj[0] + cj[1] * cj[1], dx=dx), dx=dx)

"""




