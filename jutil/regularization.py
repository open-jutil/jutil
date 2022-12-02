import numpy as np
import scipy.sparse as sp


def create_l0(xs, stds, corr):
    """ This function generates the L0 regularization
        in sparse representation.

    Parameters
    ----------
    xs : 1darray (N,)
        axis of altitude layers
    stds : 1darray (N,) or float
        standard deviations of altitude layers
    corr : 1darray (N-1,) or float
        correlation lengths of the spacing between two altitude layers

    Returns
    -------
    2darray
        sparse L0 regularization matrix

    Note
    ----
        This approximates the first integral
        in Eq. (7.311: Tarantola) by trapezoidal
        rule with flexible clorrelation length
        and standard deviation.

    Use as such
    -----------
        temp = L.dot(ys)
        norm = (temp.T.dot(temp)).sum()

    Ref
    ---
        Inverse Problem Theory and Methods for
        Model Parameter Estimation, Tarantola, A (2005),
        ISBN 978-0-89871-572-9 978-0-89871-792-1
    """
    dxs = np.diff(xs) / corr
    dx_int = np.zeros_like(xs)
    dx_int[:-1] += dxs
    dx_int[1:] += dxs
    return sp.diags(np.sqrt(dx_int / 2) / stds)


def create_l1(xs, stds, corr):
    """
    Creates a L1 regularization matrix to compute the L2 Norm of the
    first derivative of a discretized function.

    Parameters
    ----------
    xs : 1darray (N,)
        axis of altitude layers
    stds : 1darray (N,) or float
        standard deviations of altitude layers
    corr : 1darray (N-1,) or float
        correlation lengths of the spacing between two altitude layers

    Returns
    -------
    2darray
        sparse L1 regularization matrix

    Note
    ----
        This approximates the second integral in Eq.
        (7.311: Tarantola). The first derivative is
        approximated by forward finite difference.

    Use as such
    -----------
        temp = L.dot(ys)
        norm = (temp.T.dot(temp)).sum()

    Ref:
    ----
        Inverse Problem Theory and Methods for
        Model Parameter Estimation, Tarantola, A (2005),
        ISBN 978-0-89871-572-9 978-0-89871-792-1
    """

    dxs = np.diff(xs) / corr
    if np.asarray(stds).ndim == 0:
        stds = np.tile(stds, len(xs))
    if np.asarray(corr).ndim == 0:
        corr = np.tile(corr, len(xs))
    scale = np.sqrt(1 / dxs) / stds[:-1]
    return sp.diags([-scale, scale], offsets=[0, 1],
                    shape=(len(scale), len(xs)))


def create_l2(xs, stds, corr):
    """
    Creates a L2 regularization matrix to compute the L2 Norm of the
    second derivative of a discretized function.

    Parameters
    ----------
    xs : 1darray (N,)
        axis of altitude layers
    stds : 1darray (N,) or float
        standard deviations of altitude layers
    corr : 1darray (N-1,) or float
        correlation lengths of the spacing between two altitude layers

    Returns
    -------
    2darray
        sparse L2 regularization matrix

    Note
    ----
    -   This approximates the second integral in Eq.
        (7.311: Tarantola). The first derivative is
        approximated by forward finite difference.
    -   Caution: varying correlation lengths returns
        errors; best practice to only use L2 with constant
        correlation length;

    Use as such
    -----------
        temp = L.dot(ys)
        norm = (temp.T.dot(temp)).sum()
    """

    dxs = np.diff(xs) / corr
    dmxs = (dxs[1:] + dxs[:-1]) / 2
    if np.asarray(stds).ndim == 0:
        stds = np.tile(stds, len(xs))
    if np.asarray(corr).ndim == 0:
        corr = np.tile(corr, len(xs))

    # integration
    dx_int = (dxs[1:] + dxs[:-1])
    dx_int[0] += dxs[0]
    dx_int[-1] += dxs[-1]
    dx_int = np.sqrt(dx_int / 2) / stds[1:-1]

    # second order finite difference differences
    dif0 = dx_int / (dxs[:-1] * dmxs)
    dif2 = dx_int / (dxs[1:] * dmxs)
    dif1 = -(dif0 + dif2)

    return sp.diags([dif0, dif1, dif2],
                    offsets=[0, 1, 2], shape=(len(dx_int), len(xs)))


def generate_regblock(xs, scale0=1., scale1=0., scale2=0., stds=1, corr=1):
    """ This function generate the inverse covariance matrix

    Parameters
    ----------
    xs : 1darray
        axis of altitude layers
    scale0 : float, optional
        scale for L0 regularization matrix, by default 1.
    scale1 : float, optional
        scale for L1 regularization matrix, by default 0.
    scale2 : float, optional
        scale for L2 regularization matrix, by default 0.
    std : float or 1darray, optional
        standard deviation of altitude layers, by default 1
    float : float or 1darray, optional
        correlation length of altitude layers, by default 1

    Returns
    -------
    Ls : list of sparse 2darrays
        L-matrices
    """

    xs = xs.astype(float)

    n = len(xs)
    Ls = []
    if np.allclose(scale0, 0.):
        Ls.append(sp.coo_matrix((n, n)))
    else:
        Ls.append(scale0 * create_l0(xs, stds, corr))
    if np.allclose(scale1, 0.):
        Ls.append(sp.coo_matrix((n, n)))
    else:
        Ls.append(scale1 * create_l1(xs, stds, corr))
    if np.allclose(scale2, 0.):
        Ls.append(sp.coo_matrix((n, n)))
    else:
        Ls.append(scale2 * create_l2(xs, stds, corr))

    return Ls


def generate_regularization(axes, scales0=None, scales1=None, scales2=None, list_stds=None, list_corr=None):
    """ This function generates blockwise the L-regularisation matrices.

    Parameters
    ----------
    axes : list os 1darrays
        containing axes for the different blocks of state variable
    scales0 : list, optional
        scale for L0 regularization matrix for each block, by default None
    scales1 : list, optional
        scale for L1 regularization matrix for each block, by default None
    scales2 : list, optional
        scale for L2 regularization matrix for each block, by default None
    list_stds : list os floats or 1darrays
        standard deviations of state variable for each block; by default None
    list_corr : list os floats or 1darrays
        correlation lengths, by default None

    Returns
    -------
    Ls : list of sparse 2darrays
        L-matrices
    """

    if scales0 is None:
        scales0 = [1.] * len(axes)
    if scales1 is None:
        scales1 = [1.] * len(axes)
    if scales2 is None:
        scales2 = [0.] * len(axes)
    if list_stds is None:
        list_stds = [1] * len(axes)
    if list_corr is None:
        list_corr = [1] * len(axes)
    assert len(axes) == len(scales0) == len(scales1) == len(scales2) == len(list_stds) == len(list_corr)
    L0, L1, L2 = [], [], []
    for (axis, scale0, scale1, scale2, stds, corr) in zip(axes, scales0, scales1, scales2, list_stds, list_corr):
        tmp = generate_regblock(axis, scale0=scale0, scale1=scale1, scale2=scale2, stds=stds)
        L0.append(tmp[0])
        L1.append(tmp[1])
        L2.append(tmp[2])

    L0 = sp.block_diag(L0)
    L1 = sp.block_diag(L1)
    L2 = sp.block_diag(L2)
    return [L0, L1, L2]


def generate_inverse_covmatrix(Ls):
    """ This function generate the inverse covaraince matrix from the Ls matrices

    Parameters
    ----------
    Ls : list of sparse 2darrays
        L-matrices

    Returns
    -------
    Sa_inv : 2darray
        sparse matrix, inverse of the covariance matrix
        corresponding to the a priori state

    Note
    ----
    The factor 0.5 comes from  Eq. 7.311:
    Inverse Problem Theory and Methods for Model Parameter Estimation,
    Tarantola, A (2005), ISBN 978-0-89871-572-9 978-0-89871-792-1
    """
    Sa_inv = 0.5 * (Ls[0].T.dot(Ls[0]) + Ls[1].T.dot(Ls[1]) + Ls[2].T.dot(Ls[2]))

    return Sa_inv
