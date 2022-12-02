import itertools
import numpy as np
import scipy.sparse as sp


def _compute_integration_weights(dxs):
    """ Helper function to compute weights from the trapezoid rule from
        interval length.

    Parameters
    ----------
    dxs : 1darray (N,)
        lengths of intervals

    Returns
    -------
    1darray
        weights

    """
    dx_int = np.zeros(len(dxs) + 1)
    dx_int[:-1] += dxs
    dx_int[1:] += dxs
    return dx_int / 2


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
    dx_int = _compute_integration_weights(dxs)
    return sp.diags(np.sqrt(dx_int) / stds)


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
    if isinstance(stds, np.ndarray) and stds.ndim == 1:
        stds = stds[:-1]
    scale = np.sqrt(1 / dxs) / stds
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
    if isinstance(stds, np.ndarray) and stds.ndim == 1:
        stds = stds[1:-1]

    # integration
    dx_int = (dxs[1:] + dxs[:-1])
    dx_int[0] += dxs[0]
    dx_int[-1] += dxs[-1]
    dx_int = np.sqrt(dx_int / 2) / stds

    # second order finite difference differences
    dif0 = dx_int / (dxs[:-1] * dmxs)
    dif2 = dx_int / (dxs[1:] * dmxs)
    dif1 = -(dif0 + dif2)

    return sp.diags([dif0, dif1, dif2],
                    offsets=[0, 1, 2], shape=(len(dx_int), len(xs)))


def create2d_l0(xs, ys, stds, corr_x, corr_y):
    """ This function generates the L0 regularization
        in sparse representation.

    Parameters
    ----------
    xs : 1darray (N,)
        axis of altitude layers
    ys : 1darray (M,)
        secondary axis
    stds : 2darray (M, N) or float
        standard deviations
    corr_x : 1darray (N-1,) or float
        correlation lengths of the spacing between two altitude layers
    corr_y : 1darray (M-1,) or float
        correlation lengths of the spacing on secondary axis

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

    Note
    ----
        The 2d integral is handled by mapping the 2-D state to
        a 1-D vector.
        It is then handled as in the 1d case by
        approximating the first integral
        in Eq. (7.311: Tarantola) by trapezoidal
        rule with flexible correlation length
        and standard deviation.

    """
    dxs = np.diff(xs) / corr_x
    dx_int = _compute_integration_weights(dxs)
    dys = np.diff(ys) / corr_y
    dy_int = _compute_integration_weights(dys)
    wgts = dx_int[np.newaxis, :] * dy_int[:, np.newaxis]
    return sp.diags((np.sqrt(wgts) / stds).reshape(-1))


def create2d_l1x(xs, ys, stds, corr_x, corr_y):
    """
    Creates a L1 regularization matrix to compute the L2 Norm of the
    first partial derivative in x-direction of a discretized function.

    Parameters
    ----------
    xs : 1darray (N,)
        axis of altitude layers
    ys : 1darray (M,)
        secondary axis
    stds : 2darray (M, N) or float
        standard deviations
    corr_x : 1darray (N-1,) or float
        correlation lengths of the spacing between two altitude layers
    corr_y : 1darray (M-1,) or float
        correlation lengths of the spacing on secondary axis

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


    Note
    ----
        The 2d integral is handled by mapping the 2-D state to
        a 1-D vector.
        It is then handled as in the 1d case by
        approximating the first integral
        in Eq. (7.311: Tarantola) by trapezoidal
        rule with flexible correlation length
        and standard deviation.

    """
    assert len(xs) == stds.shape[1] and len(ys) == stds.shape[0]
    stds = stds.reshape(-1)
    dxs_sqrt = np.sqrt(np.diff(xs) / corr_x)
    dys = np.diff(ys) / corr_y
    dy_int_sqrt = np.sqrt(_compute_integration_weights(dys))

    scale = np.zeros_like(stds)
    for i in range(len(ys)):
        sl = slice(i * len(xs), (i + 1) * len(xs) - 1)
        scale[sl] = dy_int_sqrt[i] / (dxs_sqrt * stds[sl])
    return sp.diags([-scale, scale], offsets=[0, 1],
                    shape=(len(stds), len(stds)))


def create2d_l1y(xs, ys, stds, corr_x, corr_y):
    """
    Creates a L1 regularization matrix to compute the L2 Norm of the
    first partial derivative in y-direction of a discretized function.

    Parameters
    ----------
    xs : 1darray (N,)
        axis of altitude layers
    ys : 1darray (M,)
        secondary axis
    stds : 2darray (M, N) or float
        standard deviations
    corr_x : 1darray (N-1,) or float
        correlation lengths of the spacing between two altitude layers
    corr_y : 1darray (M-1,) or float
        correlation lengths of the spacing on secondary axis

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

    Note
    ----
        The 2d integral is handled by mapping the 2-D state to
        a 1-D vector.
        It is then handled as in the 1d case by
        approximating the first integral
        in Eq. (7.311: Tarantola) by trapezoidal
        rule with flexible correlation length
        and standard deviation.

    """
    assert len(xs) == stds.shape[1] and len(ys) == stds.shape[0]
    stds = stds.reshape(-1)
    dxs = np.diff(xs) / corr_x
    dys_sqrt = np.sqrt(np.diff(ys) / corr_y)
    dx_int_sqrt = np.sqrt(_compute_integration_weights(dxs))
    scale = np.zeros_like(stds)
    for i in range(len(ys) - 1):
        sl = slice(i * len(xs), (i + 1) * len(xs))
        scale[sl] = dx_int_sqrt / (dys_sqrt[i] * stds[sl])
    return sp.diags([-scale, scale], offsets=[0, len(xs)],
                    shape=(len(stds), len(stds)))


def generate_regblock(xs, scale0=1, scale1=0, scale2=0, stds=1, corr=1):
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

    Ls = [
        scale * create(xs, stds, corr)
        for create, scale in zip(
            (create_l0, create_l1, create_l2),
            (scale0, scale1, scale2))
        if not np.allclose(scale, 0.)]

    return sp.vstack(Ls)


def generate_regularization(
        axes,
        scales0=itertools.repeat(1),
        scales1=itertools.repeat(1),
        scales2=itertools.repeat(0),
        list_stds=itertools.repeat(1),
        list_corr=itertools.repeat(1)):
    """ This function generates blockwise the L-regularisation matrices.

    Parameters
    ----------
    axes : list os 1darrays
        containing axes for the different blocks of state variable
    scales0 : list, optional
        scale for L0 regularization matrix for each block, by default 1
    scales1 : list, optional
        scale for L1 regularization matrix for each block, by default 1
    scales2 : list, optional
        scale for L2 regularization matrix for each block, by default 0
    list_stds : list of floats or 1darrays, optional
        standard deviations of state variable for each block; by default 1
    list_corr : list of floats or 1darrays, optional
        correlation lengths, by default 1

    Returns
    -------
    Ls : list of sparse 2darrays
        L-matrices
    """
    Ls = [
        generate_regblock(axis, scale0=scale0, scale1=scale1, scale2=scale2,
                          stds=stds, corr=corr)
        for axis, scale0, scale1, scale2, stds, corr
        in zip(axes, scales0, scales1, scales2, list_stds, list_corr)]

    return sp.block_diag(Ls)


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
    return 0.5 * Ls.T.dot(Ls)
