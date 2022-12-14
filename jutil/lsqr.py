#
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Ungermann
#

import numpy as np
import numpy.linalg as la
import logging

LOG = logging.getLogger(__name__)


def lsqr_solve(A, b, P=None, x_0=None,
               max_iter=-1, abs_tol=1e-20, rel_tol=1e-20, rel_change_tol=1e-20,
               verbose=False):
    r"""
    Preconditioned LSQR method to minimize ||Ax - b|| = ||r||.

    where A is a matrix with m rows and n columns, b is an m-vector.
    The matrix A is treated as a linear operator.  It is accessed
    by means of method calls with the following purpose:

    A.mult_ls(x, y) must compute y = A*x

    A.mult_ls_transpose(x, y) must compute y = A^T*x

    A.cond_ls(x, y) must compute y = M*x, where M is an approximate inverse
    of A

    A.cond_ls_inverse(x, y) must compute y = M^{-1}*x, where M is an
    approximate inverse of A

    Parameters
    ----------
    A : matrix
        the Matrix functor offering mult_ls, mult_ls_transpose, cond_ls
        and cond_ls_inverse functions.
    x : vector
        the vector containing the result. This is NOT an input
        parameter supplying the first guess!
    b : vector
        the r.h.s of the equation system.
    max_iter : int
        maximum allowed number of iterations. According to Paige
        and Saunders, this should be n/2 for well-conditioned systems and 4*n
        otherwise.
    abs_tol : float
        required absolute reduction of error, i.e. the iteration
        is terminated if ||A^T r|| < \mathrm{abs\_tol}.
    rel_tol : float
        required relative reduction of error, i.e. the iteration
        is terminated if ||A^T r|| < \mathrm{rel\_tol} ||A^T b||
    rel_change_tol : float
        Third stopping criteria, stopping if :math:`1 -
        (||r_{i+1}|| / ||r_i||) < \mathrm{rel\_change\_tol}`

    Notes
    -----
    The iteration stops if :math:`||A^T r|| < \mathrm{abs\_tol}`,
    :math:`||A^T r|| / ||A^T b|| < \mathrm{rel\_tol},
    1 - (||r_{k+1}|| / ||r_k||) < \mathrm{rel\_change\_tol}`,
    or the maximum number of iterations is reached.

    LSQR uses an iterative method to approximate the solution.
    The number of iterations required to reach a certain accuracy
    depends strongly on the scaling of the problem.  Poor scaling of
    the rows or columns of A should therefore be avoided where
    possible.

    Note that x is not an input parameter. If some initial estimate x_0 is
    known, one could proceed as follows:

    1. Compute a residual vector     :math:`r_0 = b - A x_0`.
    2. Use LSQR to solve the system  :math:`A dx = r_0`.
    3. Add the correction dx to obtain a final solution :math:`x = x_0 + dx`.

    References
    ----------
    .. [1] C.C. Paige and M.A. Saunders,  LSQR: An algorithm for sparse
         linear equations and sparse least squares,
         ACM Transactions on Mathematical Software 8, 1 (March 1982),
         pp. 43-71.

    .. [2] C.C. Paige and M.A. Saunders,  Algorithm 583, LSQR: Sparse
         linear equations and least-squares problems,
         ACM Transactions on Mathematical Software 8, 2 (June 1982),
         pp. 195-209.
    """

    if P is None:
        from jutil.operator import Identity
        P = Identity(A.shape[1])
    else:
        assert hasattr(P, "I"), "Preconditioner needs to support inverse!"

    if max_iter < 0:
        max_iter = 2 * A.shape[1]

    phi_bar_old = 1e-40

    norm_ATb = la.norm(A.T.dot(b))
    beta = la.norm(b)
    u = np.asarray(b) / beta
    p = A.T.dot(u)
    v = np.asarray(P.dot(p))
    alpha = la.norm(v)
    v /= alpha
    w = v.copy()
    x = np.zeros(A.shape[1])
    phi_bar = beta
    rho_bar = alpha

    i = 0
    while i < max_iter:
        # Continue bidiagonalization
        p = P.dot(v)
        u = -alpha * u + np.asarray(A.dot(p))
        beta = la.norm(u)
        if beta > 0:
            u /= beta
            p = A.T.dot(u)
            v = np.asarray(P.dot(p)) - beta * v
            alpha = la.norm(v)
            if alpha > 0:
                v /= alpha

        # Construct and apply next orthogonal transformation (Givens)
        scale = abs(rho_bar) + abs(beta)
        rho = scale * np.hypot(rho_bar / scale, beta / scale)
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar

        # Update x and w
        x += (phi / rho) * w
        i += 1
        w = v - (theta / rho) * w

        # Test for convergence, phi_bar = ||r||
        norm_ATr = la.norm(P.I.dot(-1.0 * phi_bar * alpha * c * v))
        rel_change = 100 * abs(1 - phi_bar / phi_bar_old)

        if ((norm_ATr < abs_tol)
                or (norm_ATr < rel_tol * norm_ATb)
                or (rel_change < rel_change_tol)):
            break
        LOG.debug("LSQR, it=%s iterations to reduce to %s %s %s %s",
                  i, phi_bar, norm_ATr, norm_ATr / norm_ATb, rel_change)

        phi_bar_old = phi_bar

    x = np.asarray(P.dot(x))
    LOG.info("LSQR needed %s%s iterations to reduce to %s %s %s %s",
             ("max=" if (i == max_iter) else ""), i, phi_bar,
             norm_ATr, norm_ATr / norm_ATb, rel_change)
    return x
