import numpy as np
import numpy.linalg as la

def lsqr_solve(A, b, P=None, x_0=None,
               max_iter=-1, abs_tol=1e-20, rel_tol=1e-20, rel_change_tol=1e-20,
               verbose=False):
    """
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

    \param A the Matrix functor offering mult_ls, mult_ls_transpose, cond_ls
    and cond_ls_inverse functions.
    \param x the vector containing the result. This is NOT an input
    parameter supplying the first guess!
    \param b the r.h.s of the equation system.
    \param max_iter maximum allowed number of iterations. According to Paige
    and Saunders, this should be n/2 for well-conditioned systems and 4*n
    otherwise.
    \param abs_tol required absolute reduction of error, i.e. the iteration
    is terminated if ||A^T r||<abs_tol.
    \param abs_tol required relative reduction of error, i.e. the iteration
    is terminated if ||A^T r||<rel_tol ||A^T b||
    \param rel_change_tol Third stopping criteria, stopping if 1 -
    (||r_{i+1}|| / ||r_i||) < rel_change_tol

    The iteration stops if ||A^T * r|| < \p abs_tol, ||A^T * r|| / ||A^T
    * b|| < \p rel_tol, 1 - (||r_{k+1}|| / ||r_k||) < \p
    rel_change_tol, or the maximum number of iterations is reached.

    LSQR uses an iterative method to approximate the solution.
    The number of iterations required to reach a certain accuracy
    depends strongly on the scaling of the problem.  Poor scaling of
    the rows or columns of A should therefore be avoided where
    possible.

    Note that x is not an input parameter. If some initial estimate x_0 is
    known, one could proceed as follows:

    1. Compute a residual vector     r_0 = b - A*x_0.
    2. Use LSQR to solve the system  A*dx = r_0.
    3. Add the correction dx to obtain a final solution x = x_0 + dx.

    References
    C.C. Paige and M.A. Saunders,  LSQR: An algorithm for sparse
         linear equations and sparse least squares,
         ACM Transactions on Mathematical Software 8, 1 (March 1982),
         pp. 43-71.

    C.C. Paige and M.A. Saunders,  Algorithm 583, LSQR: Sparse
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
    u = b / beta
    p = A.T.dot(u)
    v = P.dot(p)
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
        u = -alpha * u + A.dot(p)
        beta = la.norm(u)
        if beta > 0:
            u /= beta
            p = A.T.dot(u)
            v = P.dot(p) - beta * v
            alpha = la.norm(v)
            if alpha > 0:
                v /= alpha;

        # Construct and apply next orthogonal transformation (Givens)
        scale = abs(rho_bar) + abs(beta)
        rho = scale * np.hypot(rho_bar / scale, beta / scale)
        c = rho_bar / rho;
        s = beta    / rho;
        theta   =  s * alpha
        rho_bar = -c * alpha
        phi     =  c * phi_bar
        phi_bar =  s * phi_bar

        # Update x and w
        x += (phi / rho) * w
        i += 1
        w = v - (theta / rho) * w

        # Test for convergence, phi_bar = ||r||
        p = P.I.dot(-1.0 * phi_bar * alpha * c * v)
        norm_ATr = la.norm(p)
        rel_change = 100 * abs(1 - phi_bar / phi_bar_old)

        if ((norm_ATr < abs_tol) or
            (norm_ATr < rel_tol * norm_ATb) or
            (rel_change < rel_change_tol)):
            break

        phi_bar_old = phi_bar

    x = P.dot(x)
    if verbose:
        print "LSQR needed {}{} iterations to reduce to {} {} {} {}".format(
            ("max=" if (i == max_iter) else ""), i, phi_bar,
            norm_ATr, norm_ATr / norm_ATb, rel_change)
    return x
