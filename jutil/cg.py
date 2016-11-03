#
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Ungermann
#

import numpy as np
import numpy.linalg as la
import logging
import jutil
from tqdm import tqdm

LOG = logging.getLogger(__name__)


def conj_grad_solve(A, b, P=None, x_0=None,
                    max_iter=-1, abs_tol=1e-20, rel_tol=1e-20,
                    verbose=False):
    """
    Simple implementation of preconditioned conjugate gradient method

    See A. Meister "Numerik linearer Gleichungssysteme", p. 218f

    Compared to the un-preconditioned CG method, this needs memory for an
    additional Vector and uses an additional scalar-product and one
    Matrix-Vector product per iteration.

    Parameters
    ----------
    A : functor
        A functor defining the matrix, realising the dot function
        for supplying a vector to be multiplied returning the result, a cond()
        function for supplying a vector and multiplying it with an approximate
        inversion and size1()/size2() functions returning its dimensions
    b : vector
        RHS of the lineqr equation system
    x_0 : vector
        gives the initial guess
    max_iter : int
        determines the maximum number of iterations.
    abs_tol : float
        determines the tolerance for the remaining
        residuum. The algorithm will terminate if
        :math:`||Ax - b||_2 / ||b||_2 < \mathrm{abs\_tol}`.
    rel_tol : float
        determines the expected residual reduction. The
        algorithm will terminate when :math:`||Ax_0 - b||_2 / ||Ax - b||_2 < \mathrm{rel_tol}`.
        By specifying red_tol to be, say 1e-2, one requires the residuum to
        be reduced to one hundreth of its initial value.

    """

    single_result = not hasattr(rel_tol, "__len__")

    rel_tol = np.array(rel_tol, copy=True).reshape(-1)
    assert len(rel_tol) > 0
    assert np.all(np.diff(rel_tol) > 0)
    assert A.shape[1] == len(b), (A.shape, len(b))

    if P is None:
        from jutil.operator import Identity
        P = Identity(A.shape[1])
    assert P.shape == A.shape

    x = np.array(x_0, copy=True) if x_0 is not None else np.zeros_like(b)
    assert len(x) == len(b)

    if max_iter < 1:
        max_iter = 2 * A.shape[1]

    r = b - A.dot(x)
    p = P.dot(r)
    alpha = np.dot(r, p)
    norm_b = la.norm(b)
    xs = [0 for _ in range(len(rel_tol))]

    i = 0
    with tqdm(total=max_iter) as pbar:
        while i < max_iter:
            norm = la.norm(r)
            norm_div_norm_b = norm / norm_b
            for j in range(len(rel_tol)):
                if norm_div_norm_b < rel_tol[j]:
                    xs[j] = x.copy()
                    if j > 0:
                        rel_tol[j] = -1
            pbar.update()
            if (norm <= abs_tol) or (norm_div_norm_b < rel_tol[0]):
                break
            LOG.debug("CG, it={}, reduced to {} {}".format(
                i, norm, norm / norm_b, norm_b))

            v = A.dot(p)
            pAp = np.dot(v, p)
            if pAp <= 0:  # negative curvature
                LOG.warn("CG encountered negative curvature. Is matrix really s.p.d.?")
                break
            lambd = alpha / pAp
            assert not np.isnan(lambd)

            x += lambd * p
            i += 1
            r -= lambd * v
            z = P.dot(r)
            new_alpha = np.dot(r, z)
            p *= new_alpha / alpha
            p += z

            alpha = new_alpha

    LOG.info("CG needed {}{} iterations to reduce to {} {}".format(
        ("max=" if (i == max_iter) else ""), i, la.norm(r),
        la.norm(r) / norm_b))

    for j in [_j for _j in range(len(rel_tol)) if rel_tol[_j] != -1]:
        xs[j] = x.copy()

    if single_result:
        assert len(xs) == 1
        xs = xs[0]

    return xs


def conj_grad_tall_solve(A, bs, P=None, x_0=None,
                         max_iter=-1, abs_tol=1e-20, rel_tol=1e-20,
                         verbose=False):
    """
    see conj_grad_solve.
    """

    if P is None:
        from jutil.operator import Identity
        P = Identity(A.shape[0])

    xs = x_0 if x_0 is not None else np.zeros_like(bs)

    if max_iter < 1:
        max_iter = A.shape[0]

    rs = bs - A.dot(xs)
    ps = P.dot(rs)
    alphas = np.einsum('ij,ij->j', rs, ps)
    norms_b = np.asarray([la.norm(b) for b in bs.T])

    i = 0
    while i <= max_iter:
        norms = np.asarray([la.norm(r) for r in rs.T])
        norm_div_norm_b = norms / norms_b

        if np.all((norms < abs_tol) | (norm_div_norm_b < rel_tol)):
            break

        LOG.debug("CG, it={}, reduced to {} {}".format(
            i, la.norm(rs),
            np.asarray([la.norm(r) for r in rs.T]) / norms_b, norms_b))

        vs = A.dot(ps)
        vs_dot_ps = np.einsum('ij,ij->j', vs, ps)
        lambds = np.asarray([alphas[j] / vs_dot_ps[j] if norms[j] != 0 else 0 for j in range(len(alphas))])

        xs += lambds * ps
        rs -= lambds * vs
        zs = A.dot(rs)

        new_alphas = np.einsum('ij,ij->j', rs, zs)
        ps *= new_alphas / alphas
        ps += zs

        alphas = new_alphas
        alphas[alphas == 0] = 1

    LOG.info("CG needed {}{}  iterations to reduce to {} {}".format(
        ("max=" if (i == max_iter) else ""), i, la.norm(rs),
        np.asarray([la.norm(r) for r in rs.T]) / norms_b, norms_b))

    return xs


def conj_grad_minimize(J, x_0=None,
                       max_iter=-1, abs_tol=1e-20, rel_tol=1e-20,
                       verbose=False):
    if x_0 is None:
        x_0 = np.zeros(J.n)
    J = jutil.costfunction.CountingCostFunction(J)
    A = jutil.operator.CostFunctionOperator(J, x_0)
    P = jutil.preconditioner.CostFunctionPreconditioner(J, x_0)

    b = -J.jac(x_0)
    x_f = x_0 + conj_grad_solve(
        A, b, P=P, max_iter=max_iter, abs_tol=abs_tol, rel_tol=rel_tol, verbose=verbose)

    result = jutil.minimizer.OptimizeResult({
        "x": x_f,
        "success": True,
        "fun": None,
        "jac": None,
        "nit": None,
        "nfev": J.cnt_call,
        "njev": J.cnt_jac,
        "nhev": J.cnt_hess,
        "nhdev": J.cnt_hess_dot,
        "nhdiagev": J.cnt_hess_diag,
    })
    return result
