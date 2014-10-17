import numpy as np
import numpy.linalg as la
import logging

LOG = logging.getLogger(__name__)

def cgne_solve(A, b, P=None, x_0=None,
               max_iter=-1, abs_tol=1e-20, rel_tol=1e-20,
               verbose=False):

    if P is None:
        from jutil.operator import Identity
        P = Identity(A.shape[1])
    x = x_0 if x_0 is not None else np.zeros(A.shape[1])

    if max_iter < 0:
        max_iter = 2 * A.shape[1]

    p = A.T.dot(b)
    norm_ATb = la.norm(p);
    r = b - A.dot(x)
    t = A.T.dot(r)
    p = P.dot(t)
    s = p
    alpha = np.dot(s, s);

    i = 0
    while i <= max_iter:
        norm = la.norm(t);
        if (norm < abs_tol) or (norm / norm_ATb < rel_tol):
            break

        LOG.debug("CGNE it={}, reduced to {} {}".format(
            i, norm, norm / norm_ATb, norm_ATb))

        t = P.dot(p)
        q = A.dot(t)
        lambd = alpha / np.dot(q, q)
        assert not np.isnan(lambd)

        x += lambd * t
        i += 1
        r -= lambd * q

        t = A.T.dot(r)
        s = P.dot(t)
        new_alpha = np.dot(s, s)
        p *= new_alpha / alpha
        p += s

        alpha = new_alpha


    if verbose:
        norm = la.norm(t)
        LOG.info("CGNE needed {}{} iterations to reduce to {} {}".format(
            ("max=" if (i == max_iter) else ""), i, norm,
            norm / norm_ATb, norm_ATb))

    return x

