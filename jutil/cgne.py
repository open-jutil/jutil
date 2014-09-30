import numpy as np
import numpy.linalg as la

def cgne_solve(A, b,
               max_iter=-1, abs_tol=1e-20, rel_tol=1e-20,
               verbose=False):

    if hasattr(A, "cond"):
        A_cond = A.cond
    else:
        A_cond = lambda x: x.copy()

    if max_iter < 0:
        max_iter = 2 * A.shape[0]

    x = np.zeros(A.shape[1])
    p = A.T.dot(b)
    norm_ATb = la.norm(p);
    r = b - A.dot(x)
    t = A.T.dot(r)
    p = A_cond(t)
    s = p
    alpha = np.dot(s, s);

    i = 0
    while i <= max_iter:
        norm = la.norm(t);
        if (norm < abs_tol) or (norm / norm_ATb < rel_tol):
            break
        t = A_cond(p)
        q = A.dot(t)
        lambd = alpha / np.dot(q, q)
        x += lambd * t
        r -= lambd * q
        t = A.T.dot(r)
        s = A_cond(t)
        new_alpha = np.dot(s, s)
        p = (new_alpha / alpha) * p + s
        alpha = new_alpha
        i += 1

    if verbose:
        norm = la.norm(t)
        print "CGNE needed {}{} iterations to reduce to {} {}".format(
            ("max=" if (i == max_iter) else ""), i, norm,
            norm / norm_ATb, norm_ATb)

    return x

