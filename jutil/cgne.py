import numpy as np
import numpy.linalg as la

def cgne_solve(A, b,
                 max_iter, abs_tol, rel_tol):

    if hasattr(A, "cond"):
        A_cond = A.cond
    else:
        A_cond = lambda x: x.copy()
    if max_iter <0 :
        max_iter = A.shape[0]
    alpha = 0;
    new_alpha = 0;
    lambd = 0;
    norm = 0;

    x = np.zeros(A.shape[1])
    p = A.T.dot(b)
    norm_A_trans_b = la.norm(p);
    r = b - A.dot(x)
    t = A.T.dot(r)
    p = A_cond(t)
    s = p
    alpha = np.dot(s, s);

    i = 0
    while i <= max_iter:
        norm = la.norm(t);
        if (norm < abs_tol) or (norm/norm_A_trans_b < rel_tol):
            break
        t = A_cond(p)
        q = A.dot(t)
        lambd = alpha / np.dot(q, q);
        x += (lambd * t)
        r -= (lambd * q)
        t = A.T.dot(r)
        s = A_cond(t)
        new_alpha = np.dot(s, s)
        p = (new_alpha / alpha) * p + s;
        alpha = new_alpha
        i += 1
    return x
#      MESSAGE(DEBUG, "CGLSJacobi needed " << ((i == max_iter) ? "max=" : "") << i <<
#              " iterations to reduce to " << norm << " " << alpha/norm_A_trans_b);

