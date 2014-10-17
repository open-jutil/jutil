import numpy as np
import numpy.linalg as la
import logging

LOG = logging.getLogger(__name__)


def split_bregman_2d(A, D, y, weight=100, max_iter=300, mu=0.01, lambd=1, rel_change_tol=1e-6, isotropic=True):
    from jutil.cg import conj_grad_solve
    from jutil.operator import Plus, Dot

    m, n = A.shape
    assert m == len(y)
    assert D.shape[1] == A.shape[1]

    b = np.zeros(2 * n)
    d = b
    u = np.zeros(n)

    ATA_DTD = Plus(Dot(A.T, A, a=(mu / lambd)), Dot(D.T, D))

    def print_info(vector):
        if not (it % 5 == 0 or it == 1):
            return
        dy = A.dot(vector) - y
        chisq_m = np.dot(dy, dy) / m
        chisq_a = (lambd / mu) * sum(np.hypot(*np.split(D.dot(vector), 2))) / m
        chisq = chisq_m + chisq_a
        LOG.info("it= {it} / chi^2/m= {chisq} (meas= {chisqm} / apr= {chisqa} ) / {err}".format(
                 it=it, chisq=chisq, chisqm=chisq_m,
                 chisqa=chisq_a, err=error))

    it, error = 0, np.inf
    print_info(u)
    while error > rel_change_tol and it <= max_iter:
        u_last = u

        rhs = (mu / lambd) * A.T.dot(y) + D.T.dot(d - b)

        # single CG step
        if it > 0:
            rhs -= ATA_DTD.dot(u)
            u = u + (np.dot(rhs, rhs) / np.dot(ATA_DTD.dot(rhs), rhs)) * rhs
        else:
            u = conj_grad_solve(ATA_DTD, rhs, max_iter=100, abs_tol=1e-20, rel_tol=1e-20)
        it += 1

        D_dot_u_plus_b = D.dot(u) + b
        if isotropic:
            s = np.hypot(*np.split(D_dot_u_plus_b, 2))
            # additional maximum to prevent division by zero in d assignment
            s_prime = np.maximum(s, 0.5 * weight / lambd)
            temp = np.maximum(s - weight / lambd, 0)
            d = (temp * D_dot_u_plus_b.reshape(2, -1) / s_prime).reshape(-1)

        else:  # anisotropic
            temp = np.maximum(np.abs(D_dot_u_plus_b) - weight / lambd, 0)
            d = temp * np.sign(D_dot_u_plus_b)
        b = D_dot_u_plus_b - d

        error = la.norm(u_last - u) / la.norm(u)
        print_info(u)
    return u
