import numpy as np
import numpy.linalg as la


def split_bregman_2d(A, D, y, weight=100, it_max=300, mu=0.01, lambd=1, tol=1e-6, isotropic=True):
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
        chisq_a = (weight / mu) * sum(np.hypot(*np.split(D.dot(vector), 2))) / m
        chisq = chisq_m + chisq_a
        print "it= {it} / chi^2/m= {chisq} (meas= {chisqm} / apr= {chisqa} ) / {err}".format(
                it=it, chisq=chisq, chisqm=chisq_m,
                chisqa=chisq_a, err=error)

    it, error = 0, np.inf
    print_info(u)
    while error > tol and it <= it_max:
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


def split_bregman_2d_image(image, ig=None, weight=50, it_max=400, mu=5, lambd=1, tol=1e-6, isotropic=True):
    import scipy.sparse
    import jutil.cg as cg
    from jutil.lnsrch import lnsrch

    n = image.shape[0] * image.shape[1]
    n_root = image.shape[0]

    DiffOp = scipy.sparse.lil_matrix((2 * n, n))
    for i in range(n):
        if i % n_root != n_root - 1:
            DiffOp[i, i] = -1
            DiffOp[i, i + 1] = 1
        if i + n_root < n:
            DiffOp[n + i, i] = -1
            DiffOp[n + i, i + n_root] = 1
    D = DiffOp.tocsr()

    b = np.zeros(2 * n)
    d = b
    image = image.reshape(-1)
    if ig is not None:
        u = ig.reshape(-1)
        d = D.dot(u)
    else:
        u = image

    def print_info():
        if it % 5 != 0:
            return
        dy = u - image
        chisq_m = np.dot(dy, dy) / len(u)
        chisq_a = weight * sum(np.hypot(*np.split(D.dot(u), 2))) / len(u)
        chisq = chisq_m + chisq_a
        print "it= {it} / chi^2/m= {chisq} (meas= {chisqm} / apr= {chisqa} ) / {err}".format(
               it=it, chisq=chisq, chisqm=chisq_m,
               chisqa=chisq_a, err=error)

    it, error = 0, 0
    print_info()
    while True:
        u_last = u

        rhs = (mu / lambd) * image + D.T.dot(d - b)
        DTD = op.Dot(D.T, D)
        DT_dot_D_plus_I = cg.AT_dot_A_plus_lambda_I_CGWrapper(D, (mu / lambd))

        # single CG step
        rhs -= DT_dot_D_plus_I.dot(u)
        u = u + (np.dot(rhs, rhs) / np.dot(DT_dot_D_plus_I.dot(rhs), rhs)) * rhs
#        u = cg.conj_grad_solve(DT_dot_D_plus_I, rhs, 200, 1e-20, 1e-20)
        it += 1

        D_dot_u_plus_b = D.dot(u) + b
        if isotropic:
            s = np.hypot(*np.split(D_dot_u_plus_b, 2))
            # additional maximum to prevent division by zero in d assignment
            s_prime = np.maximum(s, 0.5 * weight / lambd)
            temp = np.maximum(s - weight / lambd, 0)
            d = (temp * D_dot_u_plus_b.reshape(2, -1) / s_prime).reshape(-1)

        else: # anisotropic
            temp = np.maximum(np.abs(D_dot_u_plus_b) - weight / lambd, 0)
            d = temp * np.sign(D_dot_u_plus_b)
        b = D_dot_u_plus_b - d

        error = la.norm(u_last - u) / la.norm(u)
        print_info()
        if error < tol or it > it_max:
            break
    return u.reshape(n_root, n_root)


def tv_denoise_2d(image, weight=50, eps=2e-4, keep_type=False):
    px = np.zeros_like(image)
    py = np.zeros_like(image)
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    d = np.zeros_like(image)
    i = 0
    while i < n_iter_max:
        print i
        d = -px -py
        d[1:] += px[:-1]
        d[:, 1:] += py[:, :-1]

        out = image + d
        E = (d**2).sum()
        gx[:-1] = np.diff(out, axis=0)
        gy[:, :-1] = np.diff(out, axis=1)
        norm = np.hypot(gx, gy)
        E += weight * norm.sum()
        norm *= 0.5 / weight
        norm += 1
        px -= 0.25 * gx
        px /= norm
        py -= 0.25 * gy
        py /= norm
        E /= float(image.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if np.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out


