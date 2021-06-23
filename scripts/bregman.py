from __future__ import print_function

# import matplotlib
# matplotlib.use('Agg')
import jutil.minimizer as minimizer
import numpy as np
import numpy.linalg as la
from jutil.taketime import TakeTime
import pylab


class CostFunction(object):
    def __init__(self):
        self._A = np.random.rand(5, 5)
        self._x_t = np.ones(5)
        self._lambda = 1e-4
        self._y = self._A.dot(self._x_t)
        self.m = len(self._y)
        self.n = len(self._x_t)

    def init(self, x_i):
        self.__call__(x_i)

    def __call__(self, x):
        dy = self._A.dot(x) - self._y
        self._chisqm = np.dot(dy, dy) / self.m
        self._chisqa = self._lambda * np.dot(x, x) / self.m
        self._chisq = self._chisqm + self._chisqa
        return self._chisq

    def jac(self, x):
        return (2. * self._A.T.dot(self._A.dot(x) - self._y) + 2. * self._lambda * x) / self.m

    def hess_dot(self, _, vec):
        return (2. * self._A.T.dot(self._A.dot(vec)) + 2. * self._lambda * vec) / self.m

    @property
    def chisq(self):
        return self._chisq

    @property
    def chisq_m(self):
        return self._chisqm

    @property
    def chisq_a(self):
        return self._chisqa


class CostFunctionImage(object):
    def __init__(self, p, lam):
        import jutil.norms as norms
        import scipy.sparse
        self._y = self._y_t + 0.5 * self._y_t.std() * np.random.randn(*self._y_t.shape)
        self._lambda = lam
        self.m = len(self._y)
        self.n = len(self._y)

        S = scipy.sparse.lil_matrix((2 * self.n, self.n))
        for i in range(self.n):
            if i % 256 != 255:
                S[i, i] = -1
                S[i, i + 1] = 1
            if i + 256 < self.n:
                S[self.n + i, i] = -1
                S[self.n + i, i + 256] = 1
        self._Sa_inv = S.tocsr()
        self._norm = norms.WeightedTVNorm(norms.LPPow(p, 1e-5), self._Sa_inv, [self.n, 2 * self.n])

    def init(self, x_i):
        self.__call__(x_i)

    def __call__(self, x):
        dy = x - self._y
        self._chisqm = np.dot(dy, dy) / self.m
        self._chisqa = self._lambda * self._norm(x) / self.m
        self._chisq = self._chisqm + self._chisqa
        return self._chisq

    def jac(self, x):
        return (2. * (x - self._y) + self._lambda * self._norm.jac(x)) / self.m

    def hess_dot(self, x, vec):
        return (2. * vec + self._lambda * self._norm.hess_dot(x, vec)) / self.m

    @property
    def chisq(self):
        return self._chisq

    @property
    def chisq_m(self):
        return self._chisqm

    @property
    def chisq_a(self):
        return self._chisqa


class CostFunctionLena(CostFunctionImage):
    def __init__(self, p, lam):
        import scipy.misc
        self._y_t = scipy.misc.lena()[384:128:-1, 128:384].reshape(-1)
        super(CostFunctionLena, self).__init__(p, lam)


class CostFunctionSquares(CostFunctionImage):
    def __init__(self, p, lam):
        self._y_t = 50 * np.ones((256, 256))
        self._y_t[:] += (np.arange(256) / 10.)[:, np.newaxis]
        self._y_t[:] += (np.arange(256) / 10.)[np.newaxis, :]
        self._y_t[100:200, 100:200] = 255.
        self._y_t[50:150, 50:150] = 200.
        self._y_t[55:145, 55:145] = 0.
        for i, j in [(i, j) for i in range(-25, 25) for j in range(-25, 25)]:
            if abs(i) + abs(j) < 25:
                self._y_t[50 + i, 50 + j] = 150
        for i, j in [(i, j) for i in range(-25, 25) for j in range(-25, 25)]:
            if np.hypot(i, j) < 25:
                self._y_t[200 + i, 50 + j] = 60
            if np.hypot(i, j) < 10:
                self._y_t[50 + i, 200 + j] = 200
        self._y_t = self._y_t.reshape(-1)
        super(CostFunctionSquares, self).__init__(p, lam)


def split_bregman_2d_test(image_t, image, ig=None, weight=100, max_iter=300, mu=0.01, lambd=1, tol=1e-6, isotropic=True):
    import scipy.sparse
    import jutil.cg as cg
    import itertools
    import jutil.norms as norms
    import jutil.operator as op
    from jutil.lnsrch import lnsrch
    import pylab

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

    A = scipy.sparse.lil_matrix((2 * 256 * 256 // 8, n))
    test = np.zeros((256, 256))
    i = 0
    for j in range(0, 256, 2):
        print(i)
        for k in range(0, 256, 4):
            delta = (k - j) / 256.
            cols = j + np.asarray(np.arange(256) * delta, dtype=int)
            assert min(cols) >= 0 and max(cols) <= 255, (k, J, delta, cols)
            rows = np.arange(256)
            A[i, rows * 256 + cols] = 1
            i += 1
            A[i, cols * 256 + rows] = 1
            i += 1

    A = A.tocsr()

    image = A.dot(image_t.reshape(-1))
    image = image + 0.01 * image.std() * np.random.randn(*image.shape)
    print(la.norm(image))

    ATA_DTD = op.Plus(op.Dot(A.T, A), op.Dot(D.T, D))

    x = cg.conj_grad_solve(ATA_DTD, A.T.dot(image), max_iter=300, abs_tol=1e-40, rel_tol=1e-40, verbose=True)
    print(la.norm(x - image_t))

    b = np.zeros(2 * n)
    d = b
    u = np.zeros_like(x)

    def printInfo(xx):
        if not (it % 5 == 0 or it == 1):
            return
        dy = A.dot(xx) - image
        chisq_m = np.dot(dy, dy) / len(image)
        chisq_a = weight * sum(np.hypot(*np.split(D.dot(xx), 2))) / len(image)
        chisq = chisq_m + chisq_a
        print("it= {it} / chi^2/m= {chisq} (meas= {chisqm} / apr= {chisqa} ) / {err}".format(
              it=it, chisq=chisq, chisqm=chisq_m,
              chisqa=chisq_a, err=error))

    it, error = 0, 0
    printInfo(x)
    printInfo(u)
    while True:
        u_last = u

        rhs = (mu / lambd) * A.T.dot(image) + D.T.dot(d - b)
        ATA_DTD = op.Plus(op.Dot(A.T, A, a=mu / lambd), op.Dot(D.T, D))

        # single CG step
        if it > 0:
            rhs -= ATA_DTD.dot(u)
            u = u + (np.dot(rhs, rhs) / np.dot(ATA_DTD.dot(rhs), rhs)) * rhs
        else:
            u = cg.conj_grad_solve(ATA_DTD, rhs, max_iter=300, abs_tol=1e-20, rel_tol=1e-20)
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
        printInfo(u)
        if error < tol or it > max_iter:
            break
    print(la.norm(u - image_t))
    pylab.subplot(1, 3, 1)
    pylab.pcolor(u.reshape(256, 256), vmin=0, vmax=256)
    pylab.subplot(1, 3, 2)
    pylab.pcolor(x.reshape(256, 256), vmin=0, vmax=256)
    pylab.subplot(1, 3, 3)
    pylab.pcolor(image_t.reshape(256, 256), vmin=0, vmax=256)
    pylab.savefig("test.png")

    return u.reshape(n_root, n_root)


def split_bregman_2d(image, ig=None, weight=50, max_iter=400, mu=5, lambd=1, tol=1e-6, isotropic=True):
    import scipy.sparse
    import cg
    import itertools
    import norms
    from lnsrch import lnsrch

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

    def printInfo():
        if it % 5 != 0:
            return
        dy = u - image
        chisq_m = np.dot(dy, dy) / len(u)
        chisq_a = weight * sum(np.hypot(*np.split(D.dot(u), 2))) / len(u)
        chisq = chisq_m + chisq_a
        print("it= {it} / chi^2/m= {chisq} (meas= {chisqm} / apr= {chisqa} ) / {err}".format(
              it=it, chisq=chisq, chisqm=chisq_m,
              chisqa=chisq_a, err=error))

    it, error = 0, 0
    printInfo()
    while True:
        u_last = u

        rhs = (mu / lambd) * image + D.T.dot(d - b)
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

        else:  # anisotropic
            temp = np.maximum(np.abs(D_dot_u_plus_b) - weight / lambd, 0)
            d = temp * np.sign(D_dot_u_plus_b)
        b = D_dot_u_plus_b - d

        error = la.norm(u_last - u) / la.norm(u)
        printInfo()
        if error < tol or it > max_iter:
            break
    return u.reshape(n_root, n_root)


def tv_denoise_2d(image, weight=50, eps=2.e-4, keep_type=False):
    px = np.zeros_like(image)
    py = np.zeros_like(image)
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    d = np.zeros_like(image)
    i = 0
    while i < n_max_iter:
        print(i)
        d = - px - py
        d[1:] += px[:-1]
        d[:, 1:] += py[:, :-1]

        out = image + d
        E = (d**2).sum()
        gx[:-1] = np.diff(out, axis=0)
        gy[:, :-1] = np.diff(out, axis=1)
        norm = np.sqrt(gx**2 + gy**2)
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


def dummy_test(Type, name, l1, l2):
    J11 = Type(1.01, 200)
    J2 = Type(2., l2)
    J2._y = J11._y
    with TakeTime("bregman"):
        den = split_bregman_2d_test(J2._y_t, J2._y.reshape(256, 256), weight=200).reshape(-1)
    solution11 = den
    solution2 = den
    img_map = pylab.cm.gray

    pylab.subplot(3, 3, 1)
    pylab.pcolor(J2._y_t.reshape(256, 256), vmin=0, vmax=255, cmap=img_map)
    pylab.xlim(0, 255); pylab.ylim(0, 255)
    pylab.subplot(3, 3, 2)
    pylab.pcolor(J2._y.reshape(256, 256), vmin=0, vmax=255, cmap=img_map)
    pylab.xlim(0, 255); pylab.ylim(0, 255)
    pylab.subplot(3, 3, 6)
    pylab.pcolor(den.reshape(256, 256), vmin=0, vmax=255, cmap=img_map)
    pylab.xlim(0, 255); pylab.ylim(0, 255)

    grad = J2._Sa_inv.dot(J2._y_t)
    tv = np.hypot(grad[:256*256], grad[256*256:])
    pylab.subplot(3, 3, 4)
    pylab.pcolor(solution2.reshape(256, 256), vmin=0, vmax=255, cmap=img_map)
    pylab.xlim(0, 255); pylab.ylim(0, 255)
    pylab.subplot(3, 3, 5)
    pylab.pcolor(solution11.reshape(256, 256), vmin=0, vmax=255, cmap=img_map)
    pylab.xlim(0, 255); pylab.ylim(0, 255)
    pylab.subplot(3, 3, 3)
    pylab.pcolor(tv.reshape(256, 256), vmax=255, cmap=pylab.cm.spectral)
    pylab.xlim(0, 255); pylab.ylim(0, 255)

    pylab.subplot(3, 3, 7)
    pylab.pcolor((solution2-J2._y_t).reshape(256, 256), vmin=-60, vmax=60, cmap=pylab.cm.RdBu)
    pylab.xlim(0, 255); pylab.ylim(0, 255)
    pylab.subplot(3, 3, 8)
    pylab.pcolor((solution11-J11._y_t).reshape(256, 256), vmin=-60, vmax=60, cmap=pylab.cm.RdBu)
    pylab.xlim(0, 255); pylab.ylim(0, 255)
    pylab.subplot(3, 3, 9)
    pylab.pcolor((den-J11._y_t).reshape(256, 256), vmin=-60, vmax=60, cmap=pylab.cm.RdBu)
    pylab.xlim(0, 255); pylab.ylim(0, 255)
    pylab.savefig(name +".png", dpi=300)

#dummy_test(CostFunctionLena, "lena", 60, 1)
dummy_test(CostFunctionSquares, "squa", 6000, 5)

