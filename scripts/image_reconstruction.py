import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import jutil
import jutil.norms as norms


class CostFunction(object):
    def __init__(self, A, D, y, lambd):
        self._A, self._D, self._y, self._lambda = A, D, y, lambd
        self.m, self.n = A.shape
        self._norm = norms.WeightedL2Square(self._D)

    def init(self, x_i):
        self.__call__(x_i)

    def __call__(self, x):
        dy = self._A.dot(x) - self._y
        self._chisqm = np.dot(dy, dy) / self.m
        self._chisqa = self._norm(x) / self.m
        self._chisq = self._chisqm + self._lambda * self._chisqa
        return self._chisq

    def jac(self, x):
        return 2. * (self._A.T.dot(self._A.dot(x) - self._y) +
                     self._lambda * self._norm.jac(x)) / self.m

    def hess_dot(self, x, vec):
        return 2. * (self._A.T.dot(self._A.dot(vec)) +
                     self._lambda * self._norm.hess_dot(x, vec)) / self.m

    def hess_diag(self, x):
        return np.ones_like(x)

    @property
    def chisq(self):
        return self._chisq

    @property
    def chisq_m(self):
        return self._chisqm

    @property
    def chisq_a(self):
        return self._chisqa


def get_tomography_operator(size, skip):
    import scipy.sparse
    A = scipy.sparse.lil_matrix((2 * (size ** 2) // (skip ** 2), size ** 2))
    i = 0
    for j in range(0, size, skip):
        for k in range(0, size, skip):
            delta = (k - j) / float(size)
            cols = j + np.asarray(list(map(int, np.arange(size) * delta)))
            rows = np.arange(size)
            A[i, rows * size + cols] = 1
            i += 1
            A[i, cols * size + rows] = 1
            i += 1
    return A.tocsr()


def execute(name, A, image, mu, lambd, noise, weight):
    import jutil.cg as cg
    import jutil.operator as op
    from jutil.diff import get_diff_operator
    import jutil.splitbregman as sb
    import os
    if os.path.exists(name + ".png"):
        return
    n_root = image.shape[0]

    D = op.VStack([get_diff_operator(np.ones((n_root, n_root), dtype=bool), axis)
                   for axis in [0, 1]])

    y_t = A.dot(image.reshape(-1))
    y = y_t + noise * y_t.std() * np.random.randn(*y_t.shape)
    J = CostFunction(A, D, y, lambd / mu)
    x_l2 = cg.conj_grad_minimize(J, max_iter=300, abs_tol=1e-40, rel_tol=1e-40, verbose=True)["x"]
    x_l1 = sb.split_bregman_2d(A, D, y, weight=weight, mu=mu, lambd=lambd, max_iter=2000)

    img_map = "plasma"
    plt.clf()
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.pcolormesh(image, vmin=0, vmax=255, cmap=img_map, rasterized=True)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 3, 2)
    plt.title("l1")
    plt.pcolormesh(x_l1.reshape(256, 256), vmin=0, vmax=255, cmap=img_map, rasterized=True)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 3, 3)
    plt.title("l2")
    plt.pcolormesh(x_l2.reshape(256, 256), vmin=0, vmax=255, cmap=img_map, rasterized=True)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.xticks([])
    plt.yticks([])
    if isinstance(A, jutil.operator.Identity):
        plt.subplot(2, 3, 4)
        plt.title("noisy")
        plt.pcolormesh(y.reshape(256, 256), vmin=0, vmax=255, cmap=img_map, rasterized=True)
        plt.xlim(0, 255)
        plt.ylim(0, 255)
        plt.xticks([])
        plt.yticks([])
    plt.subplot(2, 3, 5)
    plt.title("diff l1")
    plt.pcolormesh(x_l1.reshape(256, 256) - image,
                   vmin=-60, vmax=60, cmap=plt.cm.RdBu, rasterized=True)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 3, 6)
    plt.title("diff l1")
    plt.pcolormesh(x_l2.reshape(256, 256) - image,
                   vmin=-60, vmax=60, cmap=plt.cm.RdBu, rasterized=True)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(name + ".png", dpi=300)


def tomography():
    T = get_tomography_operator(256, 8)
    for mu in [1e-5, 1e-4, 1e-3, 1e-1]:
        for weight in [1]:
            execute("tlena256_{:010.8f}_{:010.4f}".format(mu, weight), T, jutil.misc.get_lena_256(), mu, 1, 0.01, weight)
            execute("tphantom1_{:010.8f}_{:010.4f}".format(mu, weight), T, jutil.misc.get_phantom_1(), mu, 1, 0.01, weight)


def denoise():
    I = jutil.operator.Identity(256 ** 2)
    for mu in [1e-3, 1e-2, 1e-1, 1]:
        for weight in [1]: # , 10, 100, 1000]:
            execute("nlena256_{:010.4f}_{:010.4f}".format(mu, weight), I, jutil.misc.get_lena_256(), mu, 1, 0.5, weight)
            execute("nphantom1_{:010.4f}_{:010.4f}".format(mu, weight), I, jutil.misc.get_phantom_1(), mu, 1, 0.5, weight)

jutil.misc.setup_logging()

# tomography example
tomography()

# denoising example
denoise()
