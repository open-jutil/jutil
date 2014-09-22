import numpy as np
import scipy.sparse as sp


class BaseNorm(object):
    def hess_dot(self, x, vec):
        return self.hess_diag(x) * vec

    def hess(self, x):
        return np.diag(self.hess_diag(x))


class Huber(BaseNorm):
    def __init__(self, c):
        self._c = c

    def __call__(self, x):
        abs_x = np.abs(x)
        return np.sum(np.where(abs_x <= self._c, x ** 2, 2 * self._c * abs_x - self._c ** 2))

    def jac(self, x):
        return 2 * np.select([np.abs(x) <= self._c, x > 0], [x, self._c], default=-self._c)

    def hess_diag(self, x):
        return 2 * np.where(np.abs(x) <= self._c, np.ones_like(x), np.zeros_like(x))


class Ekblom(BaseNorm):
    def __init__(self, eps, p):
        self._eps = eps
        self._p = p

    def __call__(self, x):
        return np.sum(pow(x ** 2 + self._eps, self._p / 2.))

    def jac(self, x):
        return self._p * pow(x ** 2 + self._eps, self._p / 2. - 1.) * x

    def hess_diag(self, x):
        return p * ((p - 2.) * pow(x ** 2 + eps, p / 2. - 2.) * (x ** 2) +
                    pow(x ** 2 + eps, p / 2. - 1.))

    def hess_dot(self, x, vec):
        return self.hess_diag(x) * vec


class WeightedTVNorm(object):
    def __init__(self, basenorm, weight, indices):
        self._base = basenorm
        self._weight = weight
        self._indices = indices


    def _map(self, x):
        xp = self._weight.dot(x)
        i0 = 0
        tv = np.zeros(self._indices[0])
        for i1 in self._indices:
            tv +=  xp[i0:i1] ** 2
            i0 = i1
        tv = np.sqrt(tv)
        return np.concatenate([tv, xp[self._indices[-1]:]])

    def _map_jacT_dot(self, x, vec):
        xp1 = self._weight.dot(x)
        xp2 = self._map(x)
        result = np.zeros(self._weight.shape[0])
        for i in range(self._indices[1]):
            if xp2[i] > 0:
                xp2[i] = 1. / xp2[i]
                for dist in self._indices:
                    print result.shape, xp2.shape, xp1.shape, vec.shape, i, dist
                    result[i + dist] = xp2[i] * xp1[i + dist] * vec[i]
        result[self._indices[1]:] = vec[self._indices[-1]:]
        return self._weight.T.dot(result)

    def _map_hess_dot(self, x, vec1, vec2):
        xp1 = self._weight.dot(x)
        i0 = 0
        tvsum = np.zeros(self._indices[0])
        for i1 in self._indices:
            tvsum +=  xp1[i0:i1] ** 2
            i0 = i1
        result = np.zeros(self._weight.shape[0])
        for i in range(self._indices[1]):
            if tvsum[i] > 0:
                base1 = tvsum ** -1.5
                base2 = 1. / np.sqrt(tvsum)
                for ij, j in enumerate(self._indices):
                    for ik, k in enumerate(self._indices):
                        if j == k:
                            fac = base2 - (xp1[i + j] ** 2) * base1
                        else:
                            fac = -base1 * xp1[i + j] * xp1[i + k]
                        result[i + j] += vec1[i] * vec2[i + k] * fac
        return self._weight.T.dot(result)


    # F(x) = f(g(Sx))
    # ddF(x) = ST * dg(Sx)T * ddf(g(Sx)) * dg(Sx) * S
    #        + ST * ddg(Sx)T * df(g(Sx)) * S

    def __call__(self, x):
        return self._base(self._map(x))

    def jac(self, x):
        temp1 = self._map(x)
        temp2 = self._base.jac(temp1)
        return self._map_jacT_dot(x, temp2)

    def hess_dot(self, x, vec):
        gSx = self._map(x)
        Svec = self._weight.dot(vec)
        dgSx_dot_Svec = self._map_jac_dot(x, Svec)
        ddfgSx_dot_dgSx_dot_Svec = self._norm.hess_dot(gSx, dgSx_dot_Svec)
        sgSxT_dot_ddfgSx_dot_dgSx_dot_Svec = self._map_jacT_dot(x, ddfgSx_dot_dgSx_dot_Svec)

        dfgSx = self._norm.jac(gSx)
        ddgSxT_dot_dfgSx_dot_Svec = self._map_hess_dot(x, dfgSx, Svec)

        temp = sgSxT_dot_ddfgSx_dot_dgSx_dot_Svec + ddgSxT_dot_dfgSx_dot_Svec
        return self._weight.T.dot(temp)



class BiSquared(BaseNorm):
    def __init__(self, k):
        self._k = k

    def __call__(self, x):
        return np.sum(np.where(np.abs(x) <= self._k, 1. - (1. - (x / self._k) ** 2) ** 3, np.ones_like(x)))

    def jac(self, x):
        return np.where(np.abs(x) <= self._k,
            3. * ((1. - (x / self._k) ** 2) ** 2) * 2 * (x / (self._k ** 2)),
            np.zeros_like(x))

    def hess_diag(self, x):
        return np.where(np.abs(x) <= self._k,
            - 6. * (1. - (x / self._k) ** 2) *  ((2 * (x / (self._k ** 2))) ** 2) +
            3. * ((1. - ((x / self._k) ** 2)) ** 2) * 2 * (1 / (self._k ** 2)),
            np.zeros_like(x))

    def hess_dot(self, x, vec):
        return self.hess_diag(x) * vec


class NormL2Square(object):
    """
    Norm is ||x||_2^2 = \sum_i |x_i|^2
    """

    def __call__(self, x):
        return np.dot(x, x)

    def jac(self, x):
        return 2 * x

    def hess(self, x):
        return np.diag(self.hess_diag(x))

    def hess_dot(self, _, vec):
        return 2 * vec

    def hess_diag(self, x):
        return 2 * np.ones_like(x)


class NormLPPow(BaseNorm):
    """
    Norm is ||x||_p = (\sum_i |x_i|^p)

    Norm is regularized to be twice-differentiable with
    |x_i| = sqrt(x_i^2 + eps)
    """

    def __init__(self, p, eps):
        self._p = p
        self._eps = eps

    def __call__(self, x):
        return np.sum(pow(x ** 2 + self._eps, self._p / 2.))

    def jac(self, x):
        return self._p * x * pow(x ** 2 + self._eps, self._p / 2. - 1)

    def hess_diag(self, x):
        return (self._p * pow(x ** 2 + self._eps, self._p / 2. - 2) *
                ((self._p - 1) * (x ** 2) + self._eps))


class WeightedNorm(object):
    """
    Norm is ||A x|| with A being a rectangular _weight matrix and
    ||.|| a supplied _base norm
    """

    def __init__(self, basenorm, weight):
        self._base = basenorm
        self._weight = weight

    def __call__(self, x):
        return self._base(self._weight.dot(x))

    def jac(self, x):
        temp1 = self._weight.dot(x)
        temp2 = self._base.jac(temp1)
        return self._weight.T.dot(temp2)

    def hess(self, x):
        temp1 = self._weight.dot(x)
        base_hess = self._base.hess(temp1)
        temp2 = self._weight.T.dot(base_hess)
        return temp2.dot(self._weight)

    def hess_dot(self, x, vec):
        w_dot_vec = self._weight * vec
        w_dot_x = self._weight * x
        temp = self._base.hess_dot(w_dot_x, w_dot_vec)
        return self._weight.T.dot(temp)

    def hess_diag(x):
        w_dot_x = self._weight.dot(x)
        base_hess_diagonal = self._base.hess_diag(w_dot_x)
        assert False
"""
    // weight^T hess weight
    la::DenseVector base_hess_diagonal = this->base_norm_->hess_diag(temp);
    result.clear();

    auto M_end = la::getCMajorIteratorEnd(this->weight_);
    for (auto M = la::getCMajorIteratorBegin(this->weight_); M != M_end; ++ M) {
      auto m_end = M.end();
      for (auto m = M.begin(); m != m_end; ++ m) {
        jassert(m.index2() < result.size(), m.index1() << " " << result.size());
        jassert(m.index1() < base_hess_diagonal.size(),
                m.index1() << " " << base_hess_diagonal.size());
        result(m.index2()) += base_hess_diagonal(m.index1()) * pow2(*m);
      }
    }
  }
"""

class WeightedL2Square(object):
    """
    Norm is ||A x||_2^2 with A being a rectangular \p weight_ matrix.

    Optimised for performance as the same effect could be gained by
    combination of WeightedNorm with L2Square norm.
    """

    def __init__(self, weight):
        self._weight = weight

    def __call__(self, x):
        w_dot_x = self._weight.dot(x)
        return np.dot(w_dot_x, w_dot_x)

    def jac(self, x):
        return 2 * self._weight.T.dot(self._weight.dot(x))

    def hess(self, _):
        return 2 * self._weight.T.dot(self._weight)

    def hess_dot(self, _, vec):
        return 2 * self._weight.T.dot(self._weight.dot(vec))

    def hess_diag(self, x):
        return np.diagonal(self.hess(x))



def _test():
    weight = np.arange(25).reshape(5, 5) / 25.
    x = np.arange(1, 6)[::-1]
    h = 1e-6
    for norm in [NormL2Square(),
                 WeightedL2Square(weight),
                 WeightedNorm(NormL2Square(), weight),
                 WeightedNorm(NormLPPow(1.1, 0), weight),
                 Huber(1.5),
                 Ekblom(1.5, 2.),
                 BiSquared(4.),
                 BiSquared(22.),
                 WeightedNorm(Huber(1.5), weight)]:
        fd_jac = np.asarray([(norm(x + h * np.eye(5)[:, i]) - norm(x)) / h for i in range(len(x))])
        print (np.ma.masked_invalid(abs(100 * fd_jac / norm.jac(x) - 100))).max()

def _test2():
    norm = WeightedTVNorm(NormL2Square(), np.diag(np.ones(4)), [1,2])
    print norm(np.ones(4))
    print norm.jac(np.ones(4))



_test2()
