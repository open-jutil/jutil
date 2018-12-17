#
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Ungermann
#

import numpy as np
import jutil.linalg


class _BaseNorm(object):
    """
    A class to provide some basic functionality based on other functions that shall
    be provided by derived classes.
    """

    def hess_dot(self, x, vec):
        """
        Provides the product of the Hessian matrix of this norm with a vector.

        Parameters
        ----------
        x : vector
             Determines the position where the Hessian matrix shall be evaluated at.
        vec : vector
            Vector to be multiplied with

        """
        return self.hess_diag(x) * vec

    def hess(self, x):
        """
        Provides the Hessian matrix of this norm.

        Parameters
        ----------
        x : vector
            Determines the position where the Hessian matrix shall be evaluated at.
        """
        return np.diag(self.hess_diag(x))


class Huber(_BaseNorm):
    def __init__(self, c):
        self._c = c

    def __call__(self, x):
        abs_x = np.abs(x)
        return np.sum(np.where(abs_x <= self._c, x ** 2, 2 * self._c * abs_x - self._c ** 2))

    def jac(self, x):
        return 2 * np.select([np.abs(x) <= self._c, x > 0], [x, self._c], default=-self._c)

    def hess_diag(self, x):
        return 2 * np.where(np.abs(x) <= self._c, np.ones_like(x), np.zeros_like(x))


class Ekblom(_BaseNorm):
    def __init__(self, eps, p):
        self._eps = eps
        self._p = p

    def __call__(self, x):
        return np.sum(pow(x ** 2 + self._eps, self._p / 2.))

    def jac(self, x):
        return self._p * pow(x ** 2 + self._eps, self._p / 2. - 1.) * x

    def hess_diag(self, x):
        return self._p * ((self._p - 2.) * pow(x ** 2 + self._eps, self._p / 2. - 2.) * (x ** 2)
                          + pow(x ** 2 + self._eps, self._p / 2. - 1.))

    def hess_dot(self, x, vec):
        return self.hess_diag(x) * vec


class BiSquared(_BaseNorm):
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
                        -6. * (1. - (x / self._k) ** 2) * ((2 * (x / (self._k ** 2))) ** 2)
                        + 3. * ((1. - ((x / self._k) ** 2)) ** 2) * 2 * (1 / (self._k ** 2)),
                        np.zeros_like(x))

    def hess_dot(self, x, vec):
        return self.hess_diag(x) * vec


class WeightedTVNorm(object):
    """
    Provides a TV norm if a proper weight matrix is supplied.
    It is assumed that the first elements of the weight-vector product represents the partial
    derivatives, whereas elements 0 to indices[0] - 1 are the partial derivative with respect
    to the first dimension, the elements indices[1] to indices[2] - 1 the partial derivative
    with respect to the second dimension and so on.
    """

    def __init__(self, basenorm, weight, indices):
        self._base = basenorm
        self._weight = weight
        self._indices = [0] + indices

    def _map(self, x, xp=None):
        if xp is None:
            xp = self._weight.dot(x)
        i0 = 0
        tv = np.zeros(self._indices[1])
        for i1 in self._indices[1:]:
            tv += xp[i0:i1] ** 2
            i0 = i1
        tv = np.sqrt(tv)
        return np.concatenate([tv, xp[self._indices[-1]:]])

    def _map_jac_dot(self, x, vec, xp=None):
        if xp is None:
            xp1 = self._weight.dot(x)
        else:
            xp1 = xp
        xp2 = self._map(x, xp=xp1)
        result = np.zeros(self._weight.shape[0] - self._indices[-2])
        Svec = self._weight.dot(vec)
        for i in np.where(xp2 > 0):
            inv = 1. / xp2[i]
            for dist in self._indices[:-1]:
                result[i] += inv * xp1[i + dist] * Svec[i + dist]

        result[self._indices[1]:] = Svec[self._indices[-1]:]
        return result

    def _map_jacT_dot(self, x, vec, xp=None):
        if xp is None:
            xp1 = self._weight.dot(x)
        else:
            xp1 = xp
        xp2 = self._map(x, xp=xp1)[:self._indices[1]]
        result = np.zeros(self._weight.shape[0])
        for i in np.where(xp2 > 0):
            inv = 1. / xp2[i]
            for dist in self._indices[:-1]:
                result[i + dist] += inv * xp1[i + dist] * vec[i]
        result[self._indices[-1]:] = vec[self._indices[1]:]
        return self._weight.T.dot(result)

    def _map_hess_dot(self, x, vec1, vec2, xp=None):
        if xp is None:
            xp = self._weight.dot(x)
        tvsum = np.zeros(self._indices[1])
        i0 = 0
        for i1 in self._indices[1:]:
            tvsum += xp[i0:i1] ** 2
            i0 = i1
        result = np.zeros(self._weight.shape[0])
        for i in np.where(tvsum > 0):
            base1 = tvsum[i] ** -1.5
            base2 = 1. / np.sqrt(tvsum[i])
            for ij, j in enumerate(self._indices[:-1]):
                for ik, k in enumerate(self._indices[:-1]):
                    if j == k:
                        fac = base2 - (xp[i + j] ** 2) * base1
                    else:
                        fac = -base1 * xp[i + j] * xp[i + k]
                    result[i + j] += vec1[i] * vec2[i + k] * fac
        return self._weight.T.dot(result)

    # F(x) = f(g(Sx))
    # ddF(x) = ST * dg(Sx)T * ddf(g(Sx)) * dg(Sx) * S
    #        + ST * ddg(Sx)T * df(g(Sx)) * S

    def __call__(self, x):
        result = self._base(self._map(x))
        return result

    def jac(self, x):
        temp1 = self._map(x)
        temp2 = self._base.jac(temp1)
        return self._map_jacT_dot(x, temp2)

    def hess(self, x):
        raise NotImplementedError

    def hess_dot(self, x, vec):
        Sx = self._weight.dot(x)
        gSx = self._map(x, xp=Sx)
        Svec = self._weight.dot(vec)
        dgSx_dot_Svec = self._map_jac_dot(x, vec, xp=Sx)
        ddfgSx_dot_dgSx_dot_Svec = self._base.hess_dot(gSx, dgSx_dot_Svec)
        sgSxT_dot_ddfgSx_dot_dgSx_dot_Svec = self._map_jacT_dot(x, ddfgSx_dot_dgSx_dot_Svec, xp=Sx)

        dfgSx = self._base.jac(gSx)
        ddgSxT_dot_dfgSx_dot_Svec = self._map_hess_dot(x, dfgSx, Svec, xp=Sx)

        temp = sgSxT_dot_ddfgSx_dot_dgSx_dot_Svec + ddgSxT_dot_dfgSx_dot_Svec
        return temp

    def hess_diag(self, x):
        raise NotImplementedError


class L2Square(_BaseNorm):
    r"""
    Norm is :math:`||x||_2^2 = \sum_i |x_i|^2`
    """

    def __call__(self, x):
        return np.dot(x, x)

    def jac(self, x):
        return 2 * x

    def hess_diag(self, x):
        return 2 * np.ones_like(x)


class LPPow(_BaseNorm):
    r"""
    Norm is :math:`||x||_p = (\sum_i |x_i|^p)`

    Norm is regularized to be twice-differentiable with
    :math:`|x_i| = \sqrt{x_i^2 + eps}`
    """

    def __init__(self, p, eps):
        self._p = p
        self._eps = eps

    def __call__(self, x):
        return np.sum(pow(x ** 2 + self._eps, self._p / 2.))

    def jac(self, x):
        return self._p * x * pow(x ** 2 + self._eps, self._p / 2. - 1)

    def hess_diag(self, x):
        return (self._p * pow(x ** 2 + self._eps, self._p / 2. - 2)
                * ((self._p - 1) * (x ** 2) + self._eps))


class L1(_BaseNorm):
    r"""
    Norm is :math:`||x||_1 = (\sum_i |x_i|)`
    """

    def __init__(self):
        self._p = 1

    def __call__(self, x):
        return np.sum(np.abs(x))

    def jac(self, x):
        return np.sign(x)

    def hess_diag(self, x):
        return np.zeros_like(x)


class LInf(_BaseNorm):
    r"""
    Norm is :math:`||x||_inf = (\max_i |x_i|)`
    """

    def __init__(self):
        self._p = np.inf

    def __call__(self, x):
        return np.max(np.abs(x))

    def jac(self, x):
        abs_x = np.abs(x)
        return np.where(abs_x == abs_x.max(), np.sign(x), np.zeros_like(x))

    def hess_diag(self, x):
        return np.zeros_like(x)


class WeightedNorm(object):
    """
    Norm is :math:`||A x||` with A being a rectangular matrix and
    :math:`||.||` a supplied _base norm
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
        return (self._weight.T.dot(temp2.T)).T

    def hess_dot(self, x, vec):
        w_dot_vec = self._weight.dot(vec)
        w_dot_x = self._weight.dot(x)
        return self._weight.T.dot(self._base.hess_dot(w_dot_x, w_dot_vec))

    def hess_diag(self, x):
        w_dot_x = self._weight.dot(x)
        base_hess_diagonal = self._base.hess_diag(w_dot_x)
        return jutil.linalg.quick_diagonal_product(self._weight, base_hess_diagonal)


class WeightedL2Square(object):
    """
    Norm is :math:`||A x||_2^2` with A being a rectangular matrix.

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
        result = 2 * self._weight.T.dot(self._weight)
        if hasattr(result, "todense"):
            result = result.todense()
        return result

    def hess_dot(self, _, vec):
        return 2 * self._weight.T.dot(self._weight.dot(vec))

    def hess_diag(self, x):
        return 2 * jutil.linalg.quick_diagonal_product(self._weight)
