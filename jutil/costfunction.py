#
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Ungermann
#

import abc
import numpy as np
import jutil.norms
import jutil.linalg
import jutil.diff


class AbstractCostFunction(object):
    """
    Abstract base class for cost functions. Functions to be minimized
    by jutil must implement this interface.

    Methods
    -------
    __call__(x)
        evaluates the cost function at position x, returns a float
    jac(x)
        evaluates the jacobian of the cost function at position x, returns a vector
    hess_dot(x, vec)
        evaluates the product of the (approximated) hessian of cost function at
        position x with vector x, returns a vector
    hess_diag(x)
        returns the (approximated) diagonal of the hessian at position x

    Attributes
    ----------
    n : int
        dimensionality of state space
    chisq : float
        value of last cost function evaluation
    m : int, optional
        number of measurements if sensible
    chisq_m : float, optional
        agreement between measurements and model, if sensible
    chisq_a : float, optional
        agreement between state vector and apriori knowledge, if sensible

    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, x):
        pass

    @abc.abstractmethod
    def jac(self, x):
        pass

    @abc.abstractmethod
    def hess_dot(self, x):
        pass

    def hess_diag(self, x):
        return np.ones(self.n)


class LeastSquaresCostFunction(AbstractCostFunction):
    """
    A cost function wrapper around a simple function mapping R^n to R^m.

    A norm is used to map R^m to R.

    jac, hess, and hess_dot methods are supplied via finite differences
    if not given as optional parameters.

    Parameters
    ----------
    func : callable
        Function to be evaluated, expects an array_like, returns a float
    n : int
        dimensionality of state space
    norm : Norm, optional
        default is L2Square
    m : int, optional
        number of measurements if applicable
    jac : callable, optional
        function returning the jacobian of func, returns an array_like
    hess : callable, optional
        function returning the hessian of func, returns a 2-D array_like
    hess_dot : callable, optional
        function returning the product of hessian with a vector, returns an array_like
    epsilon : float, optional
    """
    def __init__(self, func, n, m=1, norm=jutil.norms.L2Square(),
                 jac=None, epsilon=1e-6):
        self._func, self._func_jac = func, jac
        self._norm = norm
        self._epsilon = epsilon
        self.m, self.n = m, n
        self._x, self.chisq = None, None

        if self._func_jac is None:
            self._func_jac = lambda x: jutil.diff.fd_jac(self._func, x, epsilon=self._epsilon)

    def init(self, x):
        self.__call__(x)
        self.update_jacobian(x)

    def update_jacobian(self, x):
        self._jac = self._func_jac(x)

    def _update(self, x):
        if self._x is None or np.any(self._x != x):
            self._y = self._func(x)
            self._x = x

    def __call__(self, x):
        self._update(x)
        self.chisq = self._norm(self._y)
        return self.chisq

    def jac(self, x):
        self._update(x)
        return self._jac.T.dot(self._norm.jac(self._y))

    def hess(self, x):
        self._update(x)
        return self._jac.T.dot(self._norm.hess(self._y).dot(self._jac))

    def hess_dot(self, x, vec):
        self._update(x)
        return self._jac.T.dot(self._norm.hess_dot(self._y, self._jac.dot(vec)))

#    def hess_diag(self, x):
#        self._update(x)
#        return jutil.linalg.quick_diagonal_product(self._jac, self._norm.hess_diag(self._y))


class WrapperCostFunction(AbstractCostFunction):
    """
    A cost function wrapper around a simple function mapping R^n to R.

    jac, hess, and hess_dot methods are supplied via finite differences
    if not given as optional parameters.

    Parameters
    ----------
    func : callable
        Function to be evaluated, expects an array_like, returns a float
    n : int
        dimensionality of state space
    m : int, optional
        number of measurements if applicable
    jac : callable, optional
        function returning the jacobian of func, returns an array_like
    hess : callable, optional
        function returning the hessian of func, returns a 2-D array_like
    hess_dot : callable, optional
        function returning the product of hessian with a vector, returns an array_like
    hess_diag : callable, optional
        function returning the diagonal of the Hessian.
    epsilon : float, optional
    """
    def __init__(self, func, n, m=1,
                 jac=None, hess=None, hess_dot=None, hess_diag=None,
                 epsilon=1e-6):
        self._func = func
        self._jac, self._hess, self._hess_dot = jac, hess, hess_dot

        self._epsilon = epsilon
        self.m, self.n = m, n
        self.chisq = None

        if self._jac is None:
            self._jac = lambda x: jutil.diff.fd_jac(self._func, x, epsilon=self._epsilon)
        if self._hess is None:
            self._hess = lambda x: jutil.diff.fd_jac(self._jac, x, epsilon=self._epsilon)
        if self._hess_dot is None:
            self._hess_dot = lambda x, vec: jutil.diff.fd_jac_dot(self._jac, x, vec, epsilon=self._epsilon)
        if hess_diag is not None:
            self.hess_diag = hess_diag

    def init(self, x):
        self.__call__(x)

    def __call__(self, x):
        self.chisq = self._func(x)
        return self.chisq

    def jac(self, x):
        return self._jac(x)

    def hess(self, x):
        return self._hess(x)

    def hess_dot(self, x, vec):
        return self._hess_dot(x, vec)


class ScaledCostFunction(AbstractCostFunction):
    def __init__(self, D, J):
        self._D, self._J = D, J
        assert len(D.shape) == 2
        assert D.shape[0] == D.shape[1]
        assert D.shape[1] == J.n
        assert D.shape[0] > 1
        self.n = J.n
        if hasattr(J, "m"):
            self.m = J.m
        if hasattr(J, "chisq_m"):
            self.chisq_m = property(J.chisq_m)
        if hasattr(J, "chisq_a"):
            self.chisq_a = property(J.chisq_a)

    def init(self, x):
        self._J.init(self._D.dot(x))

    def __call__(self, x):
        return self._J(self._D.dot(x))

    def jac(self, x):
        return self._J.jac(self._D.dot(x)).dot(self._D)

    def hess(self, x):
        return self._D.T.dot(self._J.hess(self._D.dot(x)).dot(self._D))

    def hess_dot(self, x, vec):
        return self._D.T.dot(self._J.hess_dot(self._D.dot(x), self._D.dot(vec)))

    def hess_diag(self, x):
        return jutil.linalg.quick_diagonal_product(self._D, self._J.hess_diag(x))

    @property
    def chisq(self):
        return self._J.chisq


class CountingCostFunction(AbstractCostFunction):
    """
    Wrapper around a given costfunction, enhanced by counting the number of
    method calls for statistical purposes.

    """
    def __init__(self, J):
        self._J = J
        self.m, self.n = J.m, J.n
        self.cnt_call, self.cnt_jac, self.cnt_hess_dot = 0, 0, 0
        self.cnt_hess, self.cnt_hess_diag = 0, 0

    def init(self, x):
        if hasattr(self._J, "init"):
            self._J.init(x)

    def __call__(self, x):
        self.cnt_call += 1
        return self._J(x)

    def jac(self, x):
        self.cnt_jac += 1
        return self._J.jac(x)

    def hess(self, x):
        self.cnt_hess += 1
        return self._J.hess(x)

    def hess_dot(self, x, vec):
        self.cnt_hess_dot += 1
        return self._J.hess_dot(x, vec)

    def hess_diag(self, x):
        self.cnt_hess_diag += 1
        return self._J.hess_diag(x)

    @property
    def chisq(self):
        return self._J.chisq

    @property
    def chisq_m(self):
        return self._J.chisq_m if hasattr(self._J, "chisq_m") else None

    @property
    def chisq_a(self):
        return self._J.chisq_a if hasattr(self._J, "chisq_a") else None
