import abc
import numpy as np
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
    @abc.abstractmethod
    def hess_diag(self, x):
        pass


class FiniteDifferencesCostFunction(AbstractCostFunction):
    """
    A cost function wrapper around a simple function mappinf R^n to R.

    jac, hess, and hess_dot methods are supplied via finite differences.

    Parameters
    ----------
    func : callable
        Function to be evaluated, expects an array_like, returns a float
    n : int
        dimensionality of state space
    m : int, optional
        number of measurements if applicable
    epsilon : float, optional
    """
    def __init__(self, func, n, m=1, epsilon=1e-6):
        self._func = func
        self._epsilon = epsilon
        self.m, self.n = m, n
        self.chisq = None

    def init(self, x):
        self.__call__(x)

    def __call__(self, x):
        self.chisq = self._func(x)
        return self.chisq

    def jac(self, x):
        return jutil.diff.fd_jac(self._func, x, epsilon=self._epsilon)

    def hess(self, x):
        return jutil.diff.fd_hess(self.jac, x, epsilon=self._epsilon)

    def hess_dot(self, x, vec):
        return jutil.diff.fd_hess_dot(self.jac, x, vec, epsilon=self._epsilon)

    def hess_diag(self, x):
        return np.ones(len(x))


class ScaledCostFunction(AbstractCostFunction):
    def __init__(self, D, J):
        self._D, self._J = D, J
        assert len(D.shape) == 2
        assert D.shape[0] == D.shape[1]
        assert D.shape[1] == J.n
        assert D.shape[0] > 1
        self.m, self.n = J.M, J.n
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
        return jutil.linalg.quick_diagonal_product(self.D, J.hess_diag(x))

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
        if hasattr(J, "chisq_m"):
            self.chisq_m = property(J.chisq_m)
        if hasattr(J, "chisq_a"):
            self.chisq_a = property(J.chisq_a)

    def init(self, x):
        self.__call__(x)

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
        return J.hess_diag(x)

    @property
    def chisq(self):
        return self._J.chisq

