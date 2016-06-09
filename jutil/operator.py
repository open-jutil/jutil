#
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Ungermann
#

import numpy as np


class CostFunctionOperator(object):
    """
    Wraps the hess_dot of a CostFunction in combination with a Tikhonov-Levenberg-Marquardt
    regularisation. Used to solve the LES posed by steps in non-linear minimization.
    May also wrap a CostFunction to provide a linear solution.

    Parameters
    ----------
    J : CostFunction
    x : vector
    lmpar : float, optional
        Levenberg-Marquardt Regularisation parameter. In effect lmpar times an Identity
        matrix is added to the CostFunction. (the default is 0, i.e. no effect).

    Attributes
    ----------
    shape : (int, int)

    Methods
    -------
    dot(vector)
        Provides multiplication with a vector
    """
    def __init__(self, J, x, lmpar=0):
        self._J = J
        self._x = x
        self._lmpar = lmpar
        self.dot = self._dot if lmpar == 0 else self._dot_lam

    def _dot(self, vec):
        return self._J.hess_dot(self._x, vec)

    def _dot_lam(self, vec):
        return self._J.hess_dot(self._x, vec) + self._lmpar * vec

    @property
    def shape(self):
        return (self._J.n, self._J.n)


class Identity(object):
    """
    An Identity operator.

    Parameters
    ----------
    n : int
        dimensionality of the operator.

    Attributes
    ----------
    shape : int, int
        provides the shape of the operator
    T : Identity
        provides the transposed operator
    I : Identity
        provides the inverted operator

    Methods
    -------
    dot(vector)
        Provides multiplication with a vector
    """
    def __init__(self, n):
        self._shape = (n, n)
        self.T = self
        self.I = self

    def dot(self, x):
        assert len(x) == self.shape[1]
        return np.array(x, copy=True)

    @property
    def shape(self):
        return self._shape


class Scale(object):
    """
    An scaling Operator.

    Parameters
    ----------
    A : Operator
    a : float
    adjoint : Operator, optional

    Attributes
    ----------
    shape : int, int
        provides the shape of the operator
    T : Scale
        provides the transposed operator
    I : Scale
        provides the inverted operator

    Methods
    -------
    dot(vector)
        Provides multiplication with a vector
    """
    def __init__(self, A, a, adjoint=None):
        self._A, self._a = A, a
        if adjoint is None:
            if hasattr(A, "T"):
                self.T = Scale(A.T, a, adjoint=self)
        else:
            self.T = adjoint
        self.I = self.T

    def dot(self, x):
        return self._a * self._A.dot(x)

    @property
    def shape(self):
        return self._A.shape


class Dot(object):
    """
    Produces the product of two operators (and an optional scaling).

    Parameters
    ----------
    A : Operator
    B : Operator
    a : float, optional
    adjoint : Operator, optional


    Attributes
    ----------
    shape : int, int
        provides the shape of the operator
    T : Dot
        provides the transposed operator

    Methods
    -------
    dot(vector)
        Provides multiplication with a vector
    """
    def __init__(self, A, B, a=1, adjoint=None):
        self._A, self._B, self._a = A, B, a
        self.dot = self._dot if a == 1 else self._dot_a
        assert self._A.shape[1] == self._B.shape[0]
        if adjoint is None:
            if hasattr(B, "T") and hasattr(A, "T"):
                self.T = Dot(B.T, A.T, a=a, adjoint=self)
        else:
            self.T = adjoint

    def _dot(self, x):
        return self._A.dot(self._B.dot(x))

    def _dot_a(self, x):
        return self._a * self._A.dot(self._B.dot(x))

    @property
    def shape(self):
        return (self._A.shape[0], self._B.shape[1])


class Plus(object):
    """
    Produces the addition of the result of two operators.

    Parameter
    ---------
    A : Operator
    B : Operator
    adjoint : Operator, optional

    Attributes
    ----------
    shape : int, int
        provides the shape of the operator
    T : Plus
        provides the transposed operator

    Methods
    -------
    dot(vector)
        Provides multiplication with a vector
    """
    def __init__(self, A, B, adjoint=None):
        self._A, self._B = A, B
        assert self._A.shape == self._B.shape
        if adjoint is None:
            if hasattr(B, "T") and hasattr(A, "T"):
                self.T = Plus(A.T, B.T, adjoint=self)
        else:
            self.T = adjoint

    def dot(self, x):
        return self._A.dot(x) + self._B.dot(x)

    @property
    def shape(self):
        return (self._A.shape[0], self._A.shape[1])


class Function(object):
    """
    Wraps a function (and optionally its adjoint) as operator.

    Parameters
    ----------
    shape : int, int
    F : function
    FT : function, optional
        adjoint function, to define the adjoint operator. (default is None)
    a : float, optional
        optional scaling parameter. (default is 1)

    Attributes
    ----------
    shape : int, int
        provides the shape of the operator
    T : Function
        provides the transposed operator

    Methods
    -------
    dot(vector)
        Provides multiplication with a vector
    """
    def __init__(self, shape, F, FT=None, a=1, _adjoint=None):
        self._F = F
        self._shape = shape
        self._a = a
        self.dot = self._dot if a == 1 else self._dot_a
        if _adjoint is None:
            if FT is not None:
                self.T = Function(shape, FT, a=a, _adjoint=self)
        else:
            self.T = _adjoint

    def _dot_a(self, x):
        return self._a * self._F(x)

    def _dot(self, x):
        return self._F(x)

    @property
    def shape(self):
        return self._shape


class HStack(object):
    """
    Horizontally "stacks" a number of operators to provide an operator accepting
    a wider input vector.

    Attributes
    ----------
    shape : int, int
        provides the shape of the operator
    T : VStack
        provides the transposed operator

    Methods
    -------
    dot(vector)
        Provides multiplication with a vector
    """
    def __init__(self, As, adjoint=None):
        self._As = As
        assert all([As[i].shape[0] == As[i - 1].shape[0] for i in range(1, len(As))])
        self._split = np.cumsum([A.shape[1] for A in As])
        self._shape = (As[0].shape[0], self._split[-1])
        if adjoint is None:
            if all([hasattr(A, "T") for A in As]):
                self.T = VStack([A.T for A in As], adjoint=self)
        else:
            self.T = adjoint

    def dot(self, long_x):
        return np.sum([A.dot(x) for A, x in zip(
            self._As, np.split(long_x, self._split))], axis=0)

    @property
    def shape(self):
        return self._shape


class VStack(object):
    """
    Vertically "stacks" a number of operators to provide an operator supplying a larger
    result vector.

    Attributes
    ----------
    shape : int, int
        provides the shape of the operator
    T : HStack
        provides the transposed operator

    Methods
    -------
    dot(vector)
        Provides multiplication with a vector
    """
    def __init__(self, As, adjoint=None):
        self._As = As
        assert all([As[i].shape[1] == As[i - 1].shape[1] for i in range(1, len(As))])
        self._shape = (sum([A.shape[0] for A in As]), As[0].shape[1])
        if adjoint is None:
            if all([hasattr(A, "T") for A in As]):
                self.T = HStack([A.T for A in As], adjoint=self)
        else:
            self.T = adjoint

    def dot(self, x):
        return np.concatenate([A.dot(x) for A in self._As])

    @property
    def shape(self):
        return self._shape
