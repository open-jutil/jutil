
def _safe_inverse(vec):
    nonzero = (vec != 0)
    result = vec.copy()
    result[nonzero] = 1. / vec[nonzero]
    return result


class JacobiPreconditioner(object):
    def __init__(self, A):
        assert A.shape[0] == A.shape[1]
        assert hasattr(A, "diagonal")
        self._A = A
        self._diagonal = _safe_inverse(A.diagonal())

    def dot(self, x):
        return (self._diagonal * x.T).T

    @property
    def shape(self):
        return self._A.shape


class CostFunctionPreconditioner(object):
    def __init__(self, J, x_i):
        self._J = J
        self._x_i = x_i
        if hasattr(J, "hess_diagonal"):
            self._diagonal = _safe_inverse(J.hess_diagonal(self._x_i))
            self.dot = self._dot_diag
        else:
            self.dot = lambda x: x.copy()

    def _dot_diag(self, x):
        return (self._diagonal * x.T).T

    @property
    def shape(self):
        return (self._J.n, self._J.n)




