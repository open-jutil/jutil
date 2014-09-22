import numpy as np
import numpy.linalg as la


class nocond_wrapper(object):
    def __init__(self, A):
        assert A.shape[0] == A.shape[1]
        self._A = A

    def dot(self, x):
        return self._A.dot(x)

    def cond(self, x):
        return np.array(x, copy=True)

    @property
    def shape(self):
        return self._A.shape


class jacobi_wrapper(object):
    def __init__(self, A):
        assert A.shape[0] == A.shape[1]
        self._A = A
        self._diagonal = np.diag(A)
        nonzero = self._diagonal != 0
        self._diagonal[nonzero] = 1. / self._diagonal[nonzero]

    def dot(self, x):
        return self._A.dot(x)

    def cond(self, x):
        return (self._diagonal * x.T).T

    @property
    def shape(self):
        return self._A.shape


def conj_grad_solve(A, b, max_iter, abs_tol, rel_tols):
    """
    /// Simple implementation of preconditioned conjugate gradient method
    ///
    /// See A. Meister "Numerik linearer Gleichungssysteme", p. 218f
    ///
    /// Compared to the un-preconditioned CG method, this needs memory for an
    /// additional Vector and uses an additional scalar-product and one
    /// Matrix-Vector product per iteration.
    ///
    /// \param A is a functor defining the matrix, realising the mult() function
    ///   for supplying a vector to be multiplied returning the result, a cond()
    ///   function for supplying a vector and multiplying it with an approximate
    ///   inversion and size1()/size2() functions returning its dimensions
    /// \param b is the RHS of the lineqr equation system
    /// \param x gives the initial guess and returns the result of the method
    /// \param max_iter determines the maximum number of iterations.
    /// \param abs_tol determines the tolerance for the remaining
    ///   residuum. The algorithm will terminate if ||Ax - b||_2 / ||b||_2 < abs_tol.
    /// \param rel_tol determines the expected residual reduction. The
    ///   algorithm will terminate when ||Ax_0 - b||_2 / ||Ax - b||_2 < rel_tol.
    ///   By specifying red_tol to be, say 1e-2, one requires the residuum to
    ///   be reduced to one hundreth of its initial value.
    """

    rel_tols = np.array(rel_tols, copy=True).reshape(-1)
    assert len(rel_tols) > 0

    assert np.all(np.diff(rel_tols) > 0)

    x = np.zeros_like(b)
    r = b - A.dot(x)
    p = A.cond(r)
    print r.shape, p.shape
    alpha = np.dot(r, p)
    norm_b = la.norm(b)
    xs = [0 for _ in xrange(len(rel_tols))]

    if max_iter < 1:
        max_iter = A.shape[0]

    for i in xrange(max_iter):
        norm = la.norm(r);
        norm_div_norm_b = norm / norm_b
        for j in xrange(len(rel_tols)):
          if norm_div_norm_b < rel_tols[j]:
            xs[j] = np.array(x, copy=True)
            if j > 0:
              rel_tols[j] = -1

        if (norm < abs_tol) or (norm_div_norm_b < rel_tols[0]):
            break

        v = A.dot(p)
        lambd = alpha / np.dot(v, p);
        assert not np.isnan(lambd)

        x += lambd * p
        r -= lambd * v

        z = A.cond(r)

        new_alpha = np.dot(r, z)
        p *= new_alpha / alpha
        p += z

        alpha = new_alpha

    print "CG needed {}{}  iterations to reduce to {} {}".format(
            ("max=" if (i == max_iter) else  ""), i, la.norm(r),
              la.norm(r) / norm_b, norm_b)

    for j in [_j for _j in range(len(rel_tols)) if rel_tols[_j] != -1]:
        xs[j] = np.array(x, copy=True)
    return xs


def conj_grad_tall_solve(A, bs, max_iter, abs_tol, rel_tol):
    """
    /// Simple implementation of preconditioned conjugate gradient method
    ///
    /// See A. Meister "Numerik linearer Gleichungssysteme", p. 218f
    ///
    /// Compared to the un-preconditioned CG method, this needs memory for an
    /// additional Vector and uses an additional scalar-product and one
    /// Matrix-Vector product per iteration.
    ///
    /// \param A is a functor defining the matrix, realising the mult() function
    ///   for supplying a vector to be multiplied returning the result, a cond()
    ///   function for supplying a vector and multiplying it with an approximate
    ///   inversion and size1()/size2() functions returning its dimensions
    /// \param b is the RHS of the lineqr equation system
    /// \param x gives the initial guess and returns the result of the method
    /// \param max_iter determines the maximum number of iterations.
    /// \param abs_tol determines the tolerance for the remaining
    ///   residuum. The algorithm will terminate if ||Ax - b||_2 / ||b||_2 < abs_tol.
    /// \param rel_tol determines the expected residual reduction. The
    ///   algorithm will terminate when ||Ax_0 - b||_2 / ||Ax - b||_2 < rel_tol.
    ///   By specifying red_tol to be, say 1e-2, one requires the residuum to
    ///   be reduced to one hundreth of its initial value.
    """

    xs = np.zeros_like(bs)
    rs = bs - A.dot(xs)
    ps = A.cond(rs)
    alphas = np.einsum('ij,ij->j', rs, ps)
    norms_b = np.asarray([la.norm(b) for b in bs.T])
    print norms_b

    if max_iter < 1:
        max_iter = A.shape[0]

    for i in xrange(max_iter):
        norms = np.asarray([la.norm(r) for r in rs.T])
        norm_div_norm_b = norms / norms_b

        if np.all((norms < abs_tol) | (norm_div_norm_b < rel_tol)):
            break

        vs = A.dot(ps)
        vs_dot_ps = np.einsum('ij,ij->j', vs, ps)
        lambds = np.asarray([alphas[j] / vs_dot_ps[j] if norms[j] != 0 else 0 for j in range(len(alphas))])

        xs += lambds * ps
        rs -= lambds * vs

        zs = A.cond(rs)

        new_alphas = np.einsum('ij,ij->j', rs, zs)
        ps += new_alphas / alphas
        ps += zs

        alphas = new_alphas
        alphas[alphas == 0] = 1

    print "CG needed {}{}  iterations to reduce to {} {}".format(
            ("max=" if (i == max_iter) else  ""), i, la.norm(rs),
              np.asarray([la.norm(r) for r in rs.T]) / norms_b, norms_b)

    return xs


def _test():
    A = np.random.rand(5, 5)
    A = A.T.dot(A)
    print A
    for i in range(5):
        A[i, i] += 1 + i ** 15
    print A
    b = np.random.rand(5)
    b_tall = np.random.rand(5, 4)
    An = nocond_wrapper(A)
    Aj = jacobi_wrapper(A)
    print la.cond(A)
    print conj_grad_solve(An, b, 100, 0, 1e-20)
    print conj_grad_solve(An, b, 100, 0, [1e-20, 1e-1, 0.9])
    print conj_grad_solve(Aj, b, 100, 0, [1e-20, 1e-1, 0.9])
    print conj_grad_tall_solve(An, b_tall, -1, 0, 1e-20)
    print conj_grad_tall_solve(Aj, b_tall, -1, 0, 1e-20)

_test()
