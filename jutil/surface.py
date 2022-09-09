import numpy as np
import scipy.sparse
import jutil.lsqr


def minimum_curvature_surface(data, lam=None, T=0.5):
    """
    Function to fill in missing points by means of minimium
    curvature/cubic splines under tension.

    See also

    Gridding with continuous curvature splines in tension
    Smith,W. H. F. et al.
    GEOPHYSICS(1990),55(3):293
    http://dx.doi.org/10.1190/1.1442837

    (the implementation here is much simpler/different,
     but quite efficient; pygmt.surface implements the
     Smith code; Results for T >= 0.4 are very similar
     to pygmt implementation)

    Parameters
    ----------

    data : 2-D masked ndarray
        2-D array with data. missing values must be masked

    lam : weight or None
        if None, the given data are forced. Otherwise lam specifies
        the weight of the "measurements" in relation to the smoothing
        condition.

    T : float
        tension parameter between 0 and 1.

    Returns
    -------

    A 2-D array with filled data

    """

    def make_Ab(x, lam=None, T=0.5):
        """
        constructs linear equation system
        """
        n = len(x.reshape(-1))
        A = scipy.sparse.lil_matrix((5 * n, n))
        b = np.zeros(A.shape[0])

        def idx(i, j):
            return np.ravel_multi_index((i, j), x.shape)

        def enter_a(x, A, b, row, i, j, val):
            """
            filters out known values for lam = None
            """
            if x.mask[i, j] or lam is not None:
                A[row, idx(i, j)] = val
            else:
                b[row] -= val * x[i, j]

        row = 0
        # known values
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if not x.mask[i, j]:
                    if lam is None:
                        A[row, idx(i, j)] = 1
                        b[row] = x[i, j]
                    else:
                        A[row, idx(i, j)] = lam
                        b[row] = x[i, j] * lam
                row += 1
        # first derivatives
        for i in range(x.shape[0] - 1):
            for j in range(x.shape[1]):
                enter_a(x, A, b, row, i, j, T)
                enter_a(x, A, b, row, i + 1, j, -T)
                row += 1
        for i in range(x.shape[0]):
            for j in range(x.shape[1] - 1):
                enter_a(x, A, b, row, i, j, T)
                enter_a(x, A, b, row, i, j + 1, -T)
                row += 1
        # second derivatives
        for i in range(x.shape[0] - 2):
            for j in range(x.shape[1]):
                enter_a(x, A, b, row, i, j, 1 - T)
                enter_a(x, A, b, row, i + 1, j, -2 * (1 - T))
                enter_a(x, A, b, row, i + 2, j, 1 - T)
                row += 1
        for i in range(x.shape[0]):
            for j in range(x.shape[1] - 2):
                enter_a(x, A, b, row, i, j, 1 - T)
                enter_a(x, A, b, row, i, j + 1, -2 * (1 - T))
                enter_a(x, A, b, row, i, j + 2, 1 - T)
                row += 1
        return A.tocsr(), b

    A, b = make_Ab(data, lam, T)
    x0 = jutil.lsqr.lsqr_solve(A, b).reshape(data.shape)
    return x0
