import jutil.operator as op
import numpy as np
from numpy.testing import assert_almost_equal, assert_raises


def test_dot():
    A = np.random.rand(6, 8)
    B = np.random.rand(8, 4)
    x, y = np.random.rand(4), np.random.rand(6)
    dut1 = op.Dot(A, B)

    assert_almost_equal(dut1.dot(x), A.dot(B.dot(x)))
    assert_almost_equal(dut1.T.dot(y), B.T.dot(A.T.dot(y)))
    assert_almost_equal(np.dot(y, dut1.dot(x)), np.dot(dut1.T.dot(y), x))

    A = np.random.rand(6, 8)
    B = np.random.rand(8, 4)
    x, y = np.random.rand(4), np.random.rand(6)
    dut1 = op.Dot(A, B, a=2.)

    assert_almost_equal(dut1.dot(x), 2 * A.dot(B.dot(x)))
    assert_almost_equal(dut1.T.dot(y), 2 * B.T.dot(A.T.dot(y)))
    assert_almost_equal(np.dot(y, dut1.dot(x)), np.dot(dut1.T.dot(y), x))


def test_plus():
    A = np.random.rand(6, 4)
    B = np.random.rand(6, 4)
    x, y = np.random.rand(4), np.random.rand(6)
    dut1 = op.Plus(A, B)

    assert_almost_equal(dut1.dot(x), A.dot(x) + B.dot(x))
    assert_almost_equal(dut1.T.dot(y), A.T.dot(y) + B.T.dot(y))
    assert_almost_equal(np.dot(y, dut1.dot(x)), np.dot(dut1.T.dot(y), x))


def test_function():
    A = np.random.rand(6, 4)
    x, y = np.random.rand(4), np.random.rand(6)
    dut1 = op.Function(A.shape, A.dot, A.T.dot)

    assert_almost_equal(dut1.dot(x), A.dot(x))
    assert_almost_equal(dut1.T.dot(y), A.T.dot(y))
    assert_almost_equal(np.dot(y, dut1.dot(x)), np.dot(dut1.T.dot(y), x))

    A = np.random.rand(6, 4)
    x, y = np.random.rand(4), np.random.rand(6)
    dut1 = op.Function(A.shape, A.dot, A.T.dot, a=2.)

    assert_almost_equal(dut1.dot(x), 2 * A.dot(x))
    assert_almost_equal(dut1.T.dot(y), 2 * A.T.dot(y))
    assert_almost_equal(np.dot(y, dut1.dot(x)), np.dot(dut1.T.dot(y), x))

    A = np.random.rand(6, 4)
    x, y = np.random.rand(4), np.random.rand(6)
    dut1 = op.Function(A.shape, A.dot)

    assert_almost_equal(dut1.dot(x), A.dot(x))

    def dummy((x1, x2)):
        return x1.T.dot(x2)

    assert_raises(AttributeError, dummy, (dut1, x))


def test_stack():
    A = np.random.rand(4, 6)
    B = np.random.rand(4, 8)
    C = np.random.rand(4, 2)

    x, y = np.random.rand(16), np.random.rand(4)

    dut1 = op.HStack([A, B, C])
    assert_almost_equal(dut1.dot(x), A.dot(x[:6]) + B.dot(x[6:6 + 8]) + C.dot(x[6 + 8:]))
    assert_almost_equal(dut1.T.dot(y), np.concatenate([A.T.dot(y), B.T.dot(y), C.T.dot(y)]))
    assert_almost_equal(np.dot(y, dut1.dot(x)), np.dot(dut1.T.dot(y), x))

    dut2 = op.VStack([A.T, B.T, C.T])
    assert_almost_equal(dut2.T.dot(x), A.dot(x[:6]) + B.dot(x[6:6 + 8]) + C.dot(x[6 + 8:]))
    assert_almost_equal(dut2.dot(y), np.concatenate([A.T.dot(y), B.T.dot(y), C.T.dot(y)]))
    assert_almost_equal(np.dot(x, dut2.dot(y)), np.dot(dut2.T.dot(x), y))


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
