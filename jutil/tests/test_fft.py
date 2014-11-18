import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

import jutil.fft as jfft
import jutil.diff as jdiff

NS = [2, 3, 4, 5, 10, 11, 16, 17, 21]


def r2c(x):
    print x.shape
    r, i = np.split(x, 2)
    return r + 1j * i


def c2r(x):
    r, i = x.real, x.imag
    return np.concatenate([r, i])


def wrap(f, shape=None, ci=False, co=False, kw={}):
    def result(x):
        if ci:
            x = r2c(x)
        if shape is not None:
            x = x.reshape(shape)
        y = f(x, **kw).reshape(-1)
        if co:
            y = c2r(y)
        return y
    return result


def test_fft():
    for n in NS:
        x = np.random.random(n) + 1j * np.random.random(n)
        assert_almost_equal(np.fft.fft(x), jfft._fft(x))


def test_ifft():
    for n in NS:
        x = np.random.random(n) + 1j * np.random.random(n)
        assert_almost_equal(np.fft.ifft(x), jfft._ifft(x))


def test_rfft():
    for n in NS:
        x = np.random.random(n)
        assert_almost_equal(np.fft.rfft(x), jfft._rfft(x))


def test_irfft():
    for n in NS:
        m = n / 2 + 1
        x = np.random.random(m) + 1j * np.random.random(m)
        assert_almost_equal(np.fft.irfft(x, n=n), jfft._irfft(x, n=n))


def test_rfft2():
    for n1 in NS:
        for n2 in NS:
            x = np.random.random((n1, n2))
            assert_almost_equal(np.fft.rfft2(x), jfft._rfft2(x))

def test_irfft2():
    for n1 in NS:
        for n2 in NS:
            m = n2 / 2 + 1
            x = np.random.random((n1, m)) + 1j * np.random.random((n1, m))
            assert_almost_equal(np.fft.irfft2(x, s=(n1, n2)), jfft._irfft2(x, n=n2))


def test_fft_adj():
    for n in NS:
        A = jdiff.fd_jac(wrap(np.fft.fft, None, True, True), np.zeros(2 * n))
        B = jdiff.fd_jac(wrap(jfft.fft_adj, None, True, True), np.zeros(2 * n)).T
        assert_almost_equal(abs(A - B).sum(), 0, err_msg=str((A, B)))

        x = np.random.random(2 * n)
        ATx_1 = A.T.dot(x)
        ATx_2 = c2r(jfft.fft_adj(r2c(x)))
        assert_almost_equal(ATx_1, ATx_2)


def test_ifft_adj():
    for n in NS:
        A = jdiff.fd_jac(wrap(np.fft.ifft, None, True, True), np.zeros(2 * n))
        B = jdiff.fd_jac(wrap(jfft.ifft_adj, None, True, True), np.zeros(2 * n)).T
        assert_almost_equal(abs(A - B).sum(), 0, err_msg=str((A, B)))

        x = np.random.random(2 * n)
        ATx_1 = A.T.dot(x)
        ATx_2 = c2r(jfft.ifft_adj(r2c(x)))
        assert_almost_equal(ATx_1, ATx_2)


def test_rfft_adj():
    for n in NS:
        m = n / 2 + 1

        A = jdiff.fd_jac(wrap(np.fft.rfft, None, False, True), np.zeros(n))
        B = jdiff.fd_jac(wrap(jfft.rfft_adj, None, True, False, {"n":n}), np.zeros(2 * m)).T
        assert_almost_equal(abs(A - B).sum(), 0, err_msg=str((A,B)))

        x = np.random.random(A.shape[0])
        ATx_1 = A.T.dot(x)
        ATx_2 = jfft.rfft_adj(r2c(x), n=n)
        assert_almost_equal(ATx_1, ATx_2)


def test_irfft_adj():
    for n in NS:
        m = n / 2 + 1

        A = jdiff.fd_jac(wrap(np.fft.irfft, m, True, False, {"n":n}), np.zeros(m * 2))
        B = jdiff.fd_jac(wrap(jfft.irfft_adj, None, False, True), np.zeros(n)).T
        assert_almost_equal(abs(A - B).sum(), 0, err_msg=str((A, B)))

        x = np.random.random(A.shape[0])
        ATx_1 = r2c(A.T.dot(x))
        ATx_2 = jfft.irfft_adj(x)
        assert_almost_equal(ATx_1, ATx_2)


def test_rfft2_adj():
    for n1 in NS:
        for n2 in NS:
            m = n2 / 2 + 1

            A = jdiff.fd_jac(wrap(np.fft.rfft2, (n1, n2), False, True), np.zeros(n1 * n2))
            B = jdiff.fd_jac(wrap(jfft.rfft2_adj, (n1, m), True, False, {"n":n2}),
                             np.zeros(2 * n1 * m)).T
            assert_almost_equal(abs(A - B).sum(), 0, err_msg=str((A, B)))

            x = np.random.random(A.shape[0])
            ATx_1 = A.T.dot(x)
            ATx_2 = jfft.rfft2_adj(r2c(x).reshape(n1, m), n=n2).reshape(-1)
            assert_almost_equal(ATx_2, ATx_2, err_msg=str((ATx_1, ATx_2)))


def test_irfft2_adj():
    for n1 in [2, 4, 10, 11, 16, 17, 21]:
        for n2 in [2, 4, 10, 11, 16, 17, 21]:
            m = n2 / 2 + 1

            A = jdiff.fd_jac(wrap(np.fft.irfft2, (n1, m), True, False, {"s":(n1, n2)}),
                             np.zeros(n1 * m * 2))
            B = jdiff.fd_jac(wrap(jfft.irfft2_adj, (n1, n2), False, True), np.zeros(n1 * n2)).T
            assert_almost_equal(abs(A - B).sum(), 0, err_msg=str((A, B)))

            x = np.random.random(A.shape[0])
            ATx_1 = r2c(A.T.dot(x))
            ATx_2 = jfft.irfft2_adj(x.reshape(n1, n2)).reshape(-1)
            assert_almost_equal(ATx_1, ATx_2)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()