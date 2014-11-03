import numpy as np
import jutil.fft as jfft
from numpy.testing import assert_equal, assert_almost_equal

import jutil.diff as jdiff

NS = [2, 3, 4, 5, 10, 11, 16, 17, 21]


def r2c(x):
    r, i = np.split(x, 2)
    return r + 1j * i


def c2r(x):
    r, i = x.real, x.imag
    return np.concatenate([r, i])


def test_fft():
    for n in NS:
        x = np.random.random(n) + 1j * np.random.random(n)
        assert_almost_equal(np.fft.fft(x), jfft.fft(x))


def test_ifft():
    for n in NS:
        x = np.random.random(n) + 1j * np.random.random(n)
        assert_almost_equal(np.fft.ifft(x), jfft.ifft(x))


def test_rfft():
    for n in NS:
        x = np.random.random(n)
        assert_almost_equal(np.fft.rfft(x), jfft.rfft(x))


def test_irfft():
    for n in NS:
        m = n / 2 + 1
        x = np.random.random(m) + 1j * np.random.random(m)
        print x, np.fft.irfft(x, n=n)#, jfft.irfft(x, n=n)
        assert_almost_equal(np.fft.irfft(x, n=n), jfft.irfft(x, n=n))


def test_rfftn():
    for n1 in NS:
        for n2 in NS:
            x = np.random.random((n1, n2))
            assert_almost_equal(np.fft.rfftn(x), jfft.rfftn(x))

def test_irfftn():
    for n1 in NS:
        for n2 in NS:
            m = n2 / 2 + 1
            x = np.random.random((n1, m)) + 1j * np.random.random((n1, m))
            assert_almost_equal(np.fft.irfftn(x, s=(n1, n2)), jfft.irfftn(x, n=n2))


def test_rfft_adj():
    for n in NS:
        A = jdiff.fd_jac(np.fft.rfft, np.zeros(n))
        x = np.random.random(A.shape[0]) + 1j * np.random.random(A.shape[0])
        ATx_1 = A.T.dot(x)
        ATx_2 = jfft.rfft_adj(x, n=n)
        print A.T.shape, ATx_1.shape, ATx_2.shape, x.shape, n
        print ATx_1 - ATx_2
        assert_almost_equal(ATx_1, ATx_2)


def test_irfft_adj():
    for n in NS:
        m = n / 2 + 1
        def irfft_wrap(x):
            x = r2c(x)
            result = np.fft.irfft(x.reshape(m), n=n)
            print "pp", x, result.reshape(-1)
            return result.reshape(-1)
        A = jdiff.fd_jac(irfft_wrap, np.zeros(m * 2))

        def irfft_wrap2(x):
            result = jfft.irfft_adj(x)
            return c2r(result.reshape(-1))
        B = jdiff.fd_jac(irfft_wrap2, np.zeros(n)).T
        assert_almost_equal(abs(A - B).sum(), 0)

        x = np.random.random(A.shape[0])# + 1j * np.random.random(A.shape[0])
        ATx_1 = r2c(A.T.dot(x))
        ATx_2 = jfft.irfft_adj(x)
        assert_almost_equal(ATx_1, ATx_2)


def test_rfftn_adj():
    for n1 in NS:
        for n2 in NS:
            def rfftn_wrap(x):
                result = np.fft.rfftn(x.reshape(n1, n2))
                return result.reshape(-1)
            A = jdiff.fd_jac(rfftn_wrap, np.zeros(n1 * n2))

            x = np.random.random(A.shape[0]) + 1j * np.random.random(A.shape[0])
            ATx_1 = A.T.dot(x)
            ATx_2 = jfft.rfftn_adj(x.reshape(n1, n2 / 2 + 1), n=n2).reshape(-1)
            assert_almost_equal(ATx_1, ATx_2)


def test_irfftn_adj():
    for n1 in [2, 4, 10, 11, 16, 17, 21]:
        for n2 in [2, 4, 10, 11, 16, 17, 21]:
            print
            print "==========================================="
            print n1, n2
            m = n2 / 2 + 1

            def irfftn_wrap(x):
                x = r2c(x)
                result = np.fft.irfftn(x.reshape(n1, m), s=(n1, n2))
                return result.reshape(-1)
            A = jdiff.fd_jac(irfftn_wrap, np.zeros(n1 * m * 2))
            """
            def irfftn_wrap2(x):
                result = jfft.irfftn_adj(x.reshape(n1, n2))
                return c2r(result.reshape(-1))
            B = jdiff.fd_jac(irfftn_wrap2, np.zeros(n1 * n2)).T

            def ifft_wrap(x):
                x = r2c(x)
                result = np.fft.ifft(x.reshape(n1, m), axis=0)
                return c2r(result.reshape(-1))
            C = jdiff.fd_jac(ifft_wrap, np.zeros(n1 * m * 2))

            def irfft_wrap(x):
                x = r2c(x)
                result = np.fft.irfft(x.reshape(n1, m), n=n2, axis=1)
                return result.reshape(-1)
            D = jdiff.fd_jac(irfft_wrap, np.zeros(n1 * m * 2))

            def irfft_wrap2(x):
                x = x.reshape(n1, n2)
                result = np.asarray([jfft.irfft_adj(x[i, :]) for i in range(x.shape[0])])
                return c2r(result.reshape(-1))
            D2 = jdiff.fd_jac(irfft_wrap2, np.zeros(n1 * n2)).T

            def dummy_wrap(x):
                x = irfft_wrap(x)
                print x.shape
                x = ifft_wrap(x)
                return x
#            DU = jdiff.fd_jac(dummy_wrap, np.zeros(n1 * m * 2))


           #C = C.reshape(n1*n2,n1,m)
            #C[:,:,0] = C[:,:,0].real
            #if n2 % 2 == 0:
            #    C[:,:,-1] = C[:,:,-1].real

#            C[:,:,2] = C[:,:,2].real
            #C=C.reshape(n1*n2, n1*m)

            print "A", A
            print "B", B
            print "C", C, C.shape
            print "DC", D.dot(C)
#            print "DU", DU
            print "D", D
            print "D2", D2
            print "C-C.T", abs(C-C.T).sum()

            print A.shape, B.shape, "\nA-B", abs(A-B).sum()
            print "A-D", abs(A - D.dot(C)).sum()
            print "D2-D", abs(D - D2).sum()
            x = np.random.random(A.shape[1])# + 1j * np.random.random(A.shape[1])
            print "A", A.dot(x).reshape(n1, n2)
            print "B", B.dot(x).reshape(n1, n2)
#            print "C", np.fft.irfftn(x.reshape(n1, m), s=(n1,n2))
            """
            x = np.random.random(A.shape[0])# + 1j * np.random.random(A.shape[0])
            ATx_1 = r2c(A.T.dot(x))
            ATx_2 = jfft.irfftn_adj(x.reshape(n1, n2)).reshape(-1)
            #print ATx_1.shape, ATx_2.shape
            #print "1\n", ATx_1
            #print "2\n", ATx_2
            #print A.shape
            ATx_2 = ATx_2.reshape(-1)
            #print abs(ATx_1 - ATx_2).sum()
            assert_almost_equal(ATx_1, ATx_2)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
