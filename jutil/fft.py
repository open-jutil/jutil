import numpy as np


def fft(x):
    n = len(x)
    return np.asarray([(x * np.exp(-np.pi * 2j * np.arange(n) * m / n)).sum() for m in np.arange(n)])


def ifft(x):
    n = len(x)
    return np.asarray([(x * np.exp(np.pi * 2j * np.arange(n) * m / n)).sum() for m in np.arange(n)]) / n


def rfft(x):
    n = len(x)
#    return np.asarray([
#        (x * np.exp(-np.pi * 2j * np.arange(n) * m / n)).sum()
#        for m in np.arange(n / 2 + 1)])

    result = np.zeros(n / 2 + 1, dtype=np.complex)
    for m in range(len(result)):
        result[m] = (x * np.exp(-np.pi * 2j * np.arange(n) * m / n)).sum()
    return result


def irfft(x, n=None):
    if n is None:
        n = 2 * (len(x) - 1)
    xp = np.ndarray(n, dtype=np.complex)
    xp[:len(x)] = x
    extra = n % 2
    if extra == 0:
        extra2 = -1
    else:
        extra2 = None
    xp[len(x):] = np.conj(x[1:extra2][::-1])
    x = xp
#   return np.asarray([
#       (x * np.exp(np.pi * 2j * np.arange(n) * m / n)).sum()
#       for m in np.arange(n)]) / n
    result = np.zeros(n, dtype=np.complex)
    for m in np.arange(n):
        result[m] = (x * np.exp(np.pi * 2j * np.arange(n) * m / n)).sum() / n
    return result
#
#def rfft_adj4(x, n=8):
#    if n is None:
#        n = 2 * len(x)
#    result = np.zeros(n)
#    for m in range(len(x)):
#        result += x[m] * (np.exp(-np.pi * 2j * np.arange(n) * m / n))
#    return result

#def rfft_adj2(x, n=8):
#    if n is None:
#        n = 2 * len(x)
#    result = np.zeros(n)
#    for m in range(len(result)):
#        result[m] = (x * (np.exp(-np.pi * 2j * np.arange(len(x)) * m / n))).sum()
#    return result

#def rfft_adj3(x, n=8):
#    xp = np.ndarray(n, dtype=x.dtype)
#    xp[:len(x)] = x
#    extra = len(x) % 2
#    if extra == 0:
#        extra2 = None
#    else:
#        extra2 = -extra
#    xp[len(x):] = 0#np.conj(x[-n+len(x)-extra:extra2][::-1])
#    x = xp
#    return np.asarray([
#        (x * np.exp(-np.pi * 2j * np.arange(n) * m / n)).sum()
#        for m in np.arange(n)])


def rfft_adj(x, n=None):
    if n is None:
        n = 2 * (len(x) - 1)
    xp = np.zeros(n, dtype=x.dtype)
    xp[:len(x)] = x
    return np.fft.fft(xp)


#def irfft_adj2(x):
#    n = len(x)
#    result = np.zeros(n, dtype=np.complex)
#    for m in np.arange(n):
#        result += (x[m] * np.exp(np.pi * 2j * np.arange(n) * m / n)) / n
#    result[1:4] += result[-3:][::-1]
#    return result[:5]


def irfft_adj(x):
    xp = np.fft.ifft(x)
    n_out = len(x) / 2 + 1
    if len(x) % 2 == 0:
        xp[1:n_out - 1] += xp[n_out:][::-1]
    else:
        xp[:n_out-1] += xp[n_out:][::-1]
    return xp[:n_out]


def rfftn_adj(x, n=None):
    if n is None:
        n = 2 * (x.shape[1] - 1)
    shape = (x.shape[0], n)
    xp = np.zeros(shape, dtype=x.dtype)
    xp[:,:x.shape[1]] = x
    return np.fft.fftn(xp)

# rfftn(x) = ifft(irfft(x)) = A * B *x, B^T A^T x'
# rfftn : nxm -> nx(m/2+1)
# rfftn : nx(m/2+1) -> nxm
# => rfftn'(x) = ifft'(irfft(x))*irfft'(x)
def irfftn_adj(x):
#    xp = np.fft.irfftn(x)
#    xp2 = np.fft.irfft(np.fft.ifft(x,axis=0), axis=1)
    xp_adj1 = np.fft.ifft(x, axis=0)
    return np.asarray([irfft_adj(xp_adj1[i, :]) for i in range(xp_adj1.shape[0])])
    xp = np.asarray([irfft_adj(x[i, :]) for i in range(x.shape[0])])
    print "xp"; xp, xp.shape
    return np.fft.ifft(xp)
    n_out =  x.shape[1] / 2 + 1
    xp2 = xp.copy()
    if x.shape[1] % 2 == 0:
        xp2 = xp.copy()
        xp[:, 1:n_out - 1] += xp[:, n_out:][:, ::-1]
        print "oo", xp.shape, (1, n_out-1)
        return xp[:, :n_out], xp2
    else:
        xp2 = xp.copy()
        xp[:, 1:n_out] += xp[:, n_out:][:, ::-1]
        print "oo"
        return xp[:, :n_out].reshape(-1), xp2.reshape(-1)

    return xp[:, :n_out]#.reshape(-1)

