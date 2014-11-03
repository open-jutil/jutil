import numpy as np


def _fft(x):
    n = len(x)
    result = np.ndarray(n, dtype=np.complex)
    for m in np.arange(n):
        result[m] = (x * np.exp(-np.pi * 2j * np.arange(n) * m / n)).sum()
    return result


def _ifft(x):
    n = len(x)
    result = np.ndarray(n, dtype=np.complex)
    for m in np.arange(n):
        result[m] = (x * np.exp(np.pi * 2j * np.arange(n) * m / n)).sum()
    return result / n


def _rfft(x):
    n = len(x)
    result = np.zeros(n / 2 + 1, dtype=np.complex)
    for m in range(len(result)):
        result[m] = (x * np.exp(-np.pi * 2j * np.arange(n) * m / n)).sum()
    return result


def _irfft(x, n=None):
    if n is None:
        n = 2 * (len(x) - 1)
    xp = np.ndarray(n, dtype=np.complex)
    xp[:len(x)] = x
    xp[0] = x[0].real

    extra = n % 2
    if extra == 0:
        extra2 = -2
        xp[len(x) - 1] = xp[len(x) - 1].real
    else:
        extra2 = None
    xp[0] = x[0].real

    xp[len(x):] = np.conj(x[extra2:0:-1])

    result = np.zeros(n, dtype=np.complex)
    for m in np.arange(n):
        result[m] = (xp * np.exp(np.pi * 2j * np.arange(n) * m / n)).sum()
    return result / n


def _rfftn(x):
    return np.fft.fft(np.fft.rfft(x, axis=1), axis=0)


def _irfftn(x, n=None):
    if n is None:
        n = 2 * (x.shape[1] - 1)
    return np.fft.irfft(np.fft.ifft(x, axis=0), n=n, axis=1)


def fft_adj(x):
    n = len(x)
    return np.fft.ifft(x) * n


def ifft_adj(x):
    n = len(x)
    return np.fft.fft(x) / n


def rfft_adj(x, n=None):
    if n is None:
        n = 2 * (len(x) - 1)
    return np.fft.ifft(x, n=n).real * n


def irfft_adj(x):
    n_out = len(x) / 2 + 1
    xp = np.fft.fft(x) / len(x)
    if len(x) % 2 == 0:
        xp[1:n_out - 1] += np.conj(xp[:n_out - 1:-1])
#        xp[n_out - 1] = xp[n_out - 1].real
    else:
        xp[1:n_out] += np.conj(xp[:n_out - 1:-1])
#    xp[0] = xp[0].real
    return xp[:n_out]


def rfftn_adj(x, n=None):
    if n is None:
        n = 2 * (x.shape[1] - 1)
    xp = np.zeros((x.shape[0], n), dtype=x.dtype)
    xp[:, :x.shape[1]] = x
    return np.fft.ifftn(xp).real * x.shape[0] * n


def irfftn_adj(x):
    n_out = x.shape[1] / 2 + 1
    xp = np.fft.fft(x, axis=1) / x.shape[1]
    if x.shape[1] % 2 == 0:
        xp[:, 1:n_out - 1] += np.conj(xp[:, :n_out-1:-1])
#        xp[:, n_out - 1] = xp[:, n_out - 1].real
    else:
        xp[:, 1:n_out] += np.conj(xp[:, :n_out-1:-1])
#    xp[:, 0] = xp[:, 0].real
    return np.fft.fft(xp[:, :n_out], axis=0) / xp.shape[0]

