#
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Ungermann
#
import numpy as np
import os
import functools

NTHREADS = int(os.getenv("OMP_NUM_THREADS", 1))

npmod = np.fft
try:
    import pyfftw
    fftwmod = pyfftw.interfaces.numpy_fft
    HAVE_FFTW = True
except ImportError:
    HAVE_FFTW = False


def configure(module=None, threads=NTHREADS):
    if module is None:
        module = "fftw" if HAVE_FFTW else "numpy"
    global fft
    global ifft
    global rfft
    global irfft
    global rfft2
    global irfft2
    global fftn
    global ifftn
    global rfftn
    global irfftn
    if module == "numpy":
        fft = npmod.fft
        ifft = npmod.ifft
        rfft2 = npmod.rfft2
        irfft2 = npmod.irfft2
        rfft = npmod.rfft
        irfft = npmod.irfft
        fftn = npmod.fftn
        ifftn = npmod.ifftn
        rfftn = npmod.rfftn
        irfftn = npmod.irfftn
    elif module == "fftw":
        assert HAVE_FFTW

        fft = functools.partial(fftwmod.fft, threads=threads)
        ifft = functools.partial(fftwmod.ifft, threads=threads)
        rfft = functools.partial(fftwmod.rfft, threads=threads)
        irfft = functools.partial(fftwmod.irfft, threads=threads)
        rfft2 = functools.partial(fftwmod.rfft2, threads=threads)
        irfft2 = functools.partial(fftwmod.irfft2, threads=threads)
        fftn = functools.partial(fftwmod.fftn, threads=threads)
        ifftn = functools.partial(fftwmod.ifftn, threads=threads)
        rfftn = functools.partial(fftwmod.rfftn, threads=threads)
        irfftn = functools.partial(fftwmod.irfftn, threads=threads)
    else:
        raise ValueError("configure accepts only 'numpy' and 'fftw', not '{}'".format(module))

def _fft(x):
    """
    Fast Fourier Transform.

    Implemented here for reference of numpy behaviour. Do not use!.
    """
    n = len(x)
    result = np.ndarray(n, dtype=np.complex)
    for m in np.arange(n):
        result[m] = (x * np.exp(-np.pi * 2j * np.arange(n) * m / n)).sum()
    return result


def _ifft(x):
    """
    Inverse Fast Fourier Transform.

    Implemented here for reference of numpy behaviour. Do not use!.
    """
    n = len(x)
    result = np.ndarray(n, dtype=np.complex)
    for m in np.arange(n):
        result[m] = (x * np.exp(np.pi * 2j * np.arange(n) * m / n)).sum()
    return result / n


def _rfft(x):
    """
    Real Fast Fourier Transform.

    Implemented here for reference of numpy behaviour. Do not use!.
    """
    n = len(x)
    result = np.zeros(n / 2 + 1, dtype=np.complex)
    for m in range(len(result)):
        result[m] = (x * np.exp(-np.pi * 2j * np.arange(n) * m / n)).sum()
    return result


def _irfft(x, n=None):
    """
    Inverse Real Fast Fourier Transform.

    Implemented here for reference of numpy behaviour. Do not use!.
    """
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


def _rfft2(x):
    """
    2-D Real Fast Fourier Transform.

    Implemented here for reference of numpy behaviour. Do not use!.
    """
    return fft(rfft(x, axis=1), axis=0)


def _irfft2(x, n=None):
    """
    2-D Inverse Real Fast Fourier Transform.

    Implemented here for reference of numpy behaviour. Do not use!.
    """
    if n is None:
        n = 2 * (x.shape[1] - 1)
    return irfft(ifft(x, axis=0), n=n, axis=1)


def fft_adj(x):
    """
    Adjoint of Fast Fourier Transform, that is the product of the adjoint Jacobian of the
    numpy fft with vector x.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array_like
    """
    n = len(x)
    return ifft(x) * n


def ifft_adj(x):
    """
    Adjoint of Inverse Fast Fourier Transform, that is the product of the adjoint Jacobian of the
    numpy ifft with vector x.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array_like
    """
    n = len(x)
    return fft(x) / n


def rfft_adj(x, n=None):
    """
    Adjoint of Real Fast Fourier Transform, that is the product of the adjoint Jacobian of the
    numpy rfft with vector x.

    Parameters
    ----------
    x : array_like
    n : int, optional
        length of original, untransformed vector

    Returns
    -------
    array_like
    """
    if n is None:
        n = 2 * (len(x) - 1)
    return ifft(x, n=n).real * n


def irfft_adj(x):
    """
    Adjoint of Inverse Real Fast Fourier Transform, that is the product of the adjoint Jacobian
    of the numpy irfft with vector x.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array_like
    """
    n_out = len(x) / 2 + 1
    xp = fft(x) / len(x)
    if len(x) % 2 == 0:
        xp[1:n_out - 1] += np.conj(xp[:n_out - 1:-1])
    else:
        xp[1:n_out] += np.conj(xp[:n_out - 1:-1])
    return xp[:n_out]


def rfft2_adj(x, n=None):
    """
    Adjoint of the 2-D Real Fast Fourier Transform, that is the product of the adjoint
    Jacobian of the numpy rfftn with vector x.

    Parameters
    ----------
    x : array_like
    n : int, optional
        Length of second dimension of original array (the one, the size of which is
        changed by employing the rfft2)

    Returns
    -------
    array_like
    """
    if n is None:
        n = 2 * (x.shape[1] - 1)
    xp = np.zeros((x.shape[0], n), dtype=x.dtype)
    xp[:, :x.shape[1]] = x
    return ifftn(xp).real * x.shape[0] * n


def irfft2_adj(x):
    """
    Adjoint of the 2-D Inverse Real Fast Fourier Transform, that is the product of the
    adjoint Jacobian of the numpy irfftn with vector x.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array_like
    """
    n_out = x.shape[1] / 2 + 1
    xp = fft(x, axis=1) / x.shape[1]
    if x.shape[1] % 2 == 0:
        xp[:, 1:n_out - 1] += np.conj(xp[:, :n_out - 1:-1])
    else:
        xp[:, 1:n_out] += np.conj(xp[:, :n_out - 1:-1])
    return fft(xp[:, :n_out], axis=0) / xp.shape[0]


configure()
