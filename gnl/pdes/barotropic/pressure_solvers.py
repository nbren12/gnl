"""Module for fourier based pressure solvers
"""
import logging
logger = logging.getLogger(__name__)

from numpy import pi, real
import numpy as np

try:
    import pyfftw

    fft2 = pyfftw.interfaces.scipy_fftpack.fft2
    ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
    fftfreq = pyfftw.interfaces.scipy_fftpack.fftfreq
    pyfftw.interfaces.cache.enable()
    logger.info("using pyfftw")
except ImportError:
    from scipy.fftpack import fft2, ifft2, fftfreq
    logger.info("using scipt fftpack")


class CollocatedFourierSolver(object):
    def __init__(self, shape, space):
        "docstring"
        nx, ny = shape
        dx, dy = space

        scal_y = 2 * pi / dy / ny
        scal_x = 2 * pi / dx / nx

        k = fftfreq(nx, 1 / nx)[:, None] * 1j * scal_x
        l = fftfreq(ny, 1 / ny)[None, :] * 1j * scal_y

        lapl = k**2 + l**2
        lapl[0, 0] = 1.0

        self.k, self.l, self.lapl = k, l, lapl

    def solve(self, uv, dx, dy):
        nx, ny = uv.shape[1:]

        k, l, lapl = self.k, self.l, self.lapl

        u = uv[0]
        v = uv[1]

        fu = fft2(u)
        fv = fft2(v)

        p = (fu * k + fv * l) / lapl

        px = real(ifft2(p * k))
        py = real(ifft2(p * l))

        u[:] -= px
        v[:] -= py

        return px, py


def shift_filter(shift, n):
    """
    >>> n =10
    >>> x = np.arange(n)
    >>> sx = np.take(x, np.r_[2:12], mode='wrap')
    >>> fsx = np.real(ifft(shift_filter(2, n) * fft(x)))
    >>> np.testing.assert_allclose(sx, fsx, atol=1e-12)
    """
    return np.exp(2 * pi * 1j * shift * fftfreq(n))


def get_cd_filter(n):
    """
    >>> n = 10
    >>> x = np.arange(10)
    >>> f = get_cd_filter(n)
    >>> df = correlate1d(x, [-1, 0, 1], axis=0, mode='wrap')
    >>> dfft = np.real(ifft(f * fft(x)))
    >>> np.testing.assert_allclose(df, dfft)
    """
    return -shift_filter(-1, n) + shift_filter(1, n)


class CollocatedFDSolver(object):
    def __init__(self, shape, space):
        "docstring"
        nx, ny = shape
        dx, dy = space
        self.mpx = (shift_filter(1, nx)[:, None] + 1) / 2
        self.mmx = (1 + shift_filter(-1, nx)[:, None]) / 2
        self.dpx = (shift_filter(1, nx) - 1)[:, None] / dx
        self.dmx = (1 - shift_filter(-1, nx))[:, None] / dx

        self.mpy = (shift_filter(1, ny)[None, :] + 1) / 2
        self.mmy = (1 + shift_filter(-1, ny)[None, :]) / 2
        self.dpy = (shift_filter(1, ny) - 1)[None, :] / dy
        self.dmy = (1 - shift_filter(-1, ny))[None, :] / dy

    def solve(self, uc, dx, dy):
        import numexpr as ne
        nx, ny = uc.shape[1:]
        u = uc[0]
        v = uc[1]

        fu = fft2(u)
        fv = fft2(v)

        mpx = self.mpx
        mmx = self.mmx
        dpx = self.dpx
        dmx = self.dmx
        mpy = self.mpy
        mmy = self.mmy
        dpy = self.dpy
        dmy = self.dmy

        d = ne.evaluate("fu*mmy * dmx + fv * mmx * dmy")
        lapl = ne.evaluate("mpy  * mmy * dpx * dmx + mpx*mmx *dpy *dmy")
        lapl[0, 0] = 1.0

        p = d / lapl
        px = np.real(ifft2(mpy * dpx * p))
        py = np.real(ifft2(mpx * dpy * p))

        u -= px
        v -= py

        return px, py
