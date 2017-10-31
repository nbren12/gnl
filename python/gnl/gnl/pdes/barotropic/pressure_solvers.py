"""Module for fourier based pressure solvers


"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
from numpy import pi, real
from numpy.linalg import norm
from scipy.ndimage import correlate1d

try:
    import pyfftw

    dct = pyfftw.interfaces.scipy_fftpack.dct
    idct = pyfftw.interfaces.scipy_fftpack.idct
    fft = pyfftw.interfaces.scipy_fftpack.fft
    ifft = pyfftw.interfaces.scipy_fftpack.ifft
    fft2 = pyfftw.interfaces.scipy_fftpack.fft2
    ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
    fftfreq = pyfftw.interfaces.scipy_fftpack.fftfreq
    pyfftw.interfaces.cache.enable()
    logger.info("using pyfftw")
except ImportError:
    from scipy.fftpack import fft2, ifft2, fftfreq, fft, ifft
    logger.info("using scipt fftpack")
import pylab as pl

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

    def solve(self, u, v, dx, dy):
        nx, ny = u.shape

        k, l, lapl = self.k, self.l, self.lapl

        fu = fft2(u)
        fv = fft2(v)

        p = (fu * k + fv * l) / lapl

        px = real(ifft2(p * k))
        py = real(ifft2(p * l))

        u[:] -= px
        v[:] -= py

        # self.p = real(ifft2(p))

        return px, py


def shift_filter(shift, n, bc=None):
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

        self.shape=shape
        self.mpx = (shift_filter(1, nx)[:, None] + 1) / 2
        self.mmx = (1 + shift_filter(-1, nx)[:, None]) / 2
        self.dpx = (shift_filter(1, nx) - 1)[:, None] / dx
        self.dmx = (1 - shift_filter(-1, nx))[:, None] / dx

        self.mpy = (shift_filter(1, ny)[None, :] + 1) / 2
        self.mmy = (1 + shift_filter(-1, ny)[None, :]) / 2
        self.dpy = (shift_filter(1, ny) - 1)[None, :] / dy
        self.dmy = (1 - shift_filter(-1, ny))[None, :] / dy

    def solve(self, u, v, dx, dy):
        import numexpr as ne
        nx, ny = u.shape
        assert u.shape == tuple(self.shape)

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

        # self.p = np.real(ifft2(p))

        u -= px
        v -= py

        return px, py


def laplace_filter_dct3(n, d=1.0):
    """Return filter of discrete laplacian in DCT-3 space

    This can be used to compute this finite difference quickly

    Examples
    --------

    >>> n = 100
    >>> L = np.diag(np.ones(n-1),-1) + np.diag(np.ones(n-1),+1)-2 * np.diag(np.ones(n),0)
    >>> L[0,0] = -1
    >>> L[-1,-1] = -1
    >>> r = np.random.rand(n)
    >>> lr = L@r
    >>> lrdct = dct(laplace_filter_dct3(n) * idct(r, type=3), type=3)/2/r.shape[0]
    >>> np.testing.assert_allclose(lrdct, lr, atol=1e-10)
    """
    from numpy import pi
    return -4*np.sin(np.arange(n) * pi / 2 / n)**2 /d**2

class CollocatedChannelFDSolver(CollocatedFDSolver):
    """
    u has neumann condition at y boundary
    v has dirichlet condition at y boundary

    periodic in x

    This object works by using even/odd extensions
    """

    def __init__(self, shape, space):
        "docstring"
        nx, ny = shape

        shape = [nx, 2*ny]
        super(CollocatedChannelFDSolver, self).__init__(shape, space)



    def solve(self, u, v, *args):

        _, n = u.shape
        up = np.pad(u, ((0,0), (0,n)), mode='symmetric')
        vp = np.pad(v, ((0,0), (0,n)), mode='symmetric')
        vp[:,n:] *= -1


        ppx, ppy = super(CollocatedChannelFDSolver, self).solve(up, vp, *args)

        px = ppx[:,:n]
        py = ppy[:,:n]

        u -= px
        v -= py

        # self.p = self.p[:,:n]

        return px, py


def solver_error(n, plot=True):
    nx =  n
    ny = n//2
    Lx, Ly = 2*pi, pi

    dx, dy = Lx/nx, Ly/ny

    x, y = np.mgrid[0:Lx:dx, dy/2:Ly:dy]

    u = np.sin(x) * np.cos(y)
    v = 0.0 * u

    solver= CollocatedChannelFDSolver(u.shape, [dx, dy])

    pex = -np.cos(x) * np.cos(y)/2
    pxex = np.sin(x) * np.cos(y)/2

    px, py = solver.solve(u, v, dx, dy)


    # px2, py2 = solver.solve(u, v, dx, dy)

    # Test that solver is actually a projection operator
    # assert norm(px2) < 1e-10
    # assert norm(py2) < 1e-10

    if plot:
        import pylab as pl
        pl.subplot(121)
        pl.pcolormesh(px)
        pl.colorbar()
        pl.subplot(122)
        pl.pcolormesh(pxex)
        pl.colorbar()
        pl.show()

    return np.mean(np.abs(pxex-px))

def test_solver():
    from ..testing import test_convergence

    ns = [16, 32, 64, 128]
    test_convergence(solver_error, ns)


if __name__ == '__main__':
    test_solver(plot=True)
