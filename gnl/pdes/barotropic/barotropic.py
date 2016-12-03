"""Barotropic 2d dynamics using Chorin's projection method

u_t + div(u u)  + f x u = - grad  p
u_x + v_y = 0
"""
import logging
logging.basicConfig(level=logging.INFO)

from numpy import pi, real
import numpy as np


try:
    import pyfftw

    fft2 = pyfftw.interfaces.scipy_fftpack.fft2
    ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
    fftfreq = pyfftw.interfaces.scipy_fftpack.fftfreq
    pyfftw.interfaces.cache.enable()
    logging.info("using pyfftw")
except ImportError:
    from scipy.fftpack import fft2, ifft2, fftfreq
    logging.info("using scipt fftpack")

from gnl.pdes.tadmor.tadmor_2d import Tadmor2D
from gnl.pdes.timestepping import steps
from gnl.pdes.grid import ghosted_grid


# initialize state
class State(object):
    def comm():
        pass

    @property
    def uc(self):
        return np.concatenate((self.u[None, ...], self.v[None, ...]), axis=0)

    @uc.setter
    def uc(self, val):
        self.u = val[0, ...]
        self.v = val[1, ...]


class BarotropicSolver(Tadmor2D):
    def fx(self, uc):
        u = uc[0]
        v = uc[1]

        f = np.empty_like(uc)

        for i in range(f.shape[0]):
            f[i] = u * uc[i]

        return f

    def fy(self, uc):
        u = uc[0]
        v = uc[1]

        f = np.empty_like(uc)

        for i in range(f.shape[0]):
            f[i] = v * uc[i]

        return f

    def advection_step(self, uc, dt):
        return self.central_scheme(uc, self.geom.dx, self.geom.dy, dt)

    def pressure_solve(self, uc):

        uv = self.geom.validview(uc)
        dx = self.geom.dx
        dy = self.geom.dy

        nx, ny = uv.shape[1:]

        u = uv[0]
        v = uv[1]

        fu = fft2(u)
        fv = fft2(v)

        scal_y = 2 * pi / dy / ny
        scal_x = 2 * pi / dx / nx

        k = fftfreq(nx, 1 / nx)[:, None] * 1j * scal_x
        l = fftfreq(ny, 1 / ny)[None, :] * 1j * scal_y

        lapl = k**2 + l**2
        lapl[0, 0] = 1.0

        p = (fu * k + fv * l) / lapl

        u[:] -= real(ifft2(p * k))
        v[:] -= real(ifft2(p * l))

        return uc, real(ifft2(p))

    def onestep(self, uc, t, dt):
        uc = self.advection_step(uc, dt / 2)
        uc, p = self.pressure_solve(uc)

        return uc

def main(plot=True):

    # Setup grid
    g = 4
    nx, ny = 200, 200
    Lx, Ly = pi, pi

    (x,y), (dx,dy) = ghosted_grid([nx, ny], [Lx, Ly], g)


    state = State()
    # monkey patch the velocity
    state.u = (y > Ly / 3) * (2 * Ly / 3 > y)
    state.v = np.sin(2 * pi * x / Lx) * .3 / (2 * pi / Lx)
    # state.u = np.random.rand(*x.shape)

    tad = BarotropicSolver()
    tad.geom.dx = dx
    tad.geom.dy = dx
    tad.geom.n_ghost = g

    uc = state.uc

    # pu, p = pressure_solve(uc)

    # ppu = pressure_solve(pu.copy())
    # np.testing.assert_almost_equal(pu, ppu)

    dt = min(dx, dy) / 4

    if plot:
        import pylab as pl
        pl.ion()

    for i, (t, uc) in enumerate(steps(tad.onestep, uc, dt, [0.0, 10000 * dt])):
        if i % 100 == 0:
            if plot:
                pl.clf()
                pl.pcolormesh(tad.geom.validview(uc)[0])
                pl.colorbar()
                pl.pause(.01)


if __name__ == '__main__':
    main()
