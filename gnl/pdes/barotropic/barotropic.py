"""Barotropic 2d dynamics using Chorin's projection method and high resolution
advection schemes

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

from gnl.pdes.tadmor.tadmor_2d import Tadmor2D, MultiFab
from gnl.pdes.timestepping import steps
from gnl.pdes.grid import ghosted_grid


def _solve_laplace(uv, dx, dy):
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

    px = real(ifft2(p*k))
    py = real(ifft2(p*l))

    u[:] -= px
    v[:] -= py

    return px, py

class BarotropicSolver(Tadmor2D):

    def init_pgrad(self, uc):
        self.pg = MultiFab(data=uc.data.copy(), n_ghost=uc.n_ghost, dof=2)

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
        self.central_scheme(uc, self.geom.dx, self.geom.dy, dt)

    def pressure_solve(self, uc):

        dx = self.geom.dx
        dy = self.geom.dy

        uv = uc.validview

        self.pg.validview[0], self.pg.validview[1] = _solve_laplace(uv, dx, dy)

    def _extra_corrector(self, uc):
        self.pg.exchange()
        uc[:2,...] += self.pg.ghostview

    def onestep(self, uc, t, dt):
        try:
            pg = self.pg
        except AttributeError:
            self.init_pgrad(uc)
        self.pressure_solve(uc)
        self.advection_step(uc, dt)

        return uc

def main(plot=True):

    # Setup grid
    g = 4
    nx, ny = 200, 200
    Lx, Ly = pi, pi

    (x,y), (dx,dy) = ghosted_grid([nx, ny], [Lx, Ly], 0)


    # monkey patch the velocity
    uc  = MultiFab(sizes=[nx, ny], n_ghost=4, dof=2)
    uc.validview[0] = (y > Ly / 3) * (2 * Ly / 3 > y)
    uc.validview[1]= np.sin(2 * pi * x / Lx) * .3 / (2 * pi / Lx)
    # state.u = np.random.rand(*x.shape)

    tad = BarotropicSolver()
    tad.geom.dx = dx
    tad.geom.dy = dx


    dt = min(dx, dy) / 4

    if plot:
        import pylab as pl
        pl.ion()

    for i, (t, uc) in enumerate(steps(tad.onestep, uc, dt, [0.0, 10000 * dt])):
        if i % 100 == 0:
            if plot:
                pl.clf()
                pl.pcolormesh(uc.validview[0])
                pl.colorbar()
                pl.pause(.01)


if __name__ == '__main__':
    main()
