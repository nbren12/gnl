"""Barotropic 2d dynamics using Chorin's projection method and high resolution
advection schemes

u_t + div(u u)  + f x u = - grad  p
u_x + v_y = 0

"""
import logging
logging.basicConfig(level=logging.INFO)

from numpy import pi, real
import numpy as np

from gnl.pdes.tadmor.tadmor_2d import Tadmor2D, MultiFab
from gnl.pdes.timestepping import steps
from gnl.pdes.grid import ghosted_grid
from gnl.pdes.barotropic.pressure_solvers import CollocatedFDSolver, CollocatedChannelFDSolver


class BarotropicSolver(Tadmor2D):

    pres_solver_cls = CollocatedFDSolver

    def init_pgrad(self, uc):
        self.pg = MultiFab(data=uc.data.copy(), n_ghost=uc.n_ghost, dof=2)
        self.pg.ghostview[:] = 0.0

        self.pressure_solver = self.pres_solver_cls(
            uc.validview.shape[1:], [self.geom.dx, self.geom.dy])

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

        px, py = self.pressure_solver.solve(uv[0], uv[1], dx, dy)

        self.pg.validview[0] = px
        self.pg.validview[1] = py

    def _extra_corrector(self, uc, dt):
        self.pg.exchange()
        uc[0:2, ...] -= self.pg.ghostview / 2

    def onestep(self, uc, t, dt):
        try:
            pg = self.pg
        except AttributeError:
            self.init_pgrad(uc)
        self.advection_step(uc, dt)
        self.pressure_solve(uc)

        return uc

    @property
    def bcs(self):
        return None

class ChannelSolver(BarotropicSolver):
    pres_solver_cls = CollocatedChannelFDSolver

    @property
    def bcs(self):
        return [('wrap', 'even'),
                ('wrap', 'odd')]



def main(plot=True):

    # Setup grid
    g = 4
    nx, ny = 200, 50
    Lx, Ly = 26, 26/4

    (x, y), (dx, dy) = ghosted_grid([nx, ny], [Lx, Ly], 0)

    # monkey patch the velocity
    uc = MultiFab(sizes=[nx, ny], n_ghost=4, dof=2)
    uc.validview[0] = (y > Ly / 3) * (2 * Ly / 3 > y)
    uc.validview[1] = np.sin(2 * pi * x / Lx) * .3 / (2 * pi / Lx)
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
