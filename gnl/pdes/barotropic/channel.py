"""Barotropic 2d dynamics using Chorin's projection method and high resolution
advection schemes

u_t + div(u u)  + f x u = - grad  p
u_x + v_y = 0

"""
import logging
logging.basicConfig(level=logging.INFO)

from numpy import pi, real
import numpy as np

from gnl.pdes.barotropic.barotropic import BarotropicSolver, ChannelSolver#, BetaEffectSolver
from gnl.pdes.fab import BCMultiFab
from gnl.pdes.timestepping import steps
from gnl.pdes.grid import ghosted_grid


class BetaPlaneSolver(ChannelSolver):
    def __init__(self, y):
        "docstring"

        self.y = y


    def coriolis(self, ucv, dt):
        yv = self.y.ghostview[0]
        ucv[0] += yv * ucv[1] * dt
        ucv[1] -= yv * ucv[0] * dt

    def _extra_corrector(self, ucv, dt):
        self.coriolis(ucv, dt)


    def onestep(self, uc, t, dt):
        super(BetaPlaneSolver, self).onestep(uc, t, dt)
        self.coriolis(uc.ghostview, dt/2)

        return uc



def main(plot=True):

    # Setup grid
    g = 4
    nx, ny = 200, 50
    Lx, Ly = 26, 26 * (ny/nx)

    (x, y), (dx, dy) = ghosted_grid([nx, ny], [Lx, Ly], 0)

    y -= Ly/2
    # monkey patch the velocity
    uc = BCMultiFab(sizes=[nx, ny], n_ghost=4, dof=2,
                    bcs=[('wrap', 'even'),
                         ('wrap', 'odd')])
    uc.validview[0] = (y < .5)  * (y > -.5) * 0
    uc.validview[1] = np.sin(5*2 * pi * x/Lx) * .1 * np.exp(-(y/Ly*20)**2) * 10
    # state.u = np.random.rand(*x.shape)

    # tad = ChannelSover()
    yfab = BCMultiFab(sizes=[nx,ny], n_ghost=uc.n_ghost, dof=1)
    yfab.validview[0] = y
    yfab.exchange()
    tad = BetaPlaneSolver(yfab)
    tad.geom.dx = dx
    tad.geom.dy = dx

    dt = min(dx, dy)/1

    if plot:
        import pylab as pl
        pl.ion()

    for i, (t, uc) in enumerate(steps(tad.onestep, uc, dt, [0.0, 10000 * dt])):
        if i % 10 == 0:
            print(i)
            if plot:
                pl.clf()
                pl.pcolormesh(x, y, uc.validview[1])
                pl.colorbar()
                pl.pause(.01)


if __name__ == '__main__':
    main()
