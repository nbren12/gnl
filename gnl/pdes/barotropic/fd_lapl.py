""" Implementation of tadmor scheme using finite difference for the laplace solve




"""
from math import pi
import numpy as np

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from gnl.pdes.barotropic.barotropic import BarotropicSolver
from gnl.pdes.petsc.fab import PETScFab, MultiFab
from gnl.pdes.grid import ghosted_grid
from gnl.pdes.petsc.operators import poisson
from gnl.timestepping import steps


class BarotropicSolverFD(BarotropicSolver):
    pass


def test_fab():
    # Setup grid
    g=2
    nx= 10

    PER = PETSc.DM.BoundaryType.PERIODIC


    da = PETSc.DMDA().create(sizes=[nx], dof=1, stencil_width=g,
                             boundary_type=[PER])
    uc = PETScFab(da)


    uc.g[:] = np.r_[0:nx]
    uc.scatter()

    # uc.exchange()
    assert uc.validview.shape == (10,)
    assert uc.ghostview.shape == (14,)

    np.testing.assert_allclose(uc.validview, np.arange(nx))

def main(plot=True):

    # Setup grid
    g = 4
    nx, ny = 200, 200
    Lx, Ly = pi, pi

    (x,y), (dx,dy) = ghosted_grid([nx, ny], [Lx, Ly], g)

    PER = PETSc.DM.BoundaryType.PERIODIC

    da = PETSc.DMDA().create(sizes=[nx, ny], dof=3, stencil_width=g,
                             boundary_type=[PER, PER])
    da_scalar = da.duplicate(dof=1,
                             stencil_width=1,
                             stencil_type='star',
                             boundary_type=[PER, PER])
    pressure = PETScFab(da_scalar)
    poisson  = Poisson(0.0, 1.0, da_scalar, [dx, dy])


    uc = PETScFab(da)
    ucv = uc.ghostview

    # init
    ucv[0] = (y > Ly / 3) * (2 * Ly / 3 > y)
    ucv[1] = np.sin(2 * pi * x / Lx) * .3 / (2 * pi / Lx)

    tad = BarotropicSolverFD()
    tad.geom.dx = dx
    tad.geom.dy = dx




    # pu, p = pressure_solve(uc)

    # ppu = pressure_solve(pu.copy())
    # np.testing.assert_almost_equal(pu, ppu)

    dt = min(dx, dy) / 4

    if plot:
        import pylab as pl
        pl.ion()

    for i, (t, uc) in enumerate(steps(tad.onestep, uc, dt, [0.0, 10000* dt])):
        if i % 100 == 0:
            if plot:
                pl.clf()
                pl.pcolormesh(uc.validview[0])
                pl.colorbar()
                pl.pause(.01)
if __name__ == '__main__':
    # main(plot=False)
    # main(plot=True)
    test_solver(plot=False)
    # test_fab()
