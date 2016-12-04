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
from gnl.pdes.petsc.operators import CollocatedPressureSolver
from gnl.timestepping import steps


class BarotropicSolverFD(BarotropicSolver):
    def __init__(self, da, h, *args, **kwargs):
        "docstring"
        super(BarotropicSolverFD, self).__init__( *args, **kwargs)
        self._pressure_solver = CollocatedPressureSolver(h, da)

        self.geom.dx, self.geom.dy = h, h

        # initialize pressure gradient
        da_gp=  da.duplicate(dof=2, stencil_type='box', stencil_width=1)
        self.gp = PETScFab(da_gp)

    def pressure_solve(self, uc):
        self._pressure_solver.project(uc, self.gp)


def main():

    opts = PETSc.Options()
    plot = opts.getBool("plot", False)

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
                             stencil_type='box',
                             boundary_type=[PER, PER])


    uc = PETScFab(da)
    ucv = uc.ghostview

    # init
    ucv[0] = (y > Ly / 3) * (2 * Ly / 3 > y)
    ucv[1] = np.sin(2 * pi * x / Lx) * .3 / (2 * pi / Lx)

    tad = BarotropicSolverFD(da_scalar, dx)




    # pu, p = pressure_solve(uc)

    # ppu = pressure_solve(pu.copy())
    # np.testing.assert_almost_equal(pu, ppu)

    dt = min(dx, dy) / 4

    if plot:
        import pylab as pl
        pl.ion()

    for i, (t, uc) in enumerate(steps(tad.onestep, uc, dt, [0.0, 100* dt])):
        print(i)
        if i % 100 == 0:
            if plot:
                pl.clf()
                pl.pcolormesh(uc.validview[0])
                pl.colorbar()
                pl.pause(.01)
if __name__ == '__main__':
    # main(plot=False)
    main()
    # test_fab()
