""" Implementation of tadmor scheme using finite difference for the laplace solve




"""
from math import pi
import numpy as np

import petsc4py
from petsc4py import PETSc

from gnl.pdes.barotropic.barotropic import BarotropicSolver
from gnl.pdes.tadmor.tadmor_2d import MultiFab
from gnl.pdes.grid import ghosted_grid
from gnl.timestepping import steps

class BarotropicSolverFD(BarotropicSolver):
    pass


class PETScFab(MultiFab):

    def __init__(self, da):
        "docstring"
        self.da = da
        self._lvec = self.da.createLocalVec()
        self._gvec = self.da.createGlobalVec()

    @property
    def n_ghost(self):
        return self.da.stencil_width

    @property
    def ghostview(self):
        return self.l[:].swapaxes(-1, 0)

    @property
    def validview(self):
        """It is more convenient to make the components the first dimension

        np.swapaxes should not change the underlying memory structures.
        """

        inds = []
        l = self.l
        for (beg, end), start in zip(self.da.ranges, l.starts):
            inds.append(slice(beg-start, end-start))
        return self.l[inds][:].swapaxes(-1, 0)

    @property
    def l(self):
        return self.da.getVecArray(self._lvec)

    @property
    def g(self):
        return self.da.getVecArray(self._gvec)

    def swap(self):
        self.da.globalToLocal(self._gvec, self._lvec)

    def exchange(self):
        self.g[:] = self.validview.swapaxes(0, -1)
        self.swap()

def test_fab():
    # Setup grid
    g=2
    nx= 10

    PER = PETSc.DM.BoundaryType.PERIODIC


    da = PETSc.DMDA().create(sizes=[nx], dof=1, stencil_width=g,
                             boundary_type=[PER])
    uc = PETScFab(da)


    uc.g[:] = np.r_[0:nx]
    uc.swap()

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
    da_scalar = PETSc.DMDA().create(sizes=[nx, ny], dof=1, stencil_width=g)



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
    import sys
    petsc4py.init(sys.argv)
    # main(plot=False)
    main(plot=True)
    # test_fab()
