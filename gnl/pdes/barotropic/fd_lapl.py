""" Implementation of tadmor scheme using finite difference for the laplace solve




"""
from math import pi
import numpy as np

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from gnl.pdes.barotropic.barotropic import BarotropicSolver
from gnl.pdes.tadmor.tadmor_2d import MultiFab
from gnl.pdes.grid import ghosted_grid
from gnl.pdes.gallery import poisson
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

    def scatter(self):
        self.da.globalToLocal(self._gvec, self._lvec)

    def gather(self):
        self.g[:] = self.validview.swapaxes(0, -1)

    def exchange(self):
        self.gather()
        self.scatter()

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


class Poisson(object):
    """PETSc Poisson solver

    Operates on PETSc FABs

    Attributes
    ----------
    Mat
    KSP
    da
    """

    def __init__(self, a, b, da, spacing):
        "docstring"
        self.a = a
        self.b = b
        self.da = da

        # get matrix
        self.spacing = spacing
        self.mat = poisson(da, spacing, a, b)

        # constant null space
        ns = PETSc.NullSpace().create(constant=True)
        self.mat.setNullSpace(ns)


        # make solver
        ksp = PETSc.KSP().create()
        ksp.setOperators(self.mat)
        ksp.setType('cg')
        pc = ksp.getPC()
        pc.setType('none')
        ksp.setFromOptions()
        self.ksp = ksp

    def solve(self, b: MultiFab, x: MultiFab):
        """
        """
        b.gather()
        x.gather()
        b._gvec *= self.spacing[0]*self.spacing[1]

        self.ksp.solve(b._gvec, x._gvec)


def test_solver(plot=False):
    nx, ny = 500, 300
    Lx, Ly = 2*pi, 2*pi

    (x,y), (dx,dy) = ghosted_grid([nx, ny], [Lx, Ly], 0)

    PER = PETSc.DM.BoundaryType.PERIODIC

    da = PETSc.DMDA().create(sizes=[nx, ny], dof=1, stencil_width=1,
                             boundary_type=[PER, PER], stencil_type='star')
    soln = PETScFab(da)
    rhs = PETScFab(da)

    rhs.g[:] = np.sin(3*x) * np.cos(2*y)
    rhs.scatter()

    p_ex = rhs.g[:] / (-9 - 4)

    poisson  = Poisson(0.0, 1.0, da, [dx, dy])
    poisson.solve(rhs,soln)


    if plot:
        import pylab as pl
        pl.subplot(131)
        pl.pcolormesh(p_ex)
        pl.colorbar()
        pl.subplot(132)
        pl.pcolormesh(soln.g[:])
        pl.colorbar()
        pl.subplot(133)
        pl.pcolormesh(soln.g[:]-p_ex)
        pl.colorbar()
        pl.show()

    np.testing.assert_allclose(soln.g[:], p_ex, atol=dx**2)

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
