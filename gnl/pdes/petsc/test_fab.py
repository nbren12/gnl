"""Tests for petsc convenienc functions
"""
import numpy as np
from .fab import PETSc, PETScFab
from .operators import pi, Poisson, CollocatedPressureSolver

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


def test_solver(plot=False):
    nx, ny = 500, 300
    Lx, Ly = 2*pi, 2*pi

    dx, dy = nx/Lx, ny/Ly

    x, y = np.mgrid[0:Lx:dx,0:Ly:dy]

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

def test_collocated_solver(plot=False):

    nx= ny = 500
    Lx, Ly = 2*pi, 2*pi

    dx, dy = Lx/nx, Ly/ny

    x, y = np.mgrid[0:Lx:dx,0:Ly:dy]

    PER = PETSc.DM.BoundaryType.PERIODIC

    # Need box stencil
    da = PETSc.DMDA().create(sizes=[nx, ny], dof=2, 
                             boundary_type=[PER, PER],
                             stencil_width=1,
                             stencil_type='box')

    uc = PETScFab(da)

    uc.validview[0] = np.sin(2*x)
    uc.validview[1] = np.sin(y)

    da_scalar=  da.duplicate(dof=1,
                             stencil_type='box',
                             stencil_width=1)

    pressure = PETScFab(da_scalar)

    solver = CollocatedPressureSolver(dx, da_scalar)
    solver.solve(uc, pressure)

    p_ex = -np.cos(2*(x-dx/2))/2 + -np.cos(y-dy/2)


    if plot:
        import pylab as pl
        pl.subplot(121)
        pl.pcolormesh(pressure.g[:].T)
        pl.colorbar()
        pl.subplot(122)
        pl.pcolormesh(p_ex)
        pl.colorbar()
        pl.show()

    np.testing.assert_allclose(p_ex, pressure.g[:].T, atol=dx**2)
