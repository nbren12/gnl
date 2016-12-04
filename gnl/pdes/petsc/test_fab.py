"""Tests for petsc convenienc functions
"""
import numpy as np
from .fab import PETSc, PETScFab
from .operators import pi, Poisson, CollocatedPressureSolver

plot = False

if plot:
    import pylab as pl

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

    # test view functions
    np.testing.assert_allclose(uc.view(1), np.r_[9,0:10,0])
    np.testing.assert_allclose(uc.view(2), np.r_[8,9,0:10,0,1])


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

def test_collocated_solver():

    nx= ny = 500
    Lx, Ly = 2*pi, 2*pi

    dx, dy = Lx/nx, Ly/ny

    x, y = np.mgrid[0:Lx:dx,0:Ly:dy]

    PER = PETSc.DM.BoundaryType.PERIODIC

    # Need box stencil
    # make the pressure and velocity have different stencil_widths
    da = PETSc.DMDA().create(sizes=[nx, ny], dof=2,
                             boundary_type=[PER, PER],
                             stencil_width=3,
                             stencil_type='box')

    uc = PETScFab(da)

    uc.validview[0] = np.sin(2*x)
    uc.validview[1] = np.sin(y)

    da_gp=  da.duplicate(dof=2,
                             stencil_type='box',
                             stencil_width=1)

    da_scalar=  da.duplicate(dof=1,
                             stencil_type='box',
                             stencil_width=1)

    pressure = PETScFab(da_scalar)

    solver = CollocatedPressureSolver(dx, da_scalar)
    solver.compute_pressure(uc)
    pressure = solver.pres

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

    np.testing.assert_allclose(p_ex, pressure.globalview, atol=dx**2)


    # get pressure correction
    # this step computes pressure and projects the velocity fields
    gp = PETScFab(da_gp)
    solver.project(uc, gp)
    if plot:
            import pylab as pl
            pl.subplot(121)
            pl.pcolormesh(gp.globalview[0])
            pl.colorbar()
            pl.subplot(122)
            pl.pcolormesh(gp.globalview[1])
            pl.colorbar()
            pl.show()

    # Test pressure gradient
    px_ex = np.sin(2*x)
    np.testing.assert_allclose(px_ex, gp.globalview[0], atol=dx**2)
    py_ex = np.sin(y)
    np.testing.assert_allclose(py_ex, gp.globalview[1], atol=dx**2)

    np.testing.assert_allclose(uc.globalview, 0, atol=1e-6)

    # test that the pressure projection is indeed a projection operator
    # another projection step should not alter the velocities
    uc0 = uc.globalview.copy()
    solver.project(uc, gp)

    if plot:
        pl.pcolormesh(uc0[0] - uc.globalview[0])
        pl.colorbar()
        pl.show()

    np.testing.assert_allclose(uc0, uc.globalview, atol=1e-6)

if __name__ == '__main__':
    test_collocated_solver()
