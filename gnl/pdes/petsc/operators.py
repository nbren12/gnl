from itertools import product
from math import pi
import numpy as np

from petsc4py import PETSc
from .fab import MultiFab, PETScFab

def poisson(da: PETSc.DM, spacing=None, a=0.0, b=1.0, A: PETSc.Mat = None,
            bcs=None) -> PETSc.Mat:
    """
    Return second order matrix for problem

    a I + b div grad

    Note
    ----
    this implementation is pretty slow. Can it be speeded up with cython somehow
    """
    if A is None:
        A = da.createMatrix()

    if spacing is None:
        spacing = [1.0] * da.dim

    if bcs is None:
        bcs = [0]*da.dim

    # use stencil to set entries
    row = PETSc.Mat.Stencil()
    col = PETSc.Mat.Stencil()


    if da.dim == 2:
        dx, dy = spacing

        hxy = dx/dy
        hyx = dy/dx
        ir = range(*da.ranges[0])
        jr = range(*da.ranges[1])

        dx, dy = spacing
        for i, j in product(ir, jr):
            row.index = (i,j)

            for index, value in [((i, j), a + b * (-2 * hyx - 2 *hxy)),
                                 ((i-1, j), b * hyx),
                                 ((i+1, j), b * hyx),
                                 ((i, j-1), b*hxy),
                                 ((i, j+1), b*hxy)]:
                col.index = index
                A.setValueStencil(row, col, value)

    A.assemblyBegin()
    A.assemblyEnd()

    return A


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
        self.mat = self.get_mat()

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

    def get_mat(self):
        return poisson(self.da, self.spacing, self.a, self.b)

    def get_rhs(self, b: MultiFab):
        b.gather()
        b._gvec *= self.spacing[0]*self.spacing[1]

        return b

    def solve(self, b: MultiFab, x: MultiFab):
        b = self.get_rhs(b)
        self.ksp.solve(b._gvec, x._gvec)


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

if __name__ == '__main__':
    import timeit
    from functools import partial
    da = PETSc.DMDA().create(sizes=[300, 300],
                             stencil_width=1,
                             stencil_type='star',
                             boundary_type=['periodic', 'periodic'])
    A = da.createMatrix()
    f =partial(poisson, da, A=A)

    time = timeit.timeit(f, number=10)/10
    print(da.comm.rank, time)
