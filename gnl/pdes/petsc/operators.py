"""Linear Operators and Solvers using petsc4py

this module provides some convenience functions and classes for using PETSc's
linear solvers.

"""
from itertools import product
from math import pi
import numpy as np

from petsc4py import PETSc
from .fab import MultiFab, PETScFab

from .kernel import div_kernel

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
        b.gvec *= self.spacing[0]*self.spacing[1]

        return b

    def solve(self, b: MultiFab, x: MultiFab):
        b = self.get_rhs(b)
        self.ksp.solve(b.gvec, x.gvec)


class CollocatedPressureSolver(object):
    """PETSc Poisson solver


    Assume that the data uses a constant grid spacing
    """

    def __init__(self, h, da):
        "docstring"

        self.h = h
        self.da= da

        if self.da.stencil_type != PETSc.DMDA.StencilType.BOX:
            raise ValueError("Need DMDA with box stencil-type")

        # initialize fab for the divergence
        self.div = PETScFab(self.da)

        # initialize fab for the pressure
        self.pres = PETScFab(self.da)

        # Get the linear operator
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
        """This returns an x-like stencil

        """
        da = self.da

        A = da.createMatrix()

        # use stencil to set entries
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()


        if da.dim == 2:
            ir = range(*da.ranges[0])
            jr = range(*da.ranges[1])

            for i, j in product(ir, jr):
                row.index = (i,j)

                for index, value in [((i, j),  -4 ),
                                    ((i-1, j-1), 1),
                                    ((i-1, j+1), 1),
                                    ((i+1, j-1), 1),
                                    ((i+1, j+1), 1)]:
                    col.index = index
                    A.setValueStencil(row, col, value)
        else:
            raise NotImplementedError("This class only works with two dimensional data")

        A.assemblyBegin()
        A.assemblyEnd()

        return A

    def compute_div(self, uc):
        uc.exchange()
        ucv = uc.ghostview

        div_kernel(ucv[0], ucv[1], self.div.ghostview, self.h)
        self.div.gather()

        return self.div.gvec

    def compute_pressure(self, uc):
        div = self.compute_div(uc)

        self.ksp.solve(div, self.pres.gvec)


    def pressure_grad(self, uc):
        pass


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
