import petsc4py
from petsc4py import PETSc
from itertools import product, starmap

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
