import numpy as np
from petsc4py import PETSc
from numba import jit

from .operators import CollocatedPressureSolver


class PythonCollocatedPressureSolver(CollocatedPressureSolver):

    def get_mat(self):

        n = np.prod(self.da.sizes)
        A = PETSc.Mat().createPython([n, n], comm=self.da.comm)
        pde = Poisson2DX(self.da)
        A.setPythonContext(pde)
        A.setUp()

        return A

@jit
def _kernel_2dx(x, y):
        nx,ny = x.shape
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                y[i-1,j-1] = -4*x[i, j] +x[i-1, j-1] +x[i-1, j+1] +x[i+1, j-1] +x[i+1, j+1] 

class Poisson2DX(object):

    def __init__(self, da):
        assert da.dim == 2
        self.da = da
        self.localX  = da.createLocalVec()


    def mult(self, mat, X, Y):

        self.da.globalToLocal(X, self.localX)

        x = self.da.getVecArray(self.localX)[:]
        y = self.da.getVecArray(Y)[:]

        # # neumann boundary conditions
        # ys, ye = da.ranges[1]

        # ny = da.sizes[1]

        # if ys == 0:
        #     x[:,0] = -x[:, 1]

        # if ye == ny:
        #     x[:,-1] = -x[:, -2]


        _kernel_2dx(x, y)
