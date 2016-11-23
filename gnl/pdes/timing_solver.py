"""Barotropic 2d dynamics using finite difference scheme from Arakawa (1966)

"""
from itertools import product
from numpy import pi, real
import numpy as np
import scipy.sparse.linalg as la
import scipy.sparse as ss

try:
    from numba import jit
except ImportError:
    def jit(x):
        print("numba not installed. Code will run extremely slowly.")
        return x

from petsc4py import PETSc


from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

# Setup grid
d = .01
nx, ny = 100, 100
Lx, Ly = nx * d, ny * d

# ghost cell
g = 1

# make grid
x = np.arange(-g, nx+g)*d
y = np.arange(-g, ny+g)*d

dx = x[1]-x[0]
dy = y[1]-y[0]

x, y=  np.meshgrid(x, y, indexing='ij')



def build_laplacian_matrix(nx, ny, g=g,d=d):
    Lx = ss.diags([-2*np.ones(nx), np.ones(nx-1), np.ones(nx-1), 1.0, 1.0],
                  [0, -1, 1, nx-1, -(nx-1)])/d/d

    Ly = ss.diags([-2*np.ones(ny), np.ones(ny-1), np.ones(ny-1), 1.0, 1.0],
                  [0, -1, 1, ny-1, -(ny-1)])/d/d


    L =   ss.kronsum(Ly, Lx).tocsr()
    I = ss.eye(nx*ny).tocsr()
    L[:,0] = 0.0
    L[0] = I[0]

    return L




A = build_laplacian_matrix(nx, ny, d=d)
I = ss.eye(nx*ny)


def solve_lapl(v, A=A, g=g):
    vgn = v[g:-g,g:-g]
    vr = vgn.ravel()

    # this step is sooo important
    # takes care of gauge invariance
    vr[0] = 0.0


    out = np.zeros_like(v)
    out[g:-g,g:-g] = la.spsolve(A, vr).reshape(vgn.shape)

    return out


def test_laplacian():

    import pylab as pl
    # Setup grid
    nx, ny = 100, 200
    d = 2*pi/nx
    Lx, Ly = nx * d, ny * d


    g = 0
    # make grid
    x = np.arange(-g, nx+g)*d
    y = np.arange(-g, ny+g)*d

    dx = x[1]-x[0]
    dy = y[1]-y[0]

    x, y =  np.meshgrid(x, y, indexing='ij')

    # build laplacian
    A = build_laplacian_matrix(nx,ny,d=d)

    # right hand side
    f = np.sin(x)*np.cos(2*y)

    p_ex = np.sin(x)*np.cos(2*y)/(-1 - 4)

    print("Timing information")
    print("==================")
    print("")


    with elapsed_timer() as elapsed:
        p_ap = la.spsolve(A, f.ravel())
        print("spsolve {0}".format(elapsed()))

    with elapsed_timer() as elapsed:
        p_cg = la.cg(A, f.ravel())[0]
        print("cg {0}".format(elapsed()))

    with elapsed_timer() as elapsed:
        p_ap = la.gmres(A, f.ravel())
        print("gmres {0}".format(elapsed()))

    pl.pcolormesh(p_cg.reshape(f.shape))
    pl.show()
test_laplacian()
