"""Barotropic 2d dynamics using finite difference scheme from Arakawa (1966)

"""
from itertools import product
from contextlib import contextmanager
from numpy import pi, real
import numpy as np
import scipy.sparse.linalg as la
import scipy.sparse as ss
from scipy.ndimage import correlate
from timestepping import steps


try:
    from numba import jit
except ImportError:
    def jit(x):
        print("numba not installed. Code will run extremely slowly.")
        return x


@contextmanager
def openlocal(da, global_vecs, yield_numpy=True):
    local_vecs = [da.createLocalVec()]*len(global_vecs)

    for g, l in zip(global_vecs, local_vecs):
        da.globalToLocal(g, l)

    vec_arrays = [da.getVecArray(l) for l in local_vecs]

    if yield_numpy:
        yield tuple(v[:] for v in vec_arrays)
    else:
        yield tuple(vec_arrays)

    for g, l, v in zip(global_vecs, local_vecs, vec_arrays):
        if da.comm.rank > 1:
            da.localToGlobal(l, g)
        else:
            # localToGlobal does not work without parallelism
            sw = da.stencil_width
            idx = [slice(a+sw, b+sw) for a,b in da.ranges]
            g[:] = v[idx]

@contextmanager
def openglobal(da, global_vecs, yield_numpy=True):

    vec_arrays = [da.getVecArray(g) for g in global_vecs]

    if yield_numpy:
        yield tuple(v[:] for v in vec_arrays)
    else:
        yield tuple(vec_arrays)


import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
@jit
def _kernel(x, y, h=1, I=0):
        nx,ny = x.shape
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                u = x[i, j] # center
                u_e = u_w = u_n = u_s = 0
                u_w = x[i-1, j] # west
                u_e = x[i+1, j] # east
                u_s = x[i, j-1] # south
                u_n = x[i, j+1] # north
                u_xx = (u_e - 2*u + u_w)
                u_yy = (u_n - 2*u + u_s)
                y[i-1, j-1] = I * u + h*(u_xx + u_yy)



class Poisson2D(object):

    def __init__(self, da, d=None, I=0, h=1):
        assert da.getDim() == 2
        self.da = da
        self.localX  = da.createLocalVec()
        self.d =d
        self.I = I
        self.h= h


    def mult(self, mat, X, Y):
        #
        self.da.globalToLocal(X, self.localX)

        x = self.da.getVecArray(self.localX)[:]
        y = self.da.getVecArray(Y)[:]
        #
        mx, my = self.da.getSizes()
        hx, hy = [1.0/m for m in [mx, my]]
        (xs, xe), (ys, ye) = self.da.getRanges()

        _kernel(x, y, I=self.I, h=self.h)


OptDB = PETSc.Options()

n  = OptDB.getInt('n', 100)
nx = OptDB.getInt('nx', n)
ny = OptDB.getInt('ny', n)
d  = OptDB.getReal('d', .001)

da = PETSc.DMDA().create([nx, ny], stencil_width=1, boundary_type=[3,3])

def getksp(ns=True, **kwargs):
    pde = Poisson2D(da, **kwargs)
    n = np.prod(da.sizes)

    A = PETSc.Mat().createPython(
        [n, n], comm=da.comm)
    A.setPythonContext(pde)

    # setup null space for periodic simulations
    if ns:
        ns = PETSc.NullSpace().create(constant=True)
        A.setNullSpace(ns)
    A.setUp()

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType('cg')
    pc = ksp.getPC()
    pc.setType('none')
    ksp.setFromOptions()


    return A, ksp





# Setup grid
Lx, Ly = nx * d, ny * d

# ghost cell
g = 1

# make grid
Xa, Ya = da.createGlobalVec(), da.createGlobalVec()
xr, yr = da.ranges

with openglobal(da, [Xa, Ya]) as (x, y):
    x[:] = (np.arange(*xr)*d)[:,None]
    y[:] = (np.arange(*yr)*d)[None,:]


@jit
def J(v, p, jac):
    """Jacobian operator from Arakawa (1966)

    Parameters
    ----------
    p: (nx + 2*g, ny+2*g)
       streamfunction
    v: (nx + 2*g, ny+2*g)
       vorticity

    Returns
    -------
    jacobian
    """

    nx, ny  = v.shape
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            jac[i-1, j-1] = -(1/12/d/d) * (
                (p[i,j-1] + p[i+1,j-1] -p[i,j+1] - p[i+1,j+1])*(v[i+1,j]-v[i,j])
                + (p[i-1,j-1] + p[i,j-1] -p[i-1,j+1] - p[i,j+1])*(v[i,j]-v[i-1,j])
                + (p[i+1,j] + p[i+1,j+1] -p[i-1,j] - p[i-1,j+1])*(v[i,j+1]-v[i,j])
                + (p[i+1,j-1] + p[i+1,j] -p[i-1,j-1] - p[i-1,j])*(v[i,j]-v[i,j-1])
                + (p[i+1,j] - p[i,j+1]) *(v[i+1,j+1] - v[i,j])
                + (p[i,j-1] - p[i-1,j]) *(v[i,j] - v[i-1,j-1])
                + (p[i,j+1] - p[i-1,j]) *(v[i-1,j+1] - v[i,j])
                + (p[i+1,j] - p[i,j-1]) *(v[i,j] - v[i+1,j-1]))


def f(vort, y):

    x = da.createGlobalVec()

    ksp.solve(vort, x)
    vl = da.createLocalVec()
    pl = vl.copy()

    da.globalToLocal(vort, vl)
    da.globalToLocal(x, pl)

    v = da.getVecArray(vl)[:]
    p = da.getVecArray(pl)[:]
    yl = da.getVecArray(y)[:]

    J(v, p, yl)


# adams bashforth
ab = [da.createGlobalVec() for i in range(3)]

def onestep(vort, t, dt, solver=None):
    y = ab.pop(0)
    f(vort, y)

    ab.append(y)

    vort = vort + dt*(23/12*ab[-1] -4/3*ab[-2] + 5/12*ab[-3])
    # vort = vort + dt*(23/12*ab[-1] -4/3*ab[-2] + 5/12*ab[-3])

    # (I - dt L)
    out = vort.copy()
    solver.solve(vort, out)
    # vort = solve_cc(vort, dt/R)
    # vort = vort + dt * y
    return out

vort_strip = lambda y, y0, xw=20:  10*(np.exp(-((y-y0)/(Ly/xw))**2) * (1 + np.sin(2*pi*x/Lx)*.2))


# # vort = 10*(np.exp(-((y-Ly/2)/(Ly/100))**2))
# # vort = np.random.randn(*x.shape)

# p = solve_lapl(vort, A=A, g=g)

vort = da.createGlobalVec()

with openglobal(da, [Xa, Ya, vort]) as (x, y, v):
    v[:] = vort_strip(y, Ly*2/3, xw=50)\
        + -vort_strip(y, Ly*1/3,xw=50)

    # v[:] = np.sin(x * 2 * pi /Lx) * np.cos(4*y * 2 * pi /Ly) * d*d
    # fac = (2*pi/Lx)**2 + (8*pi/Ly)**2

    # pex = -np.sin(x * 2 * pi /Lx) * np.cos(4*y * 2 * pi /Ly) /fac


from functools import partial


import pylab as pl
pl.ion()

dt = d*2

R = 1/d/d


# Pressure Solver
A, ksp = getksp(h=1/d/d)

# Diffusion backward euler
B, kspd = getksp(I=1.0, h=-dt/d/d/R)
onestep=partial(onestep, solver=kspd)

# out = vort.copy()
# kspd.solve(vort, out)

# pl.subplot(211)
# pl.pcolormesh(da.getVecArray(vort)[:]) ; pl.colorbar()
# pl.subplot(212)
# pl.pcolormesh(da.getVecArray(out)[:]) ; pl.colorbar()

# k = input()
# sys.exit(0)
for i, ( t, v ) in enumerate(steps(onestep, vort, dt, [0.0, 10000 * dt])):
    print(t)
    if i%100 == 0:
        pl.clf()
        pl.pcolormesh(da.getVecArray(v)[:])
        pl.colorbar()
        pl.pause(.01)
