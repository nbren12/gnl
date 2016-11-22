"""Barotropic 2d dynamics using finite difference scheme from Arakawa (1966)

"""
from itertools import product
from numpy import pi, real
import numpy as np
import scipy.sparse.linalg as la
import scipy.sparse as ss
from scipy.ndimage import correlate
from .timestepping import steps
from .bc import periodic_bc


try:
    from numba import jit
except ImportError:
    def jit(x):
        print("numba not installed. Code will run extremely slowly.")
        return x

# Setup grid
d = .01
nx, ny = 200, 200
Lx, Ly = nx * d, ny * d

# ghost cell
g = 1

# make grid
x = np.arange(-g, nx+g)*d
y = np.arange(-g, ny+g)*d

dx = x[1]-x[0]
dy = y[1]-y[0]

x, y=  np.meshgrid(x, y, indexing='ij')

@jit
def J(v, p, nx=nx, ny=ny, d=d, g=g):
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

    jac = np.zeros_like(p)

    for i in range(g,nx+g):
        for j in range(g,ny+g):
            jac[i, j] = -(1/12/d/d) * (
                (p[i,j-1] + p[i+1,j-1] -p[i,j+1] - p[i+1,j+1])*(v[i+1,j]-v[i,j])
                + (p[i-1,j-1] + p[i,j-1] -p[i-1,j+1] - p[i,j+1])*(v[i,j]-v[i-1,j])
                + (p[i+1,j] + p[i+1,j+1] -p[i-1,j] - p[i-1,j+1])*(v[i,j+1]-v[i,j])
                + (p[i+1,j-1] + p[i+1,j] -p[i-1,j-1] - p[i-1,j])*(v[i,j]-v[i,j-1])
                + (p[i+1,j] - p[i,j+1]) *(v[i+1,j+1] - v[i,j])
                + (p[i,j-1] - p[i-1,j]) *(v[i,j] - v[i-1,j-1])
                + (p[i,j+1] - p[i-1,j]) *(v[i-1,j+1] - v[i,j])
                + (p[i+1,j] - p[i,j-1]) *(v[i,j] - v[i+1,j-1]))


    return jac

def apply_lapl(vg, d=d):
    weights = np.array([[0, 1.0, 0],
                        [1, -4, 1],
                        [0, 1, 0]])/d/d

    return correlate(vg, weights, origin=-1, mode='wrap')


def build_laplacian_matrix(nx, ny):
    Lx = ss.diags([-2*np.ones(nx), np.ones(nx-1), np.ones(nx-1), 1.0, 1.0],
                  [0, -1, 1, (nx-1),-(nx-1)])

    Ly = ss.diags([-2*np.ones(ny), np.ones(ny-1), np.ones(ny-1), 1.0, 1.0],
                  [0, -1, 1, ny-1,-(ny-1)])

    return ss.kronsum(Ly, Lx)


A = build_laplacian_matrix(*x.shape)/d/d

def invert_vort(v, d=d, g=g):
    """Vorticity inversion
    """

    sh = v.shape
    return la.spsolve(A, v.ravel()).reshape(sh)

def f(vort,R=1/d/d):

    periodic_bc(vort, g=1, axes=(0,1))
    psi = invert_vort(vort)

    # arakawa scheme
    periodic_bc(psi, g=1, axes=(0,1))
    y = J(vort, psi)

    # explicit diffusion on grid scale
    y += apply_lapl(vort)/R


    return y


# adams bashforth
ab = [0]*3

def onestep(vort, t, dt):
    adv = f(vort)
    ab.append(adv); ab.pop(0)
    vort = vort + dt*(23/12*ab[-1] -4/3*ab[-2] + 5/12*ab[-3])
    return vort

vort = 10*(np.exp(-((y-Ly/2)/(Ly/100))**2) * (1 + np.sin(x)*.1))
vort = np.random.randn(*x.shape)*4


import pylab as pl
pl.ion()


def test_laplacian():

    import pylab as pl
    # Setup grid
    nx, ny = 200, 200
    d = 2*pi/nx
    Lx, Ly = nx * d, ny * d


    # make grid
    x = np.arange(0, nx)*d
    y = np.arange(0, ny)*d

    dx = x[1]-x[0]
    dy = y[1]-y[0]

    x, y =  np.meshgrid(x, y, indexing='ij')

    # build laplacian
    A = build_laplacian_matrix(*x.shape)/d/d

    # right hand side
    f = np.sin(x)*np.cos(2*y)

    p_ex = np.sin(x)*np.cos(2*y)/(-1 - 4)

    p_ap = la.spsolve(A, f.ravel()).reshape(*f.shape)

    pl.subplot(211)
    pl.pcolormesh(p_ex)
    pl.colorbar()

    pl.subplot(212)
    pl.pcolormesh(p_ap)
    pl.colorbar()

    pl.show()
    input()

test_laplacian()
    


# pu, p = pressure_solve(uc)

# ppu = pressure_solve(pu.copy())
# np.testing.assert_almost_equal(pu, ppu)

dt = dx


pl.pcolormesh(vort)
k = input()
for i, ( t, v ) in enumerate(steps(onestep, vort, dt, [0.0, 10000 * dt])):
    print(t)
    if i%20 == 0:
        pl.clf()
        pl.pcolormesh(v)
        pl.pause(.01)
