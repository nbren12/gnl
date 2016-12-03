"""Barotropic 2d dynamics using Chorin's projection method

u_t + div(u u)  + f x u = - grad  p
u_x + v_y = 0

this method is extremely dissipative
"""
from numpy import pi, real
import numpy as np
from gnl.pdes.tadmor.tadmor_2d import Tadmor2D
from gnl.pdes.timestepping import steps
from scipy.fftpack import fft2, ifft2, fftfreq

# Setup gridcython -a yourmod.pyx
nx, ny = 200, 200
Lx, Ly = pi, pi

x = np.arange(-2,nx+2)/nx * Lx
y = np.arange(-2,ny+2)/ny * Ly

dx = x[1]-x[0]
dy = y[1]-y[0]

x, y=  np.meshgrid(x, y, indexing='ij')

# initialize state
class State(object):
    v = 0 * x
    u = 0 * x

    def comm():
        pass

    @property
    def uc(self):
        return np.concatenate((self.u[None,...], self.v[None,...]), axis=0)

    @uc.setter
    def uc(self, val):
        self.u = val[0,...]
        self.v = val[1,...]


def fy(uc):

    vort = uc[0]
    u = uc[1]
    v = uc[2]

    f = np.zeros_like(uc)

    f[0] = v * vort

    return f

def fx(uc):

    vort = uc[0]
    u = uc[1]
    v = uc[2]

    f = np.zeros_like(uc)

    f[0] = u * vort

    return f




tad = Tadmor2D()
tad.fx =fx
tad.fy =fy

def advection_step(uc, dt, dx=dx, dy=dy, fx=fx, fy=fy, tad=tad):
    return tad.central_scheme(uc, dx, dy, dt)

def invert_vort(uc, dx=dx, dy=dy, nx=nx, ny=ny, geom=tad.geom):

    ucv = geom.validview(uc)
    vort = ucv[0]
    u = ucv[1]
    v = ucv[2]

    f = fft2(vort)

    nx, ny = vort.shape

    scal_y = 2*pi/dy/ny
    scal_x = 2*pi/dx/nx

    k = fftfreq(nx, 1/nx)[:,None] * 1j * scal_x
    l = fftfreq(ny, 1/ny)[None,:] * 1j * scal_y


    lapl = k**2 + l**2
    lapl[0,0] = 1.0

    psi = f/lapl
    u[:] = -real(ifft2(psi * l))
    v[:] = real(ifft2(psi * k))

    return uc

def onestep(uc, t, dt):
    invert_vort(uc)
    uc = advection_step(uc, dt)
    return uc

vort_strip = lambda y, y0, xw=20:  10*np.exp(-((y-y0)/(xw))**2) 


# # vort = 10*(np.exp(-((y-Ly/2)/(Ly/100))**2))
# # vort = np.random.randn(*x.shape)

# p = solve_lapl(vort, A=A, g=g)
d = dx

vort = (vort_strip(y, Ly*2/3, xw=Lx/50)
    + -vort_strip(y, Ly*1/3,xw=Lx/50)) * ( 1+  np.sin(2*pi*x/Lx)*.3)



import pylab as pl
pl.ion()

uc = np.zeros([3]+list(x.shape))
uc[0]=  vort
invert_vort(uc)

# pu, p = pressure_solve(uc)

# ppu = pressure_solve(pu.copy())
# np.testing.assert_almost_equal(pu, ppu)

# dt = min(dx, dy)/4/np.max(np.abs(uc[1:,...]))
dt = dx/2


pl.pcolormesh(uc[0])
k = input()
for i, ( t, uc ) in enumerate(steps(onestep, uc, dt, [0.0, 1000 * dt])):
    print(t)
    if i%10 == 0:
        pl.clf()
        pl.pcolormesh(tad.geom.validview(uc)[0])
        pl.pause(.01)
