"""Barotropic 2d dynamics using Chorin's projection method

u_t + div(u u)  + f x u = - grad  p
u_x + v_y = 0
"""
from numpy import pi, real
import numpy as np
from ..tadmor.tadmor_2d import Tadmor2D
from ..swe.timestepping import steps
import logging
logging.basicConfig(level=logging.INFO)
try:
    import pyfftw

    fft2 = pyfftw.interfaces.scipy_fftpack.fft2
    ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
    fftfreq = pyfftw.interfaces.scipy_fftpack.fftfreq
    pyfftw.interfaces.cache.enable()
    logging.info("using pyfftw")
except:
    from scipy.fftpack import fft2, ifft2, fftfreq
    logging.info("using scipt fftpack")

# Setup grid
g = 4
nx, ny = 200, 200
Lx, Ly = pi, pi

x = np.arange(-g,nx+g)/nx * Lx
y = np.arange(-g,ny+g)/ny * Ly

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



def fx(uc):
    u = uc[0]
    v = uc[1]

    f = np.empty_like(uc)

    for i in range(f.shape[0]):
        f[i] = u*uc[i]

    return f

def fy(uc):
    u = uc[0]
    v = uc[1]

    f = np.empty_like(uc)

    for i in range(f.shape[0]):
        f[i] = v*uc[i]

    return f

def comm(uc):
    """doubly periodic for now"""
    from .tadmor.tadmor_common import periodic_bc

    return periodic_bc(uc, g=2, axes=(1,2))


tad = Tadmor2D()
tad.n_ghost = g
tad.fx= fx
tad.fy= fy
tad.comm = comm

def advection_step(uc, dt, dx=dx, dy=dy, tad=tad):
    return tad.central_scheme(uc, dx, dy, dt)

def pressure_solve(uc, dx=dx, dy=dy, nx=nx, ny=ny):
    u = uc[0,g:-g,g:-g]
    v = uc[1,g:-g,g:-g]

    fu = fft2(u)
    fv = fft2(v)

    scal_y = 2*pi/dy/ny
    scal_x = 2*pi/dx/nx

    k = fftfreq(nx, 1/nx)[:,None] * 1j * scal_x
    l = fftfreq(ny, 1/ny)[None,:] * 1j * scal_y


    lapl = k**2 + l**2
    lapl[0,0] = 1.0


    p  = (fu * k + fv * l)/lapl

    u -= real(ifft2(p * k))
    v -= real(ifft2(p * l))

    uc[0,g:-g,g:-g] = u
    uc[1,g:-g,g:-g] = v

    return uc, real(ifft2(p))



def onestep(uc, t, dt):
    uc = advection_step(uc, dt/2)
    uc, p = pressure_solve(uc)


    return uc

def main(plot=True):

    state = State()
    state.u = (y>Ly/3) *(2*Ly/3 > y)
    state.v = np.sin(2*pi*x/Lx) * .3 / (2*pi/Lx)
    # state.u = np.random.rand(*x.shape)
    if plot:
        import pylab as pl
        pl.ion()

    uc = state.uc

    # pu, p = pressure_solve(uc)

    # ppu = pressure_solve(pu.copy())
    # np.testing.assert_almost_equal(pu, ppu)

    dt = min(dx, dy)/4


    if plot:
        pl.pcolormesh(uc[0])
        k = input()
    for i, ( t, uc ) in enumerate(steps(onestep, uc, dt, [0.0, 10000 * dt])):
        if i%100 == 0:
            if plot:
                pl.clf()
                pl.pcolormesh(uc[0])
                pl.colorbar()
                pl.pause(.01)
if __name__ == '__main__':
    main()
