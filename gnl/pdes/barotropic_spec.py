"""Barotropic 2d dynamics using pseudo-spectral method

"""
from numpy import pi, real
import numpy as np
from .timestepping import steps
from scipy.fftpack import fft2, ifft2, fftfreq

# Setup grid
dx, dy =  .01/1.5, .01/1.5
nx, ny = 150, 150
Lx, Ly = dx*nx, dy*ny

x = np.arange(nx)/nx * Lx
y = np.arange(ny)/ny * Ly


x, y=  np.meshgrid(x, y, indexing='ij')


# used for difference
scal_y = 2*pi/dy/ny
scal_x = 2*pi/dx/nx

k = fftfreq(nx, 1/nx)[:,None] * 1j * scal_x
l = fftfreq(ny, 1/ny)[None,:] * 1j * scal_y

lapl = k**2 + l**2
lapl[0,0] = 1.0

# filter
filter23 = (np.abs(k) < 2/3 * nx/2) * (np.abs(l) < 2/3 * ny/2)

def lapl_solve(vort):

    fv = fft2(vort)
    psi = fv/lapl

    return real(ifft2(psi))


def f(vort, t, R=1/dx**2/4, k=k, l=l, lapl=lapl):
    fv = fft2(vort)
    psi = fv/lapl

    u = ifft2(-psi * l)
    v = ifft2(psi * k)

    adv = -(u* ifft2(fv*k) + v*ifft2(fv*l)) + ifft2(lapl * fv/R)

    return adv

ab = [0]*3

def onestep(vort, t, dt):
    adv = f(vort, t)
    ab.append(adv); ab.pop(0)

    vort = vort + dt*(23/12*ab[-1] -4/3*ab[-2] + 5/12*ab[-3])
    vort = real(ifft2(filter23*fft2(vort)))

    return vort


vort_strip = lambda y, y0, xw=20:  10*(np.exp(-((y-y0)/(Ly/xw))**2) * (1 + np.sin(2*pi*x/Lx)*.2))

vort = vort_strip(y, Ly*2/3, xw=50)\
       + -vort_strip(y, Ly*1/3,xw=50)


# vort = np.random.randn(*x.shape)*4

import pylab as pl
pl.ion()


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
        pl.colorbar()
        pl.pause(.01)
