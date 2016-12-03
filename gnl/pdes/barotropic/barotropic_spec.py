"""Barotropic 2d dynamics using Chorin's projection method

u_t + div(u u)  + f x u = - grad  p
u_x + v_y = 0

this method is extremely dissipative
"""
from numpy import pi, real
import numpy as np
from ..swe.timestepping import steps
from scipy.fftpack import fft2, ifft2, fftfreq

# Setup grid
nx, ny = 400, 400
Lx, Ly = pi, pi

x = np.arange(nx)/nx * Lx
y = np.arange(ny)/ny * Ly

dx = x[1]-x[0]
dy = y[1]-y[0]

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




def f(vort, t, R=1/dx**2, k=k, l=l, lapl=lapl):
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

vort = 10*(np.exp(-((y-Ly/2)/(Ly/100))**2) * (1 + np.sin(x)*.1))
vort = np.random.randn(*x.shape)*4

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
        pl.pause(.01)
