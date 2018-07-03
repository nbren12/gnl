# kursiv.m - solution of Kuramoto-Sivashinsky equation by ETDRK4 scheme
#
# u_t = -u*u_x - u_xx - u_xxxx, periodic BCs on [0,32*pi]
# computation is based on v = fft(u), so linear term is diagonal
# compare p27.m in Trefethen, "Spectral Methods in MATLAB", SIAM 2000
# AK Kassam and LN Trefethen, July 2002
# Spatial grid and initial condition:
from scipy.fftpack import fft, ifft, fftfreq
from numpy import cos, sin, pi, exp
import numpy as np
import torch


tmax = 1000
N = 128
L = 32
d = L / N
x = L * np.r_[:N]/N
u = cos(x * L / 2)*(1+sin(x * L / 2))
v = fft(u)

# Precompute various ETDRK4 scalar quantities:
h = 0.25  # time step
# nplt = np.floor((tmax/100)/h)
nplt = 1

k = fftfreq(N, d=d)
# k = [0:N/2-1 0 -N/2+1:-1]’/16 # wave numbers

L = k**2 - k**4  # Fourier multipliers
E = exp(h*L)
E2 = exp(h*L/2)
M = 16  # no. of points for complex means
r = exp(1j*pi*(np.arange(1, M+1)-.5)/M)  # roots of unity
LR = h*L[:, None] + r[None, :]

Q = h*np.mean((exp(LR/2)-1) / LR, axis=1).real
f1 = h*np.mean((-4-LR+exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1).real
f2 = h*np.mean((2+LR+exp(LR)*(-2+LR))/LR**3, axis=1).real
f3 = h*np.mean((-4-3*LR-LR**2+exp(LR)*(4-LR))/LR**3, axis=1).real

# Main time-stepping loop:
uu = []
tt = []

nmax = round(tmax/h)

g = -0.5j*k
for n in range(nmax):
    t = n*h
    Nv = g*fft(np.real(ifft(v))**2)
    a = E2*v + Q*Nv
    Na = g*fft(np.real(ifft(a))**2)
    b = E2*v + Q*Na
    Nb = g*fft(np.real(ifft(b))**2)
    c = E2*a + Q*(2*Nb-Nv)
    Nc = g*fft(np.real(ifft(c))**2)
    v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3

    if n % nplt == 0:
        u = ifft(v).real
        uu.append(u)
        tt.append(t)


torch.save(uu, "x.pt")

# import matplotlib.pyplot as plt
# plt.contourf(uu)
# plt.show()

# Plot results:
# surf(tt,x,uu), shading interp, lighting phong, axis tight
# view([-90 90]), colormap(autumn) set(gca,’zlim’,[-5 50])
# light(’color’,[1 1 0],’position’,[-1,2,2
