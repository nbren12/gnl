""" Python implementation of the Tadmor centered scheme in 1d


Routines
----------------
periodic_bc - periodic boundary condition in arbitrary dimensions
central_scheme - 1d implementation of tadmor centered scheme

"""
import numpy as np
from numpy import pi
from ..testing import test_convergence
from ..fab import MultiFab
from .tadmor import Tadmor1DBase


class AdvectionSolver(Tadmor1DBase):
    def fx(self, ret, uc):
        ret[:] = 1.0


def tadmor_error(n):
    uc = np.zeros((1, n + 4))

    L = 1.0
    dx = L / n

    x = np.arange(-2, n + 2) * dx

    u0 = np.sin(2 * pi * x)**10

    # data needs to have three dimensions
    # so pad with singletons
    uc = u0[None,:,None]

    # get solver
    solver = AdvectionSolver()

    dt = dx / 2

    def fx(u):
        return u

    tend = .87
    t = 0
    while (t < tend - 1e-10):
        dt = min(dt, tend - t)
        uc = solver.central_scheme(uc, dx, dt)
        t += dt

    uexact = np.sin(2 * pi * (x - t))**10

    return np.sum(np.abs((uc[0, :] - uexact))) / n


def test_tadmor_convergence(plot=False):
    """
    Create error convergence plots for 1d advection problem
    """
    nlist = [50, 100, 200, 400, 800, 1600]

    test_convergence(tadmor_error, nlist)



def plot_tadmor_1d(n=2000):
    """
    scalar advection for tadmor scheme
    """
    import matplotlib.pyplot as plt
    uc = np.zeros((1, n + 4))

    L = 1.0
    dx = L / n

    x = np.arange(-2, n + 2) * dx

    uc[0, :] = np.exp(-((x - .5) / .10)**2)

    dt = dx / 2

    def fx(u):
        return u

    plt.plot(x, uc[0, :], label='exact')

    tend = 1.8
    t = 0
    while (t < tend - 1e-10):
        dt = min(dt, tend - t)
        uc = central_scheme(fx, uc, dx, dt)
        t += dt

    plt.plot(x, uc[0, :], label='tadmor', c='k')


def plot_upwind_1d(n=2000):
    """
    scalar advection for upwinding scheme
    """
    import matplotlib.pyplot as plt
    uc = np.zeros((1, n + 4))

    L = 1.0
    dx = L / n

    x = np.arange(-2, n + 2) * dx

    uc[0, :] = np.exp(-((x - .5) / .10)**2)

    dt = dx / 2

    tend = 1.8
    t = 0
    while (t < tend - 1e-16):
        dt = min(tend - t, dt)
        uc += dt / dx * (np.roll(uc, 1, axis=-1) - uc)
        t += dt

    plt.plot(x, uc[0, :], label='upwind')


def compare_upwind_tadmor():
    import matplotlib.pyplot as plt

    n = 200
    plot_tadmor_1d(n)
    plot_upwind_1d(n)
    plt.legend()
    plt.show()
