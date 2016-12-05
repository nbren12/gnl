import numpy as np
from numpy import sin, cos, pi

from .barotropic import BarotropicSolver, MultiFab, ghosted_grid
from ..timestepping import steps

plot = False

def taylor_vortex(x, y):
    return cos(x) * sin(y), -sin(x)*cos(y)

def error(x,y):
    return np.mean(np.abs(x-y))

def taylor_vortex_error(time_end, n):

    # Setup grid
    g = 4
    nx = ny = n
    Lx= Ly = 2*pi

    (x,y), (dx,dy) = ghosted_grid([nx, ny], [Lx, Ly], 0)

    u0, v0 = taylor_vortex(x, y)

    # monkey patch the velocity
    uc  = MultiFab(sizes=[nx, ny], n_ghost=4, dof=2)
    uc.validview[0] = u0
    uc.validview[1]=  v0
    # state.u = np.random.rand(*x.shape)

    tad = BarotropicSolver()
    tad.geom.dx = dx
    tad.geom.dy = dx


    dt = min(dx, dy) / 4

    for i, (t, uc) in enumerate(steps(tad.onestep, uc, dt, [0.0, time_end])):
        pass

    if plot:
        import pylab as pl
        pl.clf()
        pl.subplot(121)
        pl.pcolormesh(uc.validview[0])
        pl.title("approx soln")
        pl.colorbar()

        pl.subplot(122)
        pl.pcolormesh(u0)
        pl.title("exact soln")
        pl.colorbar()
        pl.show()

    approx_soln = np.hstack((uc.validview[0], uc.validview[1]))
    exact_soln = np.hstack((u0, v0))
    return error(approx_soln, exact_soln)

def test_taylor_convergence():

    nlist = [50, 100, 200, 400]

    err = np.array([taylor_vortex_error(1.0, n) for n in nlist])
    p = np.polyfit(np.log(nlist), np.log(err), 1)

    print("")
    print("Convergence test results")
    print("----------------------------")
    print("Grid Sizes: "+repr(nlist))
    print("Errors: "+repr(err))
    print("Ratio: "+repr(err[:-1]/err[1:]))
    print("Order of convergence:" +repr(-p[0]))
    print("")

    if plot:
        import matplotlib.pyplot as plt
        plt.loglog(nlist, err)
        plt.title('Order of convergence p = %.2f'%p[0])
        plt.show()

    if  -p[0] < 1.9:
        raise ValueError('Order of convergence (p={p})is less than 2'.format(p=-p[0]))


if __name__ == '__main__':
    test_taylor_convergence()
