import pytest
import numpy as np
from numpy import sin, cos, pi

from .barotropic import BarotropicSolver, ghosted_grid,\
    ChannelSolver
from ..fab import BCMultiFab
from ..timestepping import steps

plot = False
PRINT_LATEX = True

def taylor_vortex(x, y):
    return sin(x) * cos(y), -cos(x)*sin(y)

def error(x,y):
    return np.mean(np.abs(x-y))

def taylor_vortex_error(time_end, n, solver=BarotropicSolver):

    # Setup grid
    g = 4
    nx = ny = n
    Lx= Ly = 2*pi

    (x,y), (dx,dy) = ghosted_grid([nx, ny], [Lx, Ly], 0)

    u0, v0 = taylor_vortex(x, y)

    #
    tad = solver()
    tad.geom.dx = dx
    tad.geom.dy = dx

    # monkey patch the velocity
    uc  = BCMultiFab(sizes=[nx, ny], n_ghost=4, dof=2,
                     bcs=tad.bcs)
    uc.validview[0] = u0
    uc.validview[1]=  v0
    # state.u = np.random.rand(*x.shape)



    dt = min(dx, dy) / 4

    for i, (t, uc) in enumerate(steps(tad.onestep, uc, dt, [0.0, time_end])):
        if np.any(np.isnan(uc.validview)):
            raise FloatingPointError("NAN in solution array!")
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

@pytest.mark.parametrize("solver", [BarotropicSolver, ChannelSolver])
def test_taylor_convergence(solver):
    from ..testing import test_convergence

    nlist = [50, 100, 200, 400]

    f = lambda n: taylor_vortex_error(1.0, n,solver)
    test_convergence(f, nlist)


if __name__ == '__main__':
    test_taylor_convergence()
