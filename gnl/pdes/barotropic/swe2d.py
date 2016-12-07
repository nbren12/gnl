""""2D two mode shallow water dynamics with barotropic mode

TODO
----
- The code in barotropic only convection example seems interesting. Replicate it using the tensor notation. Maybe write a test.
- Rename SWE2NonlinearSolver to TensorNonlinearSolver
- study stability of scheme in advective form

"""
from numpy import pi
import numpy as np
from gnl.pdes.grid import ghosted_grid
from gnl.pdes.timestepping import steps
from gnl.pdes.tadmor.tadmor_2d import MultiFab
from gnl.io import NetCDF4Writer
from gnl.pdes.barotropic.swe import SWE2NonlinearSolver



def main(plot=False):

    # Setup grid
    g = 4
    nx, ny = 200, 200
    Lx, Ly = pi, pi

    (x,y), (dx,dy) = ghosted_grid([nx, ny], [Lx, Ly], 0)


    # tad = SWE2Solver()
    tad = SWE2NonlinearSolver()
    tad.geom.dx = dx
    tad.geom.dy = dx

    uc  = MultiFab(sizes=[nx, ny], n_ghost=4, dof=len(tad.inds))

    # vortex init conds
    # uc.validview[0] = (y > Ly / 3) * (2 * Ly / 3 > y)
    # uc.validview[1]= np.sin(2 * pi * x / Lx) * .3 / (2 * pi / Lx)

    # bubble init conds
    uc.validview[tad.inds.index(('t', 1))] = (((x-Lx/2)**2 + (y-Ly/2)**2  - .5**2 < 0) -.5) * .4

    from scipy.ndimage import gaussian_filter
    uc.validview[:] = gaussian_filter(uc.validview, [0.0, 1.5, 1.5])

    grid = {'x': x[:,0], 'y': y[0,:]}
    writer = NetCDF4Writer(grid, filename="swe2d.nc")

    dt = min(dx, dy) / 4

    if plot:
        import pylab as pl
        import os

        try:
            os.mkdir("_frames")
        except:
            pass
        pl.ion()

    iframe = 0
    for i, (t, uc) in enumerate(steps(tad.onestep, uc, dt, [0.0, 1000 * dt])):
        if (np.any(np.isnan(uc.validview))):
            raise FloatingPointError("NaN in solution array...quitting")
        if i % 20 == 0:
            writer.collect(t, tad.ncvars(uc))
            if plot:
                pl.clf()
                # pl.contourf(uc.validview[tad.inds.index(('t', 1))], np.linspace(-1,2.5,11), cmap='YlGnBu')
                # pl.colorbar()
                # pl.contour(uc.validview[tad.inds.index(('t', 1))], np.linspace(-1,2.5,11), colors='k')
                pl.pcolormesh(uc.validview[tad.inds.index(('u', 1))], vmin=-.5, vmax=1, cmap='YlGnBu_r')
                pl.colorbar()
                pl.savefig("_frames/%04d.png"%iframe)
                iframe +=1
                pl.pause(.01)

if __name__ == '__main__':
    main(plot=False)
