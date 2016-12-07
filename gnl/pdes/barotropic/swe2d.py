""""2D two mode shallow water dynamics with barotropic mode

TODO
----
- The code in barotropic only convection example seems interesting. Replicate it using the tensor notation. Maybe write a test.
- Rename SWE2NonlinearSolver to TensorNonlinearSolver
- study stability of scheme in advective form

"""
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from numpy import pi
import numpy as np
from gnl.pdes.grid import ghosted_grid
from gnl.pdes.timestepping import steps
from gnl.pdes.tadmor.tadmor_2d import MultiFab
from gnl.io import NetCDF4Writer
from gnl.pdes.barotropic.swe import *

from gnl.pdes.barotropic.barotropic import BarotropicSolver
from gnl.pdes.tadmor.tadmor import divergence

import numexpr as ne

def nemult(out, val, a, b, cache={}):
    """
    nemult(fa[i], val, uc[j] , uc[k])
    """
    try:
        intermediate = cache[out.shape]
    except KeyError:
        logger.info("Allocating intermediate array for nemult")
        intermediate = np.empty_like(out)
        cache[out.shape] = intermediate

    ne.evaluate("val * a* b", out=intermediate)
    out[:] += intermediate


class ixer(object):
    def __init__(self, uc, inds):
        self.uc = uc
        self.inds = inds

    def __getitem__(self, key):
        return self.uc[self.inds.index(key)]

    def __setitem__(self, key, val):
        self.uc[self.inds.index(key)] = val

class SWE2Solver(BarotropicSolver):
    """Solver for two mode shallow water dynamics on a double periodic domain

    uc = [u_0, v_0, u_1, u_2, v_1, v_2, t_1, t_2]
    """
    inds = [('u',0), ('v',0) ,('u',1), ('u',2), ('v', 1), ('v', 2), ('t',1), ('t', 2)]

    def ix(self, uc):
        return ixer(uc, self.inds)

    def fx(self, fa, uc, dim='x'):

        fa[:] = 0.0

        q = self.ix(uc)
        f = self.ix(fa)

        if dim == 'x':
            adv = 'u'
        else:
            adv = 'v'


        # barotropic nonlinearity
        for i in range(3):
            f['u', 0] += q[adv, i] * q['u', i]
            f['v', 0] += q[adv, i] * q['v', i]

        # Advection by barotropic flow
        for v, i in product(['u', 'v'], range(1,3)):
            f[v, i] += q[adv, 0] * q[v, i]

        for  i in range(1,3):
            f['t', i] += q[adv, 0] * q['t', i] * i*i

        f[adv, 1] -= q['t', 1]
        f[adv, 2] -= q['t', 2]

        # f['v', 1] -= 0
        # f['v', 2] -= 0

        f['t', 1] -= q[adv, 1]
        f['t', 2] -= q[adv, 2]/4


    def fy(self, fa, uc):
        return self.fx(fa, uc, dim='y')

    def ncvars(self, uc):
        get_name = lambda name, m: "{}{}".format(name, m)
        for i, namet in enumerate(self.inds):
            yield get_name(*namet), ('x', 'y'), uc.validview[i]

class SWE2NonlinearSolver(SWE2Solver):

    ntrunc = 2
    def __init__(self):
        "docstring"
        from math import sqrt, pi
        from galerkin import flux_div_t, flux_div_u, sparsify

        # get coefficients
        au, bu = flux_div_u()
        at, bt = flux_div_t()

        self._au = sparsify(au)
        self._bu = sparsify(bu)
        self._at = sparsify(at)
        self._bt = sparsify(bt)


        # these setting replicate the barotropic only class
        # self._au = [((0, 0,0), 1.0),
        #             ((0, 1,1), 1.0),
        #             ((0,2,2), 1.0),
        #             ((1,0,1), 1.0),
        #             ((2,0,2), 1.0)]

        # self._at = [((1,0,1), 1.0),
        #             ((2, 0, 2), 1.0)]
        # self._bu = []
        # self._bt = []
    def flux_spec(self, dim):
        """Specification of the flux  terms

        These terms are given by
        ((adv_vel)_j phi_k)_i
        """

        if dim == 'x':
            adv = 'u'
        elif dim == 'y':
            adv = 'v'
        else:
            raise ValueError("Invalid dimension description")

        out = []

        # u advection terms
        for (i,j,k),val in self._au:
            for var in ['u', 'v']:
                i1 = self.inds.index((var, i))
                j1 = self.inds.index((adv, j))
                k1 = self.inds.index((var, k))
                out.append(((i1, j1, k1), val))

        # temperature advection terms
        for (i,j,k), val in self._at:
            i1 = self.inds.index(('t', i))
            j1 = self.inds.index((adv, j))
            k1 = self.inds.index(('t', k))
            out.append(((i1, j1, k1), val))

        return out

    def source_spec(self):
        """Specification of the split-in-time source terms

        These terms have are given by the form:
            (- phi_k div u_j )_i
        """


        out = []

        # u,v terms
        for (i,j,k),val in self._bu:
            for var in ['u', 'v']:
                i1 = self.inds.index((var, i))
                j1 = j
                k1 = self.inds.index((var, k))
                out.append(((i1, j1, k1), val))

        # temperature terms
        for (i,j,k), val in self._bt:
            i1 = self.inds.index(('t', i))
            j1 = j
            k1 = self.inds.index(('t', k))
            out.append(((i1, j1, k1), val))

        return out

    def fx(self, fa, uc, dim='x'):

        fa[:] = 0

        if dim == 'x':
            adv = 'u'
        else:
            adv = 'v'

        # provide access to variables by name
        q = self.ix(uc)
        f = self.ix(fa)

        for (i, j, k), val in self.flux_spec(dim):
            nemult(fa[i], val, uc[j] , uc[k])

        for i in range(1,self.ntrunc+1):
            f['t', i] -= q[adv, i]
            f['t', i] /= i * i
            f[adv, i] -= q['t', i]

    def _extra_corrector(self, uc, dt):
        super(SWE2NonlinearSolver, self)._extra_corrector(uc, dt)
        # predictor corrector
        f = self.split_terms(uc)
        uc[:] += f*dt

    def split_terms(self, uc):

        dx, dy = self.geom.dx, self.geom.dy

        fa = np.zeros_like(uc)

        q = self.ix(uc)

        # from scipy.ndimage import correlate1d
        # w = {j:(correlate1d(q['u', j], [1/2/dx, 0, -1/2/dx], origin=0, axis=0)
        #         + correlate1d(q['v', j], [1/2/dy, 0, -1/2/dy], origin=0, axis=1))
        #      for j in range(1,self.ntrunc +1)}
        w = {j:divergence(q['u', j], q['v',j], dx, dy)
             for j in range(1,self.ntrunc +1)}

        for (i,j,k), val in self.source_spec():
             nemult(fa[i], val, w[j] , uc[k])

        return fa

    def advection_step(self, uc, dt):
        self.central_scheme(uc, self.geom.dx, self.geom.dy, dt)

        #predictor corrector
        uc.exchange()
        uc0 = uc.ghostview.copy()
        f = self.split_terms(uc.ghostview)
        uc.ghostview[:] += f*dt/2

def main(plot=False):

    # Setup grid
    g = 4
    nx, ny = 1000, 250
    nx, ny = 512, 128
    Lx, Ly = 26, 26 * (ny/nx)

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
