""""2D one mode shallow water dynamics with barotropic mode

u0_t + div u_1 u_1 + u_0 u_0 + f . u0= - grad p0
u_t + div(u0 u_1) = - grad b
b_t + div(u0 t_1) - div(u1)  = 0

"""
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from numpy import pi
import numpy as np
from ..fab import BCMultiFab
from ..grid import ghosted_grid
from ..timestepping import steps
from ..tadmor.tadmor_2d import MultiFab, convergence
from ...io import NetCDF4Writer
from .barotropic import BarotropicSolver
from .swe import *


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
    u_nums = [0, 1]
    t_nums = [1]

    def ix(self, uc):
        return ixer(uc, self.inds)


    def fy(self, fa, uc):
        return self.fx(fa, uc, dim='y')


    def fx(self, fa, uc, dim='x'):

        fa[:] = 0

        if dim == 'x':
            adv = 'u'
        else:
            adv = 'v'

        # provide access to variables by name
        q = self.ix(uc)
        f = self.ix(fa)

        # temperature fluxes
        f['t', 1] += q[adv, 1]
        f['t', 1] += q[adv, 0] * q['t', 1]

        # velocity fluxes

        for (i, j, k), val in self.flux_spec(dim):
            nemult(fa[i], val, uc[j] , uc[k])

        # TODO specify these terms using a linear_spec property
        # like we do for the nonlinear terms
        for i in range(1,self.ntrunc+1):
            f['t', i] += q[adv, i]
            f['t', i] /= i * i
            f[adv, i] += q['t', i]


    @property
    def ncvars_stacked(self, uc):
        variables = list(self.inds)

        q = self.ix(uc.validview)
        for v in ['u', 'v']:
            catme = []
            for i in self.u_nums:
                catme.append(q[v, i][None,...])
                variables.remove((v, i))
            yield v, ('m', 'x', 'y'), np.concatenate(catme, axis=0)

        catme = []
        for i in self.t_nums:
            catme.append(q['t', i][None,...])
            variables.remove(('t', i))
        yield 't', ('n', 'x', 'y'), np.concatenate(catme, axis=0)

        for namet, i in variables:
            if namet not in ['u', 'v', 't']:
                yield namet, ('x', 'y'), uc.validview[i]


    def ncvars(self, uc):
        """Yields a tuples of names, dimensions, and data

        this list is used for as input to NetCDF4Writer.

        Notes
        -----
        see ncvars_stacked for outut which stacks the baroclinic modes into 3d
        netcdf variables.
        """
        get_name = lambda name, m: "{}{}".format(name, m)
        for i, namet in enumerate(self.inds):
            yield get_name(*namet), ('x', 'y'), uc.validview[i]

class SWE2NonlinearSolver(SWE2Solver):
    """Shallow water wave solver with nonlinear

    The coefficients are obtained from a galerkin projection procedure, which is carried out using numerical quadrature in galerkin.py.

    Notes
    -----

    This class does not currently work with bases other than the sin basis. The
    convention in the notes is to express the tempearture in terms of phi''(z)
    where phi(z) is the vertical velocity mode. Therefore the basis functions
    for temperature have the opposite sign as velocity. 
    """

    ntrunc = 2
    n_ghost = 4
    bcs = None
    def __init__(self, sizes):
        "docstring"
        from math import sqrt, pi
        from .galerkin import flux_div_t, flux_div_u, sparsify

        # get coefficients
        au, bu = flux_div_u()
        at, bt = flux_div_t()

        self._au = sparsify(au)
        self._bu = sparsify(bu)
        self._at = sparsify(at)
        self._bt = sparsify(bt)


        self.sizes= sizes

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


    def init_uc(self):
        # create initial data
        uc  = BCMultiFab(sizes=self.sizes,
                       n_ghost=self.n_ghost,
                       dof=len(self.inds),
                       bcs=self.bcs)

        return uc

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

            # the -val is because we want temperature to have the right sign
            # see the note in the docstring of this class.
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
            # the -val is because we want temperature to have the right sign
            # see the note in the docstring of this class.
            out.append(((i1, j1, k1), val))

        return out


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
        w = {j:convergence(q['u', j], q['v',j], dx, dy)
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

class BetaPlaneMixin(object):
    def coriolis(self, ucv, dt):
        yv = self.y.ghostview[0]

        q = self.ix(ucv)
        for i in range(self.ntrunc):
            q['u', i ] += yv * ucv['v', i] * dt
            q['v', i ] -= yv * ucv['u', i] * dt


class BetaPlaneSWESolver(SWE2NonlinearSolver,BetaPlaneMixin):
    def __init__(self,  y):
        "docstring"
        super(BetaPlaneSWESolver, self).__init__(y.shape)

        # create initial data
        yfab = BCMultiFab(sizes=y.shape, dof=1,
                          n_ghost=self.n_ghost)

        yfab.validview[0] = y
        yfab.exchange()

        self.y = yfab

    @property
    def bcs(self):
        out = []
        for v,_ in self.inds:
            if v == 'u':
                # neumann for u
                out.append(('wrap','even'))
            elif v =='v':
                # dirichlet for v
                out.append(('wrap','odd'))
            else:
                out.append(('wrap', 'even'))
        return out


    def coriolis(self, ucv, dt):
        yv = self.y.ghostview[0]

        q = self.ix(ucv)
        for i in range(self.ntrunc):
            q['u', i ] += yv * q['v', i] * dt
            q['v', i ] -= yv * q['u', i] * dt

    def _extra_corrector(self, ucv, dt):
        self.coriolis(ucv, dt)

    def onestep(self, uc, t, dt):
        super(BetaPlaneSWESolver, self).onestep(uc, t, dt)
        self.coriolis(uc.ghostview, dt/2)

        return uc

def main(plot=False, problem='dam_break'):




    if problem == 'beta_plane':
        # Setup grid
        g = 4
        nx, ny = 1000, 250
        nx, ny = 512, 128
        Lx, Ly = 26, 26 * (ny/nx)

        (x,y), (dx,dy) = ghosted_grid([nx, ny], [Lx, Ly], 0)

        x-= Lx/2
        y-= Ly/2

        tad = BetaPlaneSWESolver(y)
        tad.geom.dx = dx
        tad.geom.dy = dx
        uc  = tad.init_uc()

        # # vortex init conds
        # uc.validview[0] = (y > Ly / 3) * (2 * Ly / 3 > y)
        # uc.validview[1]= np.sin(2 * pi * x / Lx) * .3 / (2 * pi / Lx)

        # vortex conditions
        L= 1
        psi = np.exp(-(x**2 + y**2)/(L)**2/2)
        q = tad.ix(uc.validview)
        q['u', 0] = x/ L**2* psi
        q['v', 0] = -y/L**2 * psi
        q['t', 1] = -psi/5

        tend = 5


    elif problem == 'dam_break':
        ## Radial dam break problem
        nx, ny = 256, 256
        Lx, Ly = pi, pi

        (x,y), (dx,dy) = ghosted_grid([nx, ny], [Lx, Ly], 0)
        x-= Lx/2
        y-= Ly/2

        tad = SWE2NonlinearSolver([nx, ny])
        tad.geom.dx = dx
        tad.geom.dy = dx
        uc = tad.init_uc()

        t0 = ((x**2 + y**2  - .5**2 < 0)) * .5


        # bubble init conds
        uc.validview[tad.inds.index(('t', 1))] = -t0

        tend = pi


    from scipy.ndimage import gaussian_filter
    uc.validview[:] = gaussian_filter(uc.validview, [0.0, 1.5, 1.5])

    grid = {'x': x[:,0], 'y': y[0,:],
            'm': tad.u_nums,
            'n': tad.t_nums}
    writer = NetCDF4Writer(grid, filename="swe2d.nc")






    dt = min(dx, dy) / 4

    if plot:
        import pylab as pl

    iframe = 0
    for i, (t, uc) in enumerate(steps(tad.onestep, uc, dt, [0.0, tend])):
        if (np.any(np.isnan(uc.validview))):
            raise FloatingPointError("NaN in solution array...quitting")
        if i % 20 == 0:
            writer.collect(t, tad.ncvars(uc))
            if plot:
                pl.clf()
                # pl.contourf(uc.validview[tad.inds.index(('t', 1))], np.linspace(-1,2.5,11), cmap='YlGnBu')
                # pl.colorbar()
                # pl.contour(uc.validview[tad.inds.index(('t', 1))], np.linspace(-1,2.5,11), colors='k')
                pl.pcolormesh(-uc.validview[tad.inds.index(('t', 1))], vmin=-.5, vmax=1, cmap='YlGnBu_r')
                pl.colorbar()
                iframe +=1
                pl.pause(.01)

if __name__ == '__main__':
    main(plot=True)
