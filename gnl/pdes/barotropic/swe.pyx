#cython: boundscheck=False, wraparound=False, nonecheck=False
from itertools import product
import numpy as np
import numexpr as ne

from .barotropic import BarotropicSolver
from ..tadmor.tadmor import divergence


from cython.parallel cimport prange, parallel
cimport openmp
from libc.stdio cimport printf

cdef nemult(outpi, double a, bi, ci):

    cdef double[:] outp = outpi.ravel()
    cdef double[:] c = ci.ravel()
    cdef double[:] b = bi.ravel()


    cdef int i

    for i in prange(outp.shape[0], nogil=True):
        outp[i]  += a * b[i] *c[i]


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

    def fx(self, fa, uc, dim='x'):

        fa[:] = 0
        q = self.ix(uc)
        f = self.ix(fa)

        if dim == 'x':
            adv = 'u'
        else:
            adv = 'v'

        for (i, j, k), val in self._au:
            nemult(f['u', i], val, q[adv, j] , q['u', k])
            nemult(f['v', i], val, q[adv, j] , q['v', k])

        for (i, j, k), val in self._at:
            nemult(f['t', i], val , q[adv, j] , q['t', k])

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
        f = self.ix(fa)

        # from scipy.ndimage import correlate1d
        # w = {j:(correlate1d(q['u', j], [1/2/dx, 0, -1/2/dx], origin=0, axis=0)
        #         + correlate1d(q['v', j], [1/2/dy, 0, -1/2/dy], origin=0, axis=1))
        #      for j in range(1,self.ntrunc +1)}
        w = {j:divergence(q['u', j], q['v',j], dx, dy)
             for j in range(1,self.ntrunc +1)}


        for (i, j, k), val in self._bu:
            nemult(f['u', i], val, w[j] , q['u', k])
            nemult(f['v', i], val, w[j] , q['v', k])

        for (i, j, k), val in self._bt:
            nemult(f['t', i], val , w[j] , q['t', k])


        return fa

    def advection_step(self, uc, dt):
        self.central_scheme(uc, self.geom.dx, self.geom.dy, dt)

        #predictor corrector
        uc.exchange()
        uc0 = uc.ghostview.copy()
        f = self.split_terms(uc.ghostview)
        uc.ghostview[:] += f*dt/2
