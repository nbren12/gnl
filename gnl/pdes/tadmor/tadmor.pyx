#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
from numpy.lib.stride_tricks import as_strided
from cython.parallel cimport prange

cimport cython


def inplacewrapper(fun):

    def f(x, *args, **kwargs):
        uy = np.empty_like(x)
        fun(uy, x, *args, **kwargs)
        return uy
    return f



cdef double minmod(double a,double b,double c) nogil:
    """Compute minmod of a variadic list of arguments

    Note
    ----
    This function requires at least numba v.28
    """
# by hand bubble sort
    if a > b : a,b = b,a
    if b > c : b,c = c,b
    if a > b : a, b = b, a


    if a > 0.0:
        return a
    elif c < 0.0:
        return c
    else:
        return 0.0

def _slopes(uy, uc, axis=-1, tht=None, limiter=None):
    """Calculate slopes

    Parameters
    ----------
    limiter: seq of int
        turns on/off limiter
    tht:
        parameter to flux limiter
    """

    cdef int j, i, k, neq, n1, n2
    uc1 = uc.swapaxes(axis, -1);

    neq, n1, n2 = uc1.shape
    cdef double[:,:,:] ucv = uc1
    cdef double[:,:,:] uyv = uy.swapaxes(axis, -1);


    if tht is None:
        tht = np.ones(neq)*2.0

    if limiter is None:
        limiter = np.ones(neq, dtype=np.int32)

    cdef double[:] thtv = tht
    cdef int[:] limv = limiter


    cdef double left, cent, right
    with nogil:
        for j in prange(neq):
            for i in range(n1):
                for k in range(1, n2 - 1):
                    if limv[j]:
                        left = thtv[j] * (ucv[j, i, k + 1] - ucv[j, i, k])
                        cent = (ucv[j, i, k + 1] - ucv[j, i, k - 1]) / 2
                        right = thtv[j] * (ucv[j, i, k] - ucv[j, i, k - 1])
                        uyv[j, i, k] = minmod(left, cent, right)
                    else:
                        cent = (ucv[j, i, k + 1] - ucv[j, i, k - 1]) / 2
                        uyv[j, i, k] = cent


@cython.boundscheck(False)
def _corrector_step(double[:,:,:] out, double[:,:,:] fx ,
                    double[:,:,:] fy, double lmd_x, double lmd_y):

    cdef int i, j, k
    with nogil:

        for i in prange(out.shape[0]):
            for j in range(out.shape[1]-1):
                for k in range(out.shape[2]-1):
                    out[i,j+1,k+1] += (fx[i,j,k] - fx[i,j+1,k]
                                    - fx[i,j+1,k+1] + fx[i,j,k+1]) * lmd_x/2 \
                                    +(fy[i,j,k]-fy[i,j,k+1]
                                    -fy[i,j+1,k+1] + fy[i,j+1,k]) * lmd_y/2

@inplacewrapper
def divergence(double[:,:] out, double[:,:] u, double[:,:] v,
               double dx, double dy):


    cdef double hx2 = 2 * dx
    cdef double hy2 = 2 * dy
    cdef int i,j
    with nogil:
        for i in prange(1, u.shape[0]-1):
            for j in range(1, u.shape[1]-1):
                out[i,j] = (-u[i+1,j]+u[i-1,j])/hx2 \
                           +(v[i,j-1] - v[i,j+1])/hy2

def _roll2d(u):
    return np.roll(np.roll(u, -1, axis=1), -1, axis=2)

cdef class Tadmor2DBase:
    cdef int initialized


    def __cinit__(self):
        self.initialized = False

    def init(self, uc):
        self.ustag = uc.copy()
        self.ux = uc.copy()
        self.uy = uc.copy()
        self.fxa = uc.copy()
        self.fya = uc.copy()
        self.initialized = True

    def fx(self, fx, uc):
        raise NotImplementedError

    def fy(self, fy, uc):
        raise NotImplementedError

    def _extra_corrector(self, uc, dt):
        pass

    def _single_step(self, uc, dx, dy, dt):

        ux = np.zeros_like(uc)
        uy = np.zeros_like(uc)
        uc = uc.copy()
        lmd_x = dt / dx
        lmd_y = dt / dy

        self.ustag[:] = 0
        self.fxa[:] = 0
        self.fya[:] = 0
        self.ux[:] = 0
        self.uy[:] = 0

        self._stagger_avg(self.ustag, uc)

        # predictor: mid-time-step pointewise values at cell-center
        # Eq. (1.1) in Jiand and Tadmor
        self.fx(self.fxa, uc)
        self.fy(self.fya, uc)
        _slopes(self.ux, self.fxa, axis=1)
        _slopes(self.uy, self.fya, axis=2)
        self._extra_corrector(uc, dt/2)
        uc -= lmd_x / 2 * self.ux + lmd_y / 2 * self.uy

        # corrector
        # Eq (1.2) in Jiang and Tadmor
        # self.fill_boundaries(uc)
        self.fxa[:] = 0
        self.fya[:] = 0
        self.fx(self.fxa, uc)
        self.fy(self.fya, uc)
        _corrector_step(self.ustag, self.fxa, self.fya, lmd_x, lmd_y)

        return self.ustag

    def central_scheme(self, uc, dx, dy, dt):
        """ One timestep of centered scheme


        Parameters
        ----------
        fx : callable
            fx(u) calculates the numeric flux in the x-direction
        uc: (neq, n)
            The state vector on the centered grid
        dx: float
            size of grid cell
        dt: float
            Time step

        Returns
        -------
        out: (neq, n)
        state vector on centered grid
        """

        if not self.initialized:
            self.init(uc)

        self._stagger_avg(uc, _roll2d(self._single_step(uc, dx, dy, dt)))
        return uc

    cdef _stagger_avg(self, avg, uci):
        _slopes(self.ux, uci, axis=1)
        _slopes(self.uy, uci, axis=2)

        cdef double[:,:,:] ux = self.ux
        cdef double[:,:,:] uc = uci
        cdef double[:,:,:] uy = self.uy
        cdef double[:,:,:] out = avg

        cdef int i, j, k
        with nogil:
            for i in prange(ux.shape[0]):
                for j in range(ux.shape[1]-1):
                    for k in range(ux.shape[2]-1):
                        out[i,j+1,k+1] = (uc[i,j,k] + uc[i,j+1,k] + uc[i,j,k+1] + uc[i,j+1,k+1])/4 \
                                        +((ux[i,j,k] - ux[i,j+1,k]) + (ux[i,j,k+1] -ux[i,j+1,k+1])
                                        + (uy[i,j,k] - uy[i,j,k+1]) +(uy[i,j+1,k] - uy[i,j+1,k+1])
                                        ) / 16
