#cython: boundscheck=False
import numpy as np
from numpy.lib.stride_tricks import as_strided
from cython.parallel cimport prange

cimport cython
cimport numpy as np


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

@inplacewrapper
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
    neq, n1, n2 = uc.shape


    cdef double[:,:,:] ucv = uc.swapaxes(axis, -1);
    cdef double[:,:,:] uyv = uy.swapaxes(axis, -1);


    if tht is None:
        tht = np.ones(neq)*2.0

    if limiter is None:
        limiter = np.zeros(neq, dtype=np.int32)

    cdef double[:] thtv = tht
    cdef int[:] limv = limiter


    cdef double left, cent, right
    with nogil:
        for j in range(neq):
            for i in prange(n1):
                for k in range(1, n2 - 1):
                    if limv[j]:
                        left = thtv[j] * (ucv[j, i, k + 1] - ucv[j, i, k])
                        cent = (ucv[j, i, k + 1] - ucv[j, i, k - 1]) / 2
                        right = thtv[j] * (ucv[j, i, k] - ucv[j, i, k - 1])
                        uyv[j, i, k] = minmod(left, cent, right)
                    else:
                        cent = (ucv[j, i, k + 1] - ucv[j, i, k - 1]) / 2
                        uyv[j, i, k] = cent

@inplacewrapper
def _stagger_avg(avg, uci):
    uxi = _slopes(uci, axis=1)
    uyi = _slopes(uci, axis=2)

    cdef double[:,:,:] ux = uxi
    cdef double[:,:,:] uc = uci
    cdef double[:,:,:] uy = uyi
    cdef double[:,:,:] out = avg

    cdef int i, j, k
    with nogil:
        for i in range(ux.shape[0]):
            for j in prange(ux.shape[1]-1):
                for k in range(ux.shape[2]-1):
                    out[i,j+1,k+1] = (uc[i,j,k] + uc[i,j+1,k] + uc[i,j,k+1] + uc[i,j+1,k+1])/4 \
                                +((ux[i,j,k] - ux[i,j+1,k]) + (ux[i,j,k+1] -ux[i,j+1,k+1])
                                + (uy[i,j,k] - uy[i,j,k+1]) +(uy[i,j+1,k] - uy[i,j+1,k+1])
                                ) / 16



@inplacewrapper
@cython.boundscheck(False)
def _corrector_step(double[:,:,:] out, double[:,:,:] fx ,
                    double[:,:,:] fy, double lmd_x, double lmd_y):

    cdef int i, j, k
    with nogil:

        for i in range(out.shape[0]):
            for j in prange(out.shape[1]-1):
                for k in range(out.shape[2]-1):
                    out[i,j+1,k+1] = (fx[i,j,k] - fx[i,j+1,k]
                                    - fx[i,j+1,k+1] + fx[i,j,k+1]) * lmd_x/2 \
                                    +(fy[i,j,k]-fy[i,j,k+1]
                                    -fy[i,j+1,k+1] + fy[i,j+1,k]) * lmd_y/2
