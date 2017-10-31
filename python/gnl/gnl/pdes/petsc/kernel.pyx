#cython: boundscheck=False
from cython.parallel cimport prange


def div_kernel(un, vn, dn, h):

    cdef double h2
    cdef int nx, ny, i, j

    nx, ny = un.shape

    h2 = h

    cdef double[:,:] u = un
    cdef double[:,:] v = vn
    cdef double[:,:] d = dn

    with nogil:

        for i in prange(1,nx):
            for j in range(1,ny):
                d[i,j] = h2 * ((u[i,j] + u[i, j-1] - u[i-1,j] - u[i-1,j-1]) +
                            (v[i,j] + v[i-1,j] - v[i,j-1] - v[i-1,j-1]))

def div_pressure(pi, pxi, pyi, h):

    cdef double h2
    cdef int nx, ny, i, j

    nx, ny = pxi.shape

    h2 = 2*h

    cdef double[:,:] p = pi
    cdef double[:,:] px = pxi
    cdef double[:,:] py = pyi

    with nogil:

        for i in prange(1,nx-1):
            for j in range(1,ny-1):
                px[i,j] = (p[i+1,j]-p[i,j] + p[i+1,j+1]-p[i,j+1])/h2
                py[i,j] = (p[i,j+1] -p[i,j]+p[i+1,j+1] -p[i+1,j])/h2
