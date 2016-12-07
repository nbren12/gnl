#cython: boundscheck=False, wraparound=False, nonecheck=False
from itertools import product
import numpy as np


from cython.parallel cimport prange, parallel
cimport openmp
from libc.stdio cimport printf

def nemult(double[:,:]outp, double a, double[:,:]b, double[:,:]c):

    cdef int i, j

    for i in prange(outp.shape[0], nogil=True):
        for j in range(outp.shape[1]):
            outp[i,j]  += a * b[i,j] *c[i,j]

