cdef class RandStream:
    """Wrapper class for GSL Random number generators"""
    cdef gsl_rng * _strm
    def __cinit__(self)
    def __call__(self, double sig)

    cdef double normal(self, double sig)

