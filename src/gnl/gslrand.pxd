cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)

cdef extern from "gsl/gsl_randist.h":
    double gamma "gsl_ran_gamma"(gsl_rng * r,double,double)
    double gaussian "gsl_ran_gaussian"(gsl_rng * r,double)

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)





cdef class RandStream:
    """Wrapper class for GSL Random number generators"""
    cdef gsl_rng * _strm
    cpdef __cinit__(self)
    cpdef __call__(self, double sig)
    cdef double normal(self, double sig)
