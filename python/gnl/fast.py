#!/usr/bin/env python
"""A module for fast calculations using the numba library. 

This module is imported separately because numba is a difficult
dependency to install and is frequently not available.

"""
import numba as nb
import matplotlib.mlab as mlab

@nb.autojit
def xcorr(x, y, lag):
    n = x.shape[0]
    start = max(-lag, 0)
    end   = min(n, n - lag)
    ac = 0.0
    for t in range(start, end):
        ac += x[t] * y[t+lag]
        
    return ac

def xcorrpy(x, y, lag):
    n = x.shape[0]
    start = max(-lag, 0)
    end   = min(n, n - lag)
    ac = 0.0
    for t in range(start, end):
        ac += x[t] * y[t+lag]
        
    return ac

@nb.autojit
def xacf(x, y, maxlags=100):
    ac = np.empty((x.shape[0], maxlags*2 +1,))
    
    for i in range(x.shape[0]):
        for l in range(-maxlags, maxlags):
            ac[i,l] = xcorr(x[i,:], y[i,:], l)
        
    return ac


def xycf(x, y, maxlags=100, filt=mlab.demean):
    """Compute the cross correlation function of the inputs"""
    
    if filt is not None:
        x = filt(x)
        y = filt(y)
    
    
    return xacf(x, y, maxlags)



def test_xycf():
    from pylab import rand

    x = rand(100, 1000)
    ac = xycf(x, x)

