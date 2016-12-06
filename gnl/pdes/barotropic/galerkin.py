"""This module computes the coefficient for the nonlinear terms
"""
from pychebfun import Chebfun, plot as cplot

import numpy as np
from math import pi, sqrt


def a(p, order):
    if p > 0:
        return Chebfun.from_function(lambda x: sqrt(2)* np.sin(p*x)/p,[0,pi])\
                      .differentiate(order)
    else:
        if order == 1:
            return Chebfun.from_function(lambda x: np.ones_like(x), [0, pi])
        else:
            return Chebfun.from_function(lambda x: np.zeros_like(x), [0, pi])


@np.vectorize
def Au(i, j, k):
    return (a(i, 1) * a(j, 1) * a(k, 1)).sum() / pi


@np.vectorize
def Bu(i, j, k):
    return (a(i, 1) * a(j, 0) * a(k, 2)).sum() / pi


@np.vectorize
def Tu(i, j):
    return (a(i, 1) * a(j, 1)).sum() / pi


@np.vectorize
def At(i, j, k):
    return (a(i, 2) * a(j, 1) * a(k, 2)).sum() / pi


@np.vectorize
def Bt(i, j, k):
    return (a(i, 2) * a(j, 0) * a(k, 3)).sum() / pi


@np.vectorize
def Tt(i, j):
    return (a(i, 2) * a(j, 2)).sum() / pi


@np.vectorize
def Tu(i, j):
    return (a(i, 1) * a(j, 1)).sum() / pi


def sparsify(tensor):
    inds = (np.abs(tensor) > 1e-10).nonzero()

    val = tensor[inds]

    inds = zip(*[list(i) for i in inds])
    return list(zip(inds, val))

def sparsify_inds(inds, tensor):
    out = []
    for i, val in sparsify(tensor):
        out.append((tuple(ind[i] for ind in inds), val))

    return out


def flux_div_u():
    # flux terms in u equation no need to invert using mass matrix
    l, k, m = np.mgrid[0:3, 0:3, 0:3]
    flux_tensor = Au(l, k, m)
    div_tensor = -Bu(l, k, m)

    div_tensor[:,0,:] = 0

    return flux_tensor, div_tensor


def flux_div_t():
    # temperature equation
    l, k, m = np.mgrid[0:3, 0:3, 0:3]
    flux_tensor = At(l, k, m)

    div_tensor = -At(l, k, m) - Bt(l, k, m)
    l[0, ...] = 1.0
    div_tensor/= l**2

    div_tensor[:,0,:] = 0

    return flux_tensor, div_tensor
