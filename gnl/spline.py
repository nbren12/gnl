"""Convenience routines for basis splines



Usage
--------
>>> %paste

import xarray as xr
import numpy as np
import gnl.spline as gs
import matplotlib.pyplot as plt

data = "/scratch/noah/Data"

ds = xr.open_dataset(f"{data}/SAM6.10.9/OUT_MOMENTS/walk-u1.nc")



x = ds.x.values
L = 2*x[-1] - x[-2]
xknots = [
    np.linspace(0, L, 3),
    np.linspace(0, L, 5),
    np.linspace(0, L, 9)
]


time = ds.time.values
T = time[-1]*2 - time[-2]
tknots = [
    np.linspace(0,T, T//20),
    np.linspace(0,T, T//10),
    np.linspace(0,T, T//5.0)
]


coords = [(2, xknots, ds.x, True), (0, tknots, ds.time, False)]


mr2 = list(gs.mrdecomp2(ds.U1.values, coords))



plt.subplot(121)
plt.contourf(gs.downlev2(mr2, 3)[:,10,:], cmap='RdBu')

plt.subplot(122)
plt.pcolormesh((gs.downlev2(mr2)-gs.downlev2(mr2, 3))[:,10,:], cmap='RdBu')

## -- End pasted text --
"""

import numpy as np
from numpy import linspace
from scipy.linalg import pinv
from scipy.interpolate import spleval


def splines(x, t, k=3, deriv=0):
    n = len(t) - 1

    I = np.eye(n + k)

    return np.vstack(spleval(
        (t.copy(), I[i], k),
        x,
        deriv=deriv) for i in range(n + k))


def psplines(x, t, k=3, periodic=True, **kwargs):
    """Periodic beta spline basis

    Parameters
    ----------
    x: ndarray
        spatial locations to evaluate spline basis
    t: array_like
        interior knot locations. assume periodicity at end points
    k: int, optional
        polynomial order of the splines (default 3).
    """
    A = splines(x, t, k=k, **kwargs)

    if not periodic:
        return A

    per = np.hstack(splines(t[0], t, k=k, deriv=p)
                    - splines(t[-1], t, k=k, deriv=p) for p in range(k))

    proj = np.eye(per.shape[0]) - per @ pinv(per)

    # remove rows which have satisfied constraints
    B = proj[:-k,:] @ A

    # normalize to unit height
    B /= np.max(B, 1, keepdims=True)

    return B

### Multi-resolution decompositions 1D

def mrknots(t, factor=2):
    """ Multiresolution generator

    Parameters
    ----------
    t: ndarray
        finest grained knot locations including end points
    factor: int, optional
        coarsening factor between each level [default 2]
    """

    n = len(t) - 1

    while n%factor == 0:
        n = n//factor
        knots = t[::n]
        yield knots


def mrdecomp(data, x, ts, axis=-1):
    """Iterative multiscale decomposition of data

    Parameters
    ----------
    data: ndarray
        input data
    x: ndarray
        coordinate information
    ts: seq
        sequence of knot vectors
    axis: int

    Yields
    ------
    c, b, bp: for all but last step
    resid: for last step
    """

    data = data.copy()

    for t in ts:
        B = psplines(x, t)
        Bp = psplines(x, t, deriv=1)

        S = pinv(B)
        coefs = np.tensordot(data, S, (axis, 0))

        # yield the coefficients and basis functions
        yield coefs, B, Bp, axis

        # compute the residual
        data -= np.tensordot(coefs, B, (axis, 0))


    yield data


def getlev(mr_output, lev, deriv=0):

    if len(mr_output[lev]) == 4:
        c, b, bp, axis = mr_output[lev]
        if deriv == 0:
            return np.tensordot(c, b, (axis, 0))
        elif deriv == 1:
            return np.tensordot(c, bp, (axis, 0))
        else:
            raise NotImplementedError()
    else:
        return mr_output[lev]


def downlev(mr, lev, deriv=0):
    if lev < 0:
        lev += len(mr)
    return sum(getlev(mr, l, deriv=deriv) for l in range(lev+1))

def uplev(mr, lev, deriv=0):
    if lev < 0:
        lev += len(mr)
    return sum(getlev(mr, l, deriv=deriv) for l in range(lev+1, len(mr)))


### Multi-resolution decompositions 2D

def getlev2(lev, deriv={}):
    """expand a level of output from mrdecomp2

    Parameters
    ----------
    lev: tuple
        Has the form (coefs, diminfo)

    deriv: dict
        {axis: order,...} means take order deriv of axis
    """


    coefs, diminfo = lev
    for axis, knots, coord, per in reversed(diminfo):
        b = psplines(coord, knots, periodic=per, deriv=deriv.get(axis, 0))
        coefs = np.tensordot(coefs.swapaxes(axis, -1), b, (-1, 0))\
                  .swapaxes(axis, -1)
    return coefs


def mrdecomp2(data, coords):
    """Iterative multiscale decomposition of data in arbitrary numbers of
    dimensions

    Notes
    -----
    This does not do what Majda's paper saysaa

    Parameters
    ----------
    data: ndarray
        input data
    coords: seq ndarray
        coordinate information. (axis, knot_list, coord, per)
    ts: seq
        sequence of knot information. each element is a seq with elements 
        ((knots, periodic), ...)

    """

    data = data.copy()

    nlev = len(coords[0][1])

    for i in range(nlev):

        diminfo = []

        coefs = data
        for axis, knots, coord, per in coords:
            b = psplines(coord, knots[i], periodic=per)

            S = pinv(b)

            coefs = np.tensordot(coefs.swapaxes(axis,-1), S, (-1, 0))\
                      .swapaxes(axis, -1)

            diminfo.append((axis, knots[i], coord, per))

        # yield the coefficients and basis functions
        yield coefs, diminfo

        # compute the residual
        data -= getlev2((coefs, diminfo))


    yield data


def smooth(a, b, axis):
    return np.tensordot(a.swapaxes(axis,-1), b, (-1, 0))\
             .swapaxes(axis, -1)

def downlev2(mr, lev=None):

    if lev is None:
        lev = len(mr)

    out = getlev2(mr[0])
    for i in range(1, min(lev, len(mr)-1)):
        out += getlev2(mr[i])


    if lev == len(mr):
        out += mr[-1]

    return out




