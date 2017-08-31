"""Calculus for xarray data
"""
import numpy as np
import xarray as xr

import dask.array as da

def dask_centdiff(x, axis=-1, boundary='reflect'):
    """Compute centered difference using dask
    """

    def fun(x):
        return  np.roll(x, -1, axis) - np.roll(x, 1, axis)

    return da.ghost.map_overlap(x, fun, {axis: 1},
                                boundary={axis: boundary},
                                dtype=x.dtype)


def centdiff(A, dim='x', boundary='periodic'):
    if A.chunks is None:
        A = A.chunk()
    dat = dask_centdiff(A.data, axis=A.get_axis_num(dim), boundary=boundary)


    try:
        name = 'd' + dim + A.name
    except TypeError:
        name = 'd' + dim

    xdat = xr.DataArray(dat, coords=A.coords, name=name)

    return xdat


def centspacing(x):
    """Return spacing of a given xarray coord object
    """

    dx = x.copy()

    x = x.values

    xpad = np.r_[2*x[0]-x[1], x, 2* x[-1] - x[-2]]

    dx.values = xpad[2:] - xpad[:-2]

    return dx


def centderiv(A, dim='x', boundary='periodic'):
    """Compute centered derivative of a data array

    Parameters
    ----------
    A: DataArray
        input dataset
    dim: str
        dimension to take derivative along
    boundary: str
        boundary conditions along dimension. One of 'periodic', or 'extrap'.

    """
    return centdiff(A, dim=dim, boundary=boundary)/centspacing(A[dim])


def cumtrapz(A, dim):
    """Cumulative Simpson's rule (aka Tai's method)

    Notes
    -----
    Simpson rule is given by
        int f (x) = sum (f_i+f_i+1) dx / 2
    """
    x = A[dim]
    dx = x - x.shift(**{dim:1})
    dx = dx.fillna(0.0)
    return ((A.shift(**{dim:1}) + A)*dx/2.0)\
          .fillna(0.0)\
          .cumsum(dim)


## monkey patch DataArray class
xr.DataArray.centderiv = centderiv
xr.DataArray.cumtrapz = cumtrapz
