"""Spline-based coarse-graining for xarrays
"""
import numpy as np
from scipy.linalg import pinv
import xarray as xr

import dask.array as da
from dask.diagnostics import ProgressBar
from dask import delayed

from gnl import spline


class Spline(object):
    """Sklearn-like object for coarse-graining xarray objects using splines"""

    spdimname = 'c_coefs'

    def __init__(self, knots=None, dim=None, order=3, bc='periodic'):
        self.knots = knots
        self.dim = dim
        self.order = order
        self.bc = bc

        # initialize to 0
        self._coef = None


    def _spline_mat(self, x, d=0):
        args = (x, self.knots, self.order)
        if self.bc == 'periodic':
            spline_mat = spline.psplines(*args, deriv=d)
        else:
            spline_mat = spline.splines(*args, deriv=d)

        return spline_mat

    def fit(self, A):
        """Fit the regression spline

        Parameters
        ----------
        A : DataArray
        """

        spline_mat = self._spline_mat(A[self.dim].values)

        # since we are applying the opertor numerous times
        # explicitly compute the pseudoinverse
        B = pinv(spline_mat)


        # wrap operator in xarray object
        dim  = self.dim
        coords = {dim: A[dim],
                  self.spdimname: np.arange(B.shape[1])}

        Bx = xr.DataArray(B, dims=[dim, self.spdimname],
                          coords=coords)

        self._coef = A.dot(Bx)

        # add appropriate attributes
        a = self._coef.attrs
        a['dim'] = self.dim
        a['order'] = self.order
        a['bc'] = self.dim
        a['knots'] = self.knots

        return self


    def predict(self, x, d=0):
        """Evaluate the spline at a set of coordinates
        """
        dim  = self.dim

        spline_mat = self._spline_mat(x, d=d)
        coords = {dim: x, self.spdimname: np.arange(spline_mat.shape[0])}

        B = xr.DataArray(spline_mat,
                         dims=[self.spdimname, dim],
                         coords=coords)


        return self._coef.dot(B)


    def save(self, name):
        self._coef.to_netcdf(name)

    @classmethod
    def load(cls, name, **kwargs):
        Bx = xr.open_dataarray(name, **kwargs)

        spl = cls(knots=Bx.knots, order=Bx.order,
                  bc=Bx.bc, dim=Bx.dim)

        spl._coef = Bx

        return spl







