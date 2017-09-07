"""Module for converting xarray datasets to and from matrix formats for Machine
learning purposes.
"""
from functools import partial
import numpy as np
import pandas as pd
import xarray as xr


def _unstack_rename(xarr, rename_dict):
    # expand/rename all coords
    for dim in xarr.coords:
        try:
            xarr = xarr.unstack(dim)
        except ValueError:
            # unstack returns error if the dim is not an multiindex
            if dim in rename_dict:
                xarr = xarr.rename({dim: rename_dict[dim]})
    return xarr


class DataMatrix(object):
    """Matrix for inputting/outputting datamatrices from xarray dataset objects

    """

    def __init__(self, feature_dims, sample_dims, variables):
        self.dims = {'samples': sample_dims, 'features': feature_dims}
        self.variables = variables

    @property
    def feature_dims(self):
        return self.dims['features']

    @property
    def sample_dims(self):
        return self.dims['samples']

    def dataset_to_mat(self, X):
        Xs = X.stack(samples=self.sample_dims, features=self.feature_dims)\
              .transpose('samples', 'features')

        self._var_coords = {k: Xs[k].coords for k in self.variables}

        # store offsets
        offset = 0
        self._var_slices = {}
        one_sample = Xs.isel(samples=0)
        for k in self.variables:
            nfeat = one_sample[k].size
            self._var_slices[k] = slice(offset, offset+nfeat)
            offset += nfeat

        return np.hstack(self.column_var(Xs[k]) for k in self.variables)

    def mat_to_dataset(self, X, new_dim_name='m'):
        """Munge 2d array into xarray object matching input to dataset_to_mat

        Parameters
        ----------
        X: array_like (1d, or 2d)
            input data matrix. If 1D it is assumed to have the same shape as
            one sample of the input.
        new_dim_name: str, optional
            Name of the trivial index to be used to represents rows of the
            input, if the number of rows of X does not match the stored
            dimension size. (default: 'm')

        Returns
        -------
        dataset: xr.Dataset
        """

        data_dict = {}

        for k in self.variables:
            coords = self._var_coords[k]

            # need data to be 2d
            if X.ndim == 2:
                data = X[:, self._var_slices[k]]
            else:
                data = X[None, self._var_slices[k]]

            if data.shape[1] == 1:
                data = data[:, 0]

            # create new index if shapes don't match
            if data.shape[0] == 1:
                coords = (coords['features'],)
                data = data[0, :]
            elif len(coords['samples']) != data.shape[0]:
                new_sample_idx = pd.Index(np.arange(data.shape[0]),
                                          name=new_dim_name)
                coords = (new_sample_idx, coords['features'])

            xarr = xr.DataArray(data, coords, name=k)
            rename_dict = {dim: val if isinstance(val, str) else val[0]
                           for dim, val in self.dims.items()}
            data_dict[k] = _unstack_rename(xarr, rename_dict)

        return xr.Dataset(data_dict)

    def column_var(self, x):
        if x.ndim == 1:
            return x.data[:, None]
        else:
            return x


class Normalizer(object):
    def __init__(self, scale=True, center=True, weight=None,
                 sample_dims=()):
        self.scale = scale
        self.center = center
        self.weight = weight
        self.sample_dims = sample_dims

    def _get_normalization(self, data):
        sig = data.std(self.sample_dims)
        if set(self.weight.dims) <= set(sig.dims):
            sig = np.sqrt((sig) ** 2 * self.weight).sum(self.weight.dims)
        return sig

    def transform(self, data):
        out = data
        if self.center:
            self.mean_ = out.mean(self.sample_dims)
            out = out - self.mean_

        if self.scale:
            self.scale_ = out.apply(self._get_normalization)
            out = out / self.scale_

        return out

    def inverse_transform(self, data):
        return data * self.scale_ + self.mean_


def _assert_dataset_approx_eq(D, x):
    for k in D.data_vars:
        np.testing.assert_allclose(D[k], x[k].transpose(*D[k].dims))


def _mul_if_share_dims(w, d):
    """Useful for multiplying by weight
    """
    if set(d.dims) >= set(w.dims):
        return d * w
    else:
        return d


class NormalizedDataMatrix(object):
    """Class for computing Normalized data matrices
    """

    def __init__(self, scale=True, center=True, weight=None,
                 sample_dims=[], feature_dims=[], variables=[]):
        self.dm_ = DataMatrix(feature_dims, sample_dims, variables)
        self.norm_ = Normalizer(scale=scale, center=center, weight=weight,
                                sample_dims=sample_dims)
        self.weight = weight

    def transform(self, data):
        data = self.norm_.transform(data)
        f = partial(_mul_if_share_dims, np.sqrt(self.weight))
        return self.dm_.dataset_to_mat(data.apply(f))

    def inverse_transform(self, arr):
        data = self.dm_.mat_to_dataset(arr)
        f = partial(_mul_if_share_dims, 1/np.sqrt(self.weight))
        return self.norm_.inverse_transform(data.apply(f))



def test_datamatrix():
    from gnl.datasets import tiltwave

    # setup data
    a = tiltwave()
    b = a.copy()
    D = xr.Dataset({'a': a, 'b': b})

    mat = DataMatrix(['z'], ['x'], ['a', 'b'])
    y = mat.dataset_to_mat(D)

    x = mat.mat_to_dataset(y)

    _assert_dataset_approx_eq(D, x)

    # test on just one sample
    x0 = mat.mat_to_dataset(y[0])
    d0 = D.isel(x=0)
    _assert_dataset_approx_eq(d0, x0)


def test_normalizer():
    from gnl.datasets import tiltwave

    # setup data
    a = tiltwave()
    b = a.copy()
    D = xr.Dataset({'a': a, 'b': b, 'c': a.isel(z=10)})

    w = np.exp(-a.z/10e3/2)
    norm = Normalizer(sample_dims=['x'], weight=w)
    d_norm = norm.transform(D)
    scales = d_norm.apply(norm._get_normalization)

    for k in scales:
        np.testing.assert_allclose(scales[k], 1.0)
