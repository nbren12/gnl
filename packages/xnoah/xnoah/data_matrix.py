"""Module for converting xarray datasets to and from matrix formats for Machine
learning purposes.
"""
from functools import partial
import numpy as np
import pandas as pd
import xarray as xr
import xarray.ufuncs as xu


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

        def mystack(val):
            feature_dims = ['variable']\
                           + self.feature_dims
            assign_coords = dict(variable=val.name)
            for dim in feature_dims:
                if (dim not in val) and dim != 'variable':
                    assign_coords[dim] = None

            expand_dims = set(feature_dims).difference(set(val.dims))
            return val.assign_coords(**assign_coords)\
                      .expand_dims(expand_dims)\
                      .stack(features=feature_dims, samples=self.sample_dims)

        Xs = [mystack(X[key]) for key in self.variables]

        catted = xr.concat(Xs, dim='features')\
                   .transpose('samples', 'features')
        self._coords = catted.coords
        return catted

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

        # Generate coordinates of data matrix
        coords = self._coords
        if X.ndim == 1:
            coords = (coords['features'],)
        elif X.shape[0] == 1:
            coords = (coords['features'],)
            X = X[0, :]
        elif len(coords['samples']) != X.shape[0]:
            new_sample_idx = pd.Index(np.arange(X.shape[0]),
                                      name=new_dim_name)
            coords = (new_sample_idx, coords['features'])

        # stacked data array
        xarr = xr.DataArray(X, coords)

        # unstack variables
        # unstacking automatically happens in sel
        data_dict = {}
        for k in self.variables:
            data_dict[k] = xarr.sel(variable=k).squeeze(drop=True)

        # unstacked dataset
        D = xr.Dataset(data_dict)
        return _unstack_rename(D, {'samples': self.sample_dims[0]})

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
            sig = (sig ** 2 * self.weight).sum(self.weight.dims).pipe(np.sqrt)
        return sig

    def fit(self, data):
        self.scale_ = data.apply(self._get_normalization)

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

    @property
    def matrix(self):
        """Return the normalizing weights in an xarray

        The weights are proportional to sig * sqrt(w). I could have used sig**2
        w as the convention, but this is more useful.
        """
        scale = self.scale_
        w = self.weight
        return xu.sqrt(w) * scale


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
                 apply_weight=False, sample_dims=[],
                 feature_dims=[], variables=[]):
        """
        Parameters
        ----------
        apply_weight: Bool
            if True weight the output by sqrt of the weight. [default: False]
        **kwargs:
            arguments for DataMatrix and Normalizers classes
        """
        self.dm_ = DataMatrix(feature_dims, sample_dims, variables)
        self.norm_ = Normalizer(scale=scale, center=center, weight=weight,
                                sample_dims=sample_dims)
        self.weight = weight
        self.apply_weight = apply_weight

    def transform(self, data):
        data = self.norm_.transform(data)
        if self.apply_weight:
            f = partial(_mul_if_share_dims, np.sqrt(self.weight))
            data = data.apply(f)
        return self.dm_.dataset_to_mat(data)

    def inverse_transform(self, arr):
        data = self.dm_.mat_to_dataset(arr)
        if self.apply_weight:
            f = partial(_mul_if_share_dims, 1/np.sqrt(self.weight))
            data = data.apply(f)
        return self.norm_.inverse_transform(data)


def test_datamatrix():
    from gnl.datasets import tiltwave

    # setup data
    a = tiltwave()
    b = a.copy()
    D = xr.Dataset({'a': a, 'b': b})

    mat = DataMatrix(['z'], ['x'], ['a', 'b'])
    y = mat.dataset_to_mat(D)

    assert y.dims == ('samples', 'features')

    x = mat.mat_to_dataset(y)

    _assert_dataset_approx_eq(D, x)

    # test on just one sample
    x0 = mat.mat_to_dataset(y[0])
    d0 = D.isel(x=0)
    _assert_dataset_approx_eq(d0, x0)

    # test when variables have different dimensionality
    D = xr.Dataset({'a': a, 'b': b.isel(z=0)})
    mat = DataMatrix(['z'], ['x'], ['a', 'b'])
    y = mat.dataset_to_mat(D)
    assert y.dims == ('samples', 'features')
    x = mat.mat_to_dataset(y)
    _assert_dataset_approx_eq(D, x)

    # test that the variables of the output are the desired ones
    # this caused a really ugly bug
    D = xr.Dataset({'a': a, 'b': b, 'c': b+1})
    variables = ['b', 'c']
    mat = DataMatrix(['z'], ['x'], variables)
    y = mat.dataset_to_mat(D)
    output_vars = set(y.unstack('features')['variable'].values)
    assert set(variables) == output_vars



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

    # test weight matrix
    normalizer = norm.matrix

    print((normalizer**2 * w).sum('z'), scales)
    _assert_dataset_approx_eq((D-D.mean('x'))/(normalizer**2).sum('z').pipe(np.sqrt),
                              d_norm)

    

if __name__ == '__main__':
    test_datamatrix()
