"""Module for converting xarray datasets to and from matrix formats for Machine
learning purposes.
"""
from functools import partial
from typing import Sequence
import warnings
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


def compute_weighted_scale(weight, sample_dims, ds):
    def f(data):
        sig = data.std(sample_dims)
        if set(weight.dims) <= set(sig.dims):
            sig = (sig ** 2 * weight).sum(weight.dims).pipe(np.sqrt)
        return sig
    return ds.apply(f)


def _stack_cat_once(ds: xr.Dataset, new_dim, dims, variable_dim='variable'):
    """Stack dimensions and variables of a Dataset

    Parameters
    ----------
    ds: xr.Dataset
    dims: Sequence

    Returns
    -------
    xr.DataArray
    """
    dims = tuple(dims)

    def f(val):
        # ensure square output

        assign_coords = {variable_dim: val.name}
        for dim in dims:
            if (dim not in val):
                assign_coords[dim] = None

        expand_dims = set(dims).difference(set(val.dims))
        expand_dims.add('variable')
        return val.assign_coords(**assign_coords) \
            .expand_dims(expand_dims) \
            .stack(**{new_dim: (variable_dim,) + dims})

    if isinstance(ds, xr.Dataset):
        Xs = [f(ds[key]) for key in ds.data_vars]
        return xr.concat(Xs, dim=new_dim)
    elif isinstance(ds, xr.DataArray):
        return ds.stack(**{new_dim: dims})
    else:
        raise ValueError("Input must be a xr.DataArray or xr.Dataset")


def unstack_cat(da: xr.DataArray, dim, level=0):
    """Unstack DataArray expanding to dataset along a given level

    Parameters
    ----------
    da
    dim
    level

    Returns
    -------
    xr.Dataset

    """
    if not isinstance(da, xr.DataArray):
        raise ValueError("da must be a DataArray object")

    idx = da.indexes[dim]
    if not isinstance(idx, pd.MultiIndex):
        raise ValueError(f"{dim} is not a stacked coordinate")
    variables = idx.levels[level]

    # pull variables out of datarray
    data_dict = {}
    for k in variables:
        data_dict[k] = da.sel(variable=k).squeeze(drop=True)

    # unstacked dataset
    return xr.Dataset(data_dict)


def stack_cat(ds: xr.Dataset, variable_dim='variable', **kwargs):
    """Dataset aware version of xr.stack
    
    Parameters
    ----------
    ds
    kwargs
    variable_dim

    Returns
    -------
    xr.DataArray

    """
    for new_dim, dims in kwargs.items():
        ds = _stack_cat_once(ds, new_dim, dims, variable_dim=variable_dim)
    return ds


def dataset_to_mat(X, sample_dims):
    sample_dims = tuple(sample_dims)

    # all the dimensions which are not sample dims are feature dims
    feature_dims = tuple(dim for dim in X.dims
                         if dim not in sample_dims)
    return stack_cat(X, features=feature_dims, samples=sample_dims)\
        .transpose('samples', 'features')


def mat_to_dataset(X, coords=None, sample_dims=None, new_dim_name='m'):
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
    warnings.warn(DeprecationWarning("Use stack_cat instead"))

    # Generate coordinates of data matrix
    try:
        coords = X.coords
    except AttributeError:
        if coords is None:
            raise ValueError("If input is not an xarray object"
                             "then coords must be passed")
    # get numpy/dask array
    if isinstance(X, xr.DataArray):
        X = X.data
    
    # Ensure data is two dimensional
    if X.ndim == 1:
        X = X[None, :]

    try:
        nsamples = len(coords['samples'])
    except TypeError:
        nsamples = 1

    if X.shape[0] != nsamples or nsamples == 1:
        new_sample_idx = pd.Index(np.arange(X.shape[0]),
                                  name=new_dim_name)
        coords = (new_sample_idx, coords['features'])

    # unstacked data array
    D = xr.DataArray(X, coords).pipe(lambda x: unstack_cat(x, 'features'))

    try:
        return D.unstack('samples')
    except ValueError:
        if 'samples' in D:
            return D.rename({'samples': sample_dims[0]})
        else:
            return D

class DataMatrix(object):
    """Matrix for inputting/outputting datamatrices from xarray dataset objects

    """

    def __init__(self, sample_dims):
        self.sample_dims = sample_dims
        warnings.warn(DeprecationWarning("Use stack_cat/unstack_cat instead"))

    def dataset_to_mat(self, X):
        out =  dataset_to_mat(X, self.sample_dims)
        self._coords = out.coords
        return out

    def mat_to_dataset(self, X, **kwargs):
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
        return mat_to_dataset(X, self._coords, sample_dims=self.sample_dims)


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


    def fit(self, data):
        self.scale_ = compute_weighted_scale(self.weight, self.sample_dims, data)

    def transform(self, data):
        out = data
        if self.center:
            self.mean_ = out.mean(self.sample_dims)
            out = out - self.mean_

        if self.scale:
            self.fit(data)
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
                 variables=[]):
        """
        Parameters
        ----------
        apply_weight: Bool
            if True weight the output by sqrt of the weight. [default: False]
        **kwargs:
            arguments for DataMatrix and Normalizers classes
        """
        self.dm_ = DataMatrix(sample_dims)
        self.norm_ = Normalizer(scale=scale, center=center, weight=weight,
                                sample_dims=sample_dims)
        self.weight = weight
        self.apply_weight = apply_weight
        self.variables = variables

    def transform(self, data):
        data = self.norm_.transform(data)
        if self.apply_weight:
            f = partial(_mul_if_share_dims, np.sqrt(self.weight))
            data = data.apply(f)
        return self.dm_.dataset_to_mat(data[self.variables])

    def inverse_transform(self, arr):
        data = self.dm_.mat_to_dataset(arr)
        if self.apply_weight:
            f = partial(_mul_if_share_dims, 1/np.sqrt(self.weight))
            data = data.apply(f)
        return self.norm_.inverse_transform(data)

