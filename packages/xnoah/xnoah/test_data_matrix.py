"""Module for converting xarray datasets to and from matrix formats for Machine
learning purposes.
"""
import numpy as np
import xarray as xr

from .datasets import tiltwave
from .data_matrix import DataMatrix, Normalizer


def _assert_dataset_approx_eq(D, x):
    for k in D.data_vars:
        np.testing.assert_allclose(D[k], x[k].transpose(*D[k].dims))


def test_datamatrix():

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
