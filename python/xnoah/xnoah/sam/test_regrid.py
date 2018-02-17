import numpy as np
import xarray as xr
from xarray.testing import assert_equal

from .regrid import (coarsen, coarsen_dim, destagger, staggered_to_left,
                     staggered_to_right, centered_to_left, centered_to_right,
                     isel_bc, get_center_coords, coarsen_centered_np,
                     coarsen_centered, coarsen_staggered)

def test_coarsen():
    x = xr.DataArray(np.arange(10), dims=['x'])
    y = coarsen_centered(x, {'x': 5})
    np.testing.assert_equal(y.values, [2, 7])


def test_coarsen_staggered():
    x = xr.DataArray(np.arange(10), dims=['x'])
    y = coarsen_staggered(x, {'x': 5}, stagger_dim='x')
    np.testing.assert_equal(y.values, [2.5, 6.5])

    x, y = np.ogrid[:40, :40]
    data = x + y * 0
    x, y = [arr.ravel() for arr in [x, y]]
    xarr = xr.DataArray(data, coords=(('x', x), ('y', y)))
    blocks = {'x': 10, 'y': 10}

    ds = destagger(xarr, 'x', mode='wrap')
    c1 = coarsen_centered(ds, blocks)

    c2 = coarsen_staggered(xarr, blocks, stagger_dim='x', mode='wrap')

    assert_equal(c1.transpose(*c2.dims), c2)

    x = xr.DataArray(np.arange(10), dims=['x'])
    x = x.assign_coords(x=x)
    y = coarsen(x, {'x': 5})
    np.testing.assert_equal(y.coords['x'].values, [2.5, 7.5])


def test_coarsen_centered_np():
    x = np.arange(10)
    y = coarsen_centered_np(x, {0: 5})
    np.testing.assert_equal(y, [2, 7])

    x = np.ones((10, 12, 10))
    y = coarsen_centered_np(x, {0:5, 1:4})
    assert y.shape == (2, 3, 10)


def test_coarsen_dim():
    x = xr.DataArray(np.arange(10), dims=['x'])
    x = x.assign_coords(x=x)
    y = coarsen(x, {'x' : 5})
    np.testing.assert_equal(y.coords['x'].values, [2.5, 7.5])
    np.testing.assert_equal(y.values, [2, 7])


def test__to_left_to_right():
    x = xr.DataArray(np.arange(10), dims=['x'])
    x = x.assign_coords(x=x.values)

    y = staggered_to_left(x, 5, dim='x')
    np.testing.assert_equal(y.values, [0, 5])
    # test coordinates
    np.testing.assert_equal(y.coords['x'].values, [2.5, 7.5])

    y = staggered_to_right(x, 5, dim='x')
    np.testing.assert_equal(y.values, [5, 0])

    y = staggered_to_right(x, 5, dim='x', boundary='extrap')
    np.testing.assert_equal(y.values, [5, 9])


def test_isel_bc():
    x = xr.DataArray(np.arange(10), dims=['x'])

    y = isel_bc(x, -1, dim='x', boundary='wrap')
    np.testing.assert_equal(y.values, 9)

    y = isel_bc(x, -1, dim='x', boundary='extrap')
    np.testing.assert_equal(y.values, 0)

    y = isel_bc(x, slice(-2, 2), dim='x', boundary='extrap')
    np.testing.assert_equal(y.values, [0,0,0,1])

    y = isel_bc(x, range(-2, 2), dim='x', boundary='extrap')
    np.testing.assert_equal(y.values, [0,0,0,1])

    y = isel_bc(x, slice(-2, 2), dim='x', boundary='wrap')
    np.testing.assert_equal(y.values, [8,9,0,1])

    y = isel_bc(x, slice(10, 12), dim='x', boundary='wrap')
    np.testing.assert_equal(y.values, [0, 1])


def test_centered_to_left():
    x = xr.DataArray(np.arange(10)+.5, dims=['x'])

    y = centered_to_left(x, 5, 'x', boundary='extrap')
    np.testing.assert_equal(y.values, [0.5, 5])

    y = centered_to_left(x, 2, 'x', boundary='extrap')
    np.testing.assert_equal(y.values, [.5, 2, 4, 6, 8])


def test_centered_to_right():
    x = xr.DataArray(np.arange(10)+.5, dims=['x'])
    y = centered_to_right(x, 2, 'x', boundary='extrap')
    np.testing.assert_equal(y.values, [2, 4, 6, 8, 9.5])

    y = centered_to_right(x, 2, 'x', boundary='wrap')
    np.testing.assert_equal(y.values, [2, 4, 6, 8, 5])


def test_get_center_coords():
    x = np.arange(10)
    y = get_center_coords(x, 5)
    np.testing.assert_equal(y, [2.5, 7.5])
