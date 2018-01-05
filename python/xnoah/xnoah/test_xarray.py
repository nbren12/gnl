import numpy as np
import xarray as xr
from . import xarray
from .xarray import swap_coord
from .datasets import tiltwave, rand3d


def test_ndimage():
    a = tiltwave()
    a.ndimage

    # try gaussian filter
    a.ndimage.gaussian_filter(dict(x=1.0, z=0.0))

    # try one dimensional filter
    a.ndimage.gaussian_filter1d('x', 1.0)


def test_reshape():
    a = tiltwave()

    mat, dl = a.reshape.to(['x'])
    assert mat.shape == (len(a.z), len(a.x))

    mat, dl = a.reshape.to(['z'])
    assert mat.shape == (len(a.x), len(a.z))

    a.reshape.get(mat, dl, a.coords)

    da = rand3d()
    arr, dl = da.reshape.to(['z', 'y'])



    assert arr.shape == (len(da.x), len(da.y)*len(da.z))
    rss = da.reshape.get(arr, dl, da.coords)
    np.testing.assert_allclose(rss - da, 0)



    return 0


def test_wrapcli():

    @xarray.wrapcli
    def theta(A, B):
        return A*B


    tiltwave().to_dataset(name="A").to_netcdf("test.nc")
    tiltwave().to_dataset(name="B").to_netcdf("testB.nc")

    AB = theta("A", "B", data=["test.nc", "testB.nc"])


def test_map_overlap():
    a = tiltwave()

    def fun(x):
        return x

    y = xarray.map_overlap(a.chunk(), fun, {'x': 1}, {'x': 'periodic'})
    np.testing.assert_allclose(y.values, a.values)


def test_swap_coord():
    x = np.arange(10)
    y = np.arange(10) + 1
    z = np.random.rand(10)

    d = xr.Dataset({'z': (['x'], z), 'y': (['x'], y)},
                   coords=dict(x=x))

    d_s = swap_coord(d, {'x': 'y'})

    assert d_s.z.dims == ('y',)
    assert d_s.x.dims == ('y',)
