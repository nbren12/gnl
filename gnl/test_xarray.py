import numpy as np
import xarray as xr
from . import xarray
from .datasets import tiltwave


def test_ndimage():
    a = tiltwave()
    a.ndimage

    # try gaussian filter
    a.ndimage.gaussian_filter(dict(x=1.0, z=0.0))

    # try one dimensional filter
    a.ndimage.gaussian_filter1d('x', 1.0)


def test_reshape():
    a = tiltwave()

    mat = a.reshape.to(['x'])
    mat.shape == (len(a.x), len(a.z))

    mat = a.reshape.to(['z'])
    mat.shape == (len(a.z), len(a.x))

def test_XRReshaper():

    sh = (100, 200, 300)
    dims = ['x', 'y', 'z']

    coords = {}
    for d, n in zip(dims, sh):
        coords[d] = np.arange(n)

    da = xr.DataArray(np.random.rand(*sh), dims=dims, coords=coords)

    arr = da.reshape.to(['z'])

    rss = rs.from_flat(arr, 'z', 'z')
    np.testing.assert_allclose(rss - da, 0)

    return 0
