import numpy as np
import xarray as xr
from . import xarray
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
