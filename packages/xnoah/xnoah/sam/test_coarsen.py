import numpy as np
import xarray as xr
from xarray.testing import assert_equal

from .coarsen import coarsen, destagger


def test_coarsen_staggered():

    x,y= np.ogrid[:40,:40]
    data = x + y*0
    x, y = [arr.ravel() for arr in [x, y]]
    xarr = xr.DataArray(data, coords=(('x', x), ('y', y)))
    blocks = {'x': 10, 'y':10}


    ds = destagger(xarr, 'x', mode='wrap')
    c1 = coarsen(ds, blocks)

    c2 = coarsen(xarr, blocks, stagger_dim='x', mode='wrap')

    assert_equal(c1.transpose(*c2.dims), c2)
