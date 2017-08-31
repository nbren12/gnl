import numpy as np
import xarray as xr
from .xcalc import centderiv, centspacing, cumtrapz
from .datasets import tiltwave


def test_centdiff():

    A = tiltwave().chunk()

    B = centderiv(A, dim='z', boundary='extrap')
    B = A.centderiv(dim='z', boundary='extrap')


    if True:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        fig, (a,b) = plt.subplots(1,2)
        B.plot(ax=a)

        A.plot(ax=b)
        plt.show()


def test_centspacing():
    x = np.arange(5)

    xd = xr.DataArray(x, coords=( ('x', x), ))

    dx = centspacing(xd).values

    np.testing.assert_allclose(dx, np.ones_like(dx) * 2)


def test_cumtrapz():
    A = tiltwave().chunk({'z':5})

    from scipy.integrate import cumtrapz




    res_scipy = cumtrapz(A.values, A.z.values,
                         axis=A.get_axis_num('z'), initial=0)

    res_xr = A.cumtrapz('z').values
    np.testing.assert_array_almost_equal(res_scipy, res_xr)
