import numpy as np
import xarray as xr
from .xspline import Spline
from .datasets import tiltwave
import os

PLOT=False

def test_xspline():
    A = tiltwave()

    dim = 'x'
    knots = np.linspace(A.x.min(), A.x.max(), 5)

    ## Test interface
    # fine grid locations
    xf = np.linspace(A.x.min(), A.x.max(), 50)

    spl = Spline(knots, dim, order=3, bc='')

    # needs an xarray dataarray A
    spl.fit(A)

    # saving spline object to disk and loading from disk
    spl.save("spline.nc")
    spl = Spline.load("spline.nc")

    # delete spline.nc object
    os.unlink("spline.nc")


    # coarse output
    xf = np.linspace(A.x.min(), A.x.max(), 10)
    B = spl.predict(xf, d=0)
    if PLOT:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        B.plot()
        plt.show()

    ## test interpolation
    spl = Spline(knots=A.x, dim='x').fit(A)
    B = spl.predict(A.x)



    np.testing.assert_array_almost_equal(A, B)


    return 0


def test_spline_der():

    x = np.linspace(0, 2 * np.pi, 101)

    y = np.sin(x)

    yd = xr.DataArray(y, (('x', x), ))

    spl = Spline(yd.x, 'x').fit(yd)
    deriv = spl.predict(yd.x, d=1)
    np.testing.assert_allclose(deriv, np.cos(x), atol=1e-5)
