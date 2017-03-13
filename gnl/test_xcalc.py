import numpy as np
from .xcalc import centderiv, centspacing
from .datasets import tiltwave


def test_centdiff():

    A = tiltwave().chunk()

    B = centderiv(A, dim='z', boundary='extrap')
    B = A.centderiv(dim='z', boundary='extrap')


    if False:
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
