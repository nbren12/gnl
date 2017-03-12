from .spline import psplines
import numpy as np

PLOT=False


def test_fine_knots_coarse_values():
    t = np.linspace(0, 1, 100)
    x = np.linspace(0, 1, 10)
    B = psplines(x, t)

    assert not np.isnan(B).any()


def test_coarse_knots_fine_values():
    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 10)
    B = psplines(x, t)

    if PLOT:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        plt.plot(B.T)
        plt.show()

    assert not np.isnan(B).any()
