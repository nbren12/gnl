from .spline import psplines
import numpy as np


def test_fine_knots_coarse_values():
    t = np.linspace(0, 1, 100)
    x = np.linspace(0, 1, 10)
    B = psplines(x, t)

    assert not np.isnan(B).any()
