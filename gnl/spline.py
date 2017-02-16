"""Convenience routines for basis splines
"""

import numpy as np
from numpy import linspace
from scipy.interpolate import spleval


def splines(x, a, b, n, k=3, deriv=0):
    t = linspace(a, b, n)

    I = np.eye(n + k - 1)
    return np.vstack(spleval(
        (t, I[i], k),
        x,
        deriv=deriv) for i in range(n + k - 1))


def psplines(x, a, b, n, k=3):
    """Periodic beta spline basis

    Parameters
    ----------
    x: ndarray
        spatial locations to evaluate spline basis
    a: float
        beginning of interval
    b: float
        end of periodic interval
    n: int
        number of knots within the interval including endpoints
    k: int, optional
        polynomial order of the splines (default 3).
    """
    A = splines(x, a, b, n, k=k)
    per = np.hstack(splines(a, a, b, n, k=k, deriv=p)
                    - splines(b, a, b, n, k=k, deriv=p) for p in range(k))
    proj = np.eye(per.shape[0]) - per @np.linalg.pinv(per)
    return proj @A
