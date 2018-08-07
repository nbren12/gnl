"""Routines for performing calculus with data on Arakawa C-grids"""
from scipy.ndimage import correlate1d


def average_to_left(x, **kwargs):
    return correlate1d(x, [.5, .5], origin=0, **kwargs)


def diff_to_right(x, **kwargs):
    mode = kwargs.pop('mode', 'wrap')

    if mode == 'neumann':
        kwargs['mode'] = 'constant'
        kwargs['cval'] = 0.0
    else:
        kwargs['mode'] = mode

    return correlate1d(x, [1, -1], origin=-1, **kwargs)


def c_grid_advective_tendency(u, v, w, f, dx, dz, rho):
    """Compute the tendency due to advection of a tracer by the winds on the staggered grid: 

    :math:`- \rho^{-1} div . (\rho v f)`

    For this to work, u, v, and w must conserve mass. Otherwise 

    Parameters
    ----------
    u, v, w
        the winds on a staggered grid. The shape is assumed to be (z, y, x) and
        the units are m/s.
    f
        the tracer on a non-staggered grid
    dx
        the grid spacing (assuming constant spacing in x and y)
    dz
        grid spacing in vertical direction
    rho
        mass in vertical direction

    Returns
    -------
    tendency : ([units f]/s)

    Notes
    -----
    Fx_i = u_i (f_{i-1} + f_{i+1})/2
    Fy_i = v_i (f_{i-1} + f_{i+1})/2

    """

    dz = dz.reshape((-1, 1, 1))
    rho = rho.reshape((-1, 1, 1))
    dm = rho * dz

    Fx_i = average_to_left(f, axis=-1, mode='wrap') * u
    Fy_i = average_to_left(f, axis=-2, mode='mirror') * v
    Fz_i = average_to_left(f * rho, axis=-3, mode='mirror') * w

    return (diff_to_right(Fx_i, axis=-1, mode='wrap') / dx +
            diff_to_right(Fy_i, axis=-2, mode='neumann') / dx +
            diff_to_right(Fz_i, axis=-3, mode='neumann') / dm)
