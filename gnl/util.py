from functools import wraps
from toolz import *
from numpy import dot
import numpy as np


# Functional programming stuff
def argkws(f):
    """Function decorator which maps f((args, kwargs)) to f(*args, **kwargs)"""

    @wraps(f)
    def fun(tup):
        args, kws = tup
        return f(*args, **kw)


def apply(f, *args, **kwargs):
    """Useful for making composable functions"""
    callkwargs = {}
    callargs = []

    if len(args) >= 1:
        callargs += args[0]
    if len(args) >= 2:
        callkwargs.update(args[1])

    callkwargs.update(kwargs)

    return f(*callargs, **callkwargs)


@curry
def rgetattr(name, x):
    """Getattr with reversed args for thread_last """

    return getattr(x, name)


def icall(name, *ar, **kw):
    """A method for calling instance methods of an object by name or static
    reference. It is designed to be used with partial, curry, and thread_last.
    To do this the returned function accepts the object as the final argument
    rather than the first:

    f(obj, *args ) --> icall(f)(*args, obj)

    pipe(rand(100), process, processagain, icall('sum', 10, axis=0))

    Args: name of instancemethod or func which takes object as first arg

    """
    # TODO: use functoools.wraps if `name` is a function
    def func(*args, **kwargs):
        x = args[-1]
        args = args[:-1]

        if isinstance(name, str):
            f = getattr(x, name)
        else:
            f = name
        cf = curry(f, *ar, **kw)
        return cf(*args, **kwargs)

    return func


def dfassoc(df, key, val, *args):
    """Datafram compatible assoc"""
    df = df.copy()
    df[key] = val

    while (args):
        key, val = args[:2]
        df[key] = val
        args = args[2:]

    return df


## Math stuff

def vdot(*arrs, l2r=True):
    """Variadic numpy dot function
    
    Args:
        *arrs:

        l2r (optional): if True then evaluate dot products from left to right
            (default: True)
    """

    if len(arrs) == 2:
        return dot(*arrs)
    else:
        return vdot(arrs[0], vdot(*arrs[1:]))


def fftdiff(u, L=4e7, axis=-1):
    """
    Function for calculated the derivative of a periodic signal using fft.

    L is the physical length of the signal (default = 4e7, m aroun earth)
    """
    from numpy.fft import fft, ifft, fftfreq

    u = np.moveaxis(u, axis, -1)


    nx = u.shape[-1]
    x = np.linspace(0, L, nx, endpoint=False)
    k = fftfreq(nx, 1.0 / nx)
    k.shape = [1]*(u.ndim-1) + [-1]
    fd = fft(u, axis=-1) * k * 1j * 2 * np.pi / L
    ud = np.real(ifft(fd, axis=-1))

    return np.moveaxis(ud, -1, axis)


def phaseshift(x, time, arr, c=0, x_index=0, time_index=-1,
               xmax=None):
    """Phase shift an array

    This is useful for examining simulation output in the moving frame.


    Parameters
    ----------
    x: (n,)
       horizontal axis
    time: (m,)
       vertical axis
    arr: (m, n)
       datavalues on (x,t) grid
    c: float
       speed of wave to track

    Returns
    --------
    phase shifted array

    Notes
    -----
    Does not work for periodic data yet

    """
    from scipy.interpolate import interp1d

    out_arr = np.zeros_like(arr)

    if xmax is None:
        xmax = x[-1] + (x[1] - x[0])

    time_index = time_index % arr.ndim
    x_index = x_index % arr.ndim

    subset_x_index = x_index - 1 if time_index < x_index else x_index

    for t in range(arr.shape[time_index]):
        sl = [slice(None)] * arr.ndim
        sl[time_index] = t

        sl1 = sl.copy()
        sl1[x_index] = slice(0, 1)

        # TODO: this array padding only works for linear or nn interpolation
        subset = np.concatenate((arr[sl], arr[sl1]), subset_x_index)
        xx = np.concatenate((x, [xmax, ]))

        f = interp1d(xx, subset, axis=subset_x_index)

        out_arr[sl] = f((x + c * time[t]) % xmax)

    return out_arr

def linearf2matrix(fun, n):
    """
    Convert linear function to a matrix of a given shape
    :param fun:  linear function to be applied
    :param n: size of input dimension
    :return:
    """

    # Basis vectors
    X = np.eye(n)

    cols = [fun(X[i])[:, None] for i in range(n)]
    return np.concatenate(cols, axis=1)

## data analysis

def meanat(y, inds, axis):
    n  = y.shape[axis]
    di = np.diff(np.r_[inds, n])
    ym = np.add.reduceat(y, inds, axis=axis)

    ym /= baxis(di, ym.ndim-axis)

    return ym



## numpy tricks
def baxis(x, size):
    """Broadcast x to given shape

    Parameters
    ----------
    x: array_like
        input 1 dimensional array
    size: int
        number of np.newaxis to pad to shape of x

    Returns
    -------
    y: array_like
        broadcastable version of x

    """
    x= np.asarray(x)

    ind = [np.newaxis]*size
    ind[0] = slice(None)

    return x[ind]
