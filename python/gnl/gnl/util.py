from functools import wraps
from toolz import *
from numpy import dot
import numpy as np
import scipy.ndimage as nd


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


def phaseshift(x, time, arr, c=0, t0=0, mode='wrap', **kwargs):
    """Phase shift an array

    This is useful for examining simulation output in the moving frame.


    Parameters
    ----------
    x: (n,)
       horizontal axis
    time: (m,)
       vertical axis
    arr: (m, ..., n)
       data on (x,t) grid
    c: float
       speed of wave to track

    Returns
    --------
    phase shifted array

    Notes
    -----
    Does not work for periodic data yet

    """
    dx = x[1] - x[0]

    assert len(time) == arr.shape[0]
    assert len(x) == arr.shape[-1]

    def shift_slice(f, t):
        delta_grid = (t - t0) * c / dx
        shift = [0]*f.ndim
        shift[-1] = -delta_grid

        pad = [(0,0)]*f.ndim
        pad[-1] = (0, 1)
        f = np.pad(f, pad, mode=mode)
        return nd.shift(f, shift, mode=mode, **kwargs)[..., :-1]

    return np.stack([shift_slice(arr[k], time[k])
                     for k in range(time.shape[0])],
                    axis=0)

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

def corr(x, y):
    return x.dot(y) /np.sqrt(x.dot(x) * y.dot(y))

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


def combine_axes(x, axes, **kwargs):
    """Combine axes and tranpose an array

    Parameters
    ----------
    x : np.ndarray
        input data
    axes : seq
        tuple of tuples
    """
    # flatten axes using a stack
    axes_flat = []
    stack = list(axes[::-1])
    counter = 0
    while stack:
        ax = stack.pop()
        if isinstance(ax, int):
            axes_flat.append(ax)
        else:
            stack.extend(ax[::-1])


    # get the new shape
    sh  = []
    for ax in axes:
        if isinstance(ax, int):
            sh.append(x.shape[ax])
        else:
            sh.append(np.prod([x.shape[aa] for aa in ax]))

    return x.transpose(axes_flat).reshape(sh, **kwargs)


def dftderiv(n, d=1.0, order=1):
    """DFT derivative filter for n points spaced by d"""
    ik =  2 * np.pi * 1j * np.fft.fftfreq(n, d=d)
    return ik**order


def pad_along_axis(x, pad_width, mode, axis):
    """Pad along a given axis
    """
    pad_widths = [(0, 0)]*x.ndim
    pad_widths[axis] = pad_width
    return np.pad(x, pad_widths, mode)
