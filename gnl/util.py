from functools import wraps
from toolz import *
from numpy import dot
import numpy as np

# embedded ipython from http://stackoverflow.com/questions/15167200/how-do-i-embed-an-ipython-interpreter-into-an-application-running-in-an-ipython
try:
    get_ipython
except NameError:
    banner=exit_msg=''
else:
    banner = '*** Nested interpreter ***'
    exit_msg = '*** Back in main IPython ***'

# First import the embed function
from IPython.terminal.embed import InteractiveShellEmbed
# Now create the IPython shell instance. Put ipshell() anywhere in your code
# where you want it to open.
ipshell = InteractiveShellEmbed(banner1=banner, exit_msg=exit_msg)


## Functional programming stuff
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

    return getattr(x,name)

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
            f = getattr(x,name)
        else:
            f = name
        cf = curry(f, *ar, **kw)
        return cf(*args, **kwargs)

    return func


def dfassoc(df, key, val, *args):
    """Datafram compatible assoc"""
    df = df.copy()
    df[key] = val

    while(args):
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

def fftdiff(u, L = 4e7, axis=-1):
    """
    Function for calculated the derivative of a periodic signal using fft.

    L is the physical length of the signal (default = 4e7, m aroun earth)
    """
    from numpy.fft import fft, ifft, fftfreq
    nx = u.shape[axis]
    x = np.linspace(0, L, nx, endpoint=False)
    k = fftfreq( nx, 1.0/nx )
    fd = fft(u, axis=axis) * k * 1j * 2 * np.pi / L
    ud = np.real( ifft(fd, axis=axis) )

    return ud
