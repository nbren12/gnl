from functools import wraps
from numpy import dot

def argkws(f):
    """Function decorator which maps f((args, kwargs)) to f(*args, **kwargs)"""
    @wraps(f)
    def fun(tup):
        args, kws = tup
        return fun(*args, **kw)

def nbsubplots(nrows=1, ncols=1, w=None, h=1.0, aspect=1.0, **kwargs):
    """Make a set of axes with fixed aspect ratio"""
    from matplotlib import pyplot as plt

    if w is not None:
        h = w * aspect
    else:
        w = h / aspect

    return  plt.subplots(nrows,ncols, figsize=(w * ncols ,h * nrows), **kwargs)

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
