import functools
import inspect

import scipy.ndimage
import xarray as xr


def wrappernd(func):
    """Wrap a subset of scipy.ndimage functions for easy use with xarray"""

    @functools.wraps(func)
    def f(x, axes_kwargs, *args, dims=[], **kwargs):
        # named axes args to list
        axes_args = [
            axes_kwargs[k] if k in axes_kwargs else 0.0 for k in x.dims
        ]
        y = x.copy()

        axes_args.extend(args)
        y.values = func(x, axes_args, **kwargs)
        y.attrs['edits'] = repr(func.__code__)

        return y

    return f


def wrapper1d(func):
    """Wrapper for 1D functions
    """

    @functools.wraps(func)
    def f(x, dim, *args, **kwargs):
        # named axes args to list
        y = x.copy()
        y.values = func(x, *args, axis=x.get_axis_num(dim), **kwargs)
        y.attrs['edits'] = repr(func.__code__)

        return y

    return f


# for each function in scipy.ndimage wrap and add to class
for func_name, func in inspect.getmembers(scipy.ndimage, inspect.isfunction):
    if func_name[-2:] == '1d':
        wrapped_func = wrapper1d(func)
    else:
        wrapped_func = wrappernd(func)
    globals()[func_name] = wrapped_func
