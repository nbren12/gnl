import functools
import inspect
import xarray as xr
import numpy as np
import scipy.ndimage
from . import util


def ndimage_wrapper(func):
    """Wrap a subset of scipy.ndimage functions for easy use with xarray"""

    @functools.wraps(func)
    def f(x, axes_kwargs, *args, dims=[], **kwargs):

        # named axes args to list
        axes_args = [axes_kwargs[k] for k in x.dims]
        y = x.copy()

        axes_args.extend(args)
        y.values = func(x, axes_args, **kwargs)
        y.attrs['edits'] = repr(func.__code__)

        return y

    return f


def xargs(z):
    """
    Returns:
    x, y, z
    """

    dims = z.dims
    y = z.coords[dims[0]].values
    x = z.coords[dims[1]].values

    return x, y, z.values


def integrate(x, axis='z'):
    """
    Integrate a dataframe along an axis using np.trapz
    """
    axisnum = list(x.dims).index(axis)

    dims = list(x.dims)
    del dims[axisnum]

    coords = {key: x.coords[key] for key in dims}

    tot = np.trapz(x.values, x=x.coords[axis], axis=axisnum)
    return xr.DataArray(tot, coords, dims)


def phaseshift(u500, c=0):
    """
    phaseshift data into a travelling wave frame
    """
    # TODO: add arguments for x and time names
    z = util.phaseshift(u500.x.values,
                        u500.time.values,
                        u500.values,
                        c=c,
                        x_index=-1,
                        time_index=0)

    out = u500.copy()
    out.values = z

    return out


def roll(z, **kwargs):
    """Rotate datarray periodically

    Example
    ------
    """
    roll(U, x=400)
    from scipy import ndimage
    sw = [kwargs[dim] if dim in kwargs else 0 for dim in z.dims]

    zout = ndimage.shift(z, sw, mode='wrap')

    out = z.copy()
    out.values = zout
    return out


def remove_repeats(data, dim='time'):

    dval = data.coords[dim].values

    inds = []
    for tval in np.sort(np.unique(dval)):
        ival = (data[dim].values == tval).nonzero()[0][-1]
        inds.append(ival)

    return data[{dim: inds}]


for func_name, func in inspect.getmembers(scipy.ndimage, inspect.isfunction):
    setattr(xr.DataArray, 'ndimage_' + func_name, ndimage_wrapper(func))


class XRReshaper(object):
    """An object for reshaping DataArrays into 2D matrices

    This can be used to easily transform dataarrays into a format suitable for
    input to scikit-learn functions.
    
    Methods
    -------
    to_flat(dim): 
        return 2D numpy array where the second dimension is specified by dim
    from_flat(arr, old_dim, new_dim):
 s      returns a DataArray where old_dim is replaced by the new_dim

    """

    def __init__(self, da):
        self._da = da

    def to_flat(self, dim):

        da = self._da
        npa = np.rollaxis(da.values, da.get_axis_num(dim), da.ndim)

        sh = npa.shape
        npa = npa.reshape((-1, sh[-1]))

        return npa

    def from_flat(self, arr, old_dim='z', new_dim='m'):

        # create new shape
        sh = list(self._da.shape)
        sh.pop(self._da.get_axis_num(old_dim))
        sh.append(arr.shape[-1])

        # reshape
        arr = arr.reshape(sh)

        # make dim names
        dims = list(self._da.dims)
        dims.remove(old_dim)
        dims.append(new_dim)

        coords = {k: self._da[k] for k in dims[:-1]}
        coords[new_dim] = np.arange(arr.shape[1])

        return xr.DataArray(arr, dims=dims, coords=coords)

# Add custom functions to DataArray class dynamically
xr.DataArray.integrate = integrate
xr.DataArray.roll = roll
xr.DataArray.remove_repeats = remove_repeats
