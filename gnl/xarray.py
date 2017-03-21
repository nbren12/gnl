"""A module containing useful patches to xarray


"""
import functools
import inspect
import xarray as xr
import dask.array as da
import numpy as np
import scipy.ndimage
from . import util


## ndimage wrapper
class MetaNdImage(type):
    def __new__(cls, name, parents, dct):

        # for each function in scipy.ndimage wrap and add to class
        for func_name, func in inspect.getmembers(scipy.ndimage, inspect.isfunction):
            if func_name[-2:] == '1d':
                dct[func_name] = MetaNdImage.wrapper1d(func)
            else:
                dct[func_name] = MetaNdImage.wrappernd(func)
            # setattr(xr.DataArray, 'ndimage_' + func_name, ndimage_wrapper(func))
        return  super(MetaNdImage, cls).__new__(cls, name, parents, dct)


    def wrappernd(func):
        """Wrap a subset of scipy.ndimage functions for easy use with xarray"""

        @functools.wraps(func)
        def f(self, axes_kwargs, *args, dims=[], **kwargs):

            x = self._obj
            # named axes args to list
            axes_args = [axes_kwargs[k] for k in x.dims]
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
        def f(self, dim, *args, **kwargs):

            x = self._obj
            # named axes args to list
            y = x.copy()
            y.values = func(x, *args, axis=x.get_axis_num(dim), **kwargs)
            y.attrs['edits'] = repr(func.__code__)

            return y

        return f

@xr.register_dataarray_accessor('ndimage')
class NdImageAccesor(metaclass=MetaNdImage):
    def __init__(self, obj):
        self._obj = obj


def xargs(z):
    """Return x,y,z

    This function is useful for plotting
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
    z = util.phaseshift(
        u500.x.values,
        u500.time.values,
        u500.values,
        c=c,
        x_index=-1,
        time_index=0)

    out = u500.copy()
    out.values = z

    return out


def phaseshift_regular_grid(A, speed):
    """
    Phase shift data for wave-frame moving averages

    Parameters
    ----------
    A: DataArray
        The input data array can have any number of dimensions, but it must
        satisfy some simple properties:

        1. periodicity in the 'x' direction
        2. 'time' is the first dimension of the dataset
        3. 'time' coord has units 's'
        4. 'x' coord has units 'm'
        5. 'x', and 'time' are defined on a regular grid

    speed: float
        The speed of waves to follow.

    Returns
    -------
    C : The phase shifted data array

    """
    from scipy.ndimage.interpolation import shift as ndshift
    C = A.copy()

    # Grid spacing
    dx = A.x[1] - A.x[0]
    dt = (A.time[1] - A.time[0]) * 86400

    # shift = (-c * t, 0) = (- c * dt * i / dx)
    def indshift(i):
        shift = [0]*(C.ndim-1)
        shift[C.get_axis_num('x')-1] = float((-speed *dt * i/dx).values)
        return shift



    # shift data
    for it, t in enumerate(A.time):
        ndshift(A.values[it,...], indshift(it), output=C.values[it,...], mode='wrap')

    return C

def roll(z, **kwargs):
    """Rotate datarray periodically

    Examples
    --------
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


# for func_name, func in inspect.getmembers(scipy.ndimage, inspect.isfunction):
#     setattr(xr.DataArray, 'ndimage_' + func_name, ndimage_wrapper(func))


@xr.register_dataarray_accessor('reshape')
class XRReshaper(object):
    """An object for reshaping DataArrays into 2D matrices

    This can be used to easily transform dataarrays into a format suitable for
    input to scikit-learn functions.

    """

    def __init__(self, da):
        self._da = da

    def to(self, feature_dims):
        """Reshape data array into 2D array

        Parameters
        ----------
        feature_dims: seq of dim names
            list of dimensions that will be the features (i.e. columns) for the result

        Returns
        -------
        arr: matrix
            reshaped data
        dims: seq of dim names
            list of dim names in the same order as the output array. useful for the from function below.

        """
        A = self._da


        dim_list = [dim for dim in A.dims if dim not in feature_dims] \
                    + feature_dims

        axes_list = [A.get_axis_num(dim) for dim in dim_list]

        npa = A.data.transpose(axes_list)

        sh = npa.shape

        nfeats =  np.prod(sh[-len(feature_dims):])
        npa = npa.reshape((-1, nfeats))

        return npa, dim_list

    def get(self, arr, dims):
        coords = {}
        for i, dim in enumerate(dims):
            if dim in self._da.coords:
                coords[dim] = self._da[dim].values
            else:
                coords[dim] = np.arange(arr.shape[i])

        # create new shape
        sh = [len(coords[dim]) for dim in dims]

        # reshape
        arr = arr.reshape(sh)

        return xr.DataArray(arr, dims=dims, coords=coords)


def coarsen(A, fun=np.mean, **kwargs):
    """Coarsen DataArray using reduction

    Parameters
    ----------
    A: DataArray
    axis: str
        name of axis
    q: int
        coarsening factor
    fun:
        reduction operator

    Returns
    -------
    y: DataArray


    Examples
    --------

    Load data and coarsen along the x dimension
    >>> name = "/scratch/noah/Data/SAM6.10.9/OUT_2D/HOMO_2km_16384x1_64_2000m_5s.HOMO_2K.smagor_16.2Dcom_*.nc"

    >>> ds = xr.open_mfdataset(name, chunks={'time': 100})
    >>> # tb = ds.apply(lambda x: x.meanevery('x', 32))
    >>> def f(x):
    ...     return x.coarsen(x=16)
    >>> dsc = ds.apply(f)
    >>> print("saving to netcdf")
    >>> dsc.to_netcdf("2dcoarsened.nc")
    """

    # this function needs a dask array to work
    if A.chunks is None:
        A = A.chunk({k:len(v) for k,v in A.coords.items()})

    coarse_dict = {A.get_axis_num(k): v for k,v in kwargs.items()}
    vals = da.coarsen(fun, A.data, coarse_dict)

    # coarsen dimension
    coords = {}
    for k in A.coords:
        if k in kwargs:
            c  = A[k].data
            dim = da.from_array(c, chunks=(len(c), ))

            q = kwargs[k]
            dim = da.coarsen(np.mean, dim, {0: q}).compute()
            coords[k] = dim
        else:
            coords[k] = A.coords[k]

    return xr.DataArray(vals, dims=A.dims, coords=coords, attrs=A.attrs,
                        name=A.name)


# Add custom functions to DataArray class dynamically
xr.DataArray.coarsen = coarsen
xr.DataArray.integrate = integrate
xr.DataArray.roll = roll
xr.DataArray.remove_repeats = remove_repeats


def wrapcli(fun):
    """Wrap a function which takes dataarrays as inputs
    """
    @functools.wraps(fun)
    def f(*args, data=[], **kwargs):
        ds = xr.merge(xr.open_dataset(f) for f in data)
        fun_args = (ds[arg] for arg in args)

        return fun(*fun_args, **kwargs)

    return f



