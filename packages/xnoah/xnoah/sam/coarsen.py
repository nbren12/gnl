import dask.array as da
import numpy as np
import xarray as xr
from skimage.measure import block_reduce
from xarray.core.computation import apply_ufunc

from functools import wraps

class with_dims(object):
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, func):

        @wraps(func)
        def f(arr, **kwargs):
            if set(arr.dims) >= set(self.dims):
                return func(arr, **kwargs)
            else:
                return arr
        return f


def coarsen_destagger_np(x, blocks, stagger=None, mode='wrap'):
    """


    Examples
    --------
    >>> x = np.arange(6)
    >>> xc = coarsen_destagger_np(x, (2,), stagger=0)
    >>> xc
    array([ 1. ,  3. ,  3.5])
    """
    blocks = tuple(blocks)
    xcoarse = block_reduce(x, blocks, np.sum)
    # TODO refactor this code to another function
    if stagger is not None:
        blk = tuple([blocks[k] if k != stagger else 1
                     for k, _ in enumerate(blocks)])

        left_inds = np.arange(0, x.shape[stagger], blocks[stagger])
        right_inds = np.arange(blocks[stagger], x.shape[stagger]+1, blocks[stagger])

        left = block_reduce(np.take(x, left_inds, stagger), blk, np.sum)
        right = block_reduce(np.take(x, right_inds, stagger, mode=mode),
                             blk, np.sum)

        xcoarse = xcoarse + (right - left)/2

    return xcoarse/np.prod(blocks)


def coarsen(A, blocks, stagger_dim=None, mode='wrap'):
    """Coarsen DataArray using reduction
    """

    np_blocks = [blocks.get(A.dims[i], 1)
                 for i in range(A.ndim)]

    kwargs = {'mode': mode}
    if stagger_dim is not None:
        kwargs['stagger'] = A.get_axis_num(stagger_dim)

    vals = coarsen_destagger_np(A.data, np_blocks, **kwargs)

    # coarsen dimension
    coords = {}
    for k in A.coords:
        if k in blocks:
            c  = A[k].data
            dim = da.from_array(c, chunks=(len(c), ))

            q = blocks[k]
            dim = da.coarsen(np.mean, dim, {0: q}).compute()
            coords[k] = dim
        else:
            coords[k] = A.coords[k]

    return xr.DataArray(vals, dims=A.dims, coords=coords, attrs=A.attrs,
                        name=A.name)


def destagger_dask(darr, mode='wrap'):
    ind = np.arange(darr.shape[-1]) + 1
    r = (np.take(darr, ind, axis=-1, mode=mode) +darr)/2
    return r


def destagger(xarr, dim, **kwargs):
    """Destagger an inteface located variable along a dimension

    Parameters
    ----------
    xarr : xr.Dataset
        input datarray
    dim : str
        dimension to destagger the data along
    mode : str
        Passed to np.take

    Returns
    -------
    destaggered : xr.Dataset
        cell centered DataArray

    See Also
    --------
    numpy.take

    Examples
    --------
    >>> x = xr.DataArray(np.arange(0, 5), [('x', np.arange(0, 5))])
    >>> destagger(x, 'x')
    <xarray.DataArray (x: 5)>
    array([ 0.5,  1.5,  2.5,  3.5,  2. ])
    Coordinates:
      * x        (x) int64 0 1 2 3 4
    """
    return apply_ufunc(destagger_dask, xarr,
                       input_core_dims=[[dim]],
                       output_core_dims=[[dim]],
                       dask_array='forbidden',
                       kwargs=kwargs)



def main(input, output, blocks, **kwargs):
    ds = xr.open_dataset(input)
    fun = with_dims(blocks.keys())(lambda ds: coarsen(ds, blocks, **kwargs))
    ds.apply(fun).to_netcdf(output)


def test_coarsen():
    import os
    file = "NG_5120x2560x34_4km_10s_QOBS_EQX_1280_0000173880_TABS.nc"
    folder = "~/.datasets/id/1381d73c091f2ea34ef8ea303c94e998/"
    path = os.path.join(folder, file)
    main(path, "coarse.nc", {'x': 40, 'y': 40}, stagger_dim=None, mode='wrap')

try:
    snakemake
except NameError:
    # import sys
    # main(sys.argv[1], sys.argv[2])
    if __name__ == '__main__':
        test_coarsen()
else:
    main(snakemake.input[0], snakemake.output[0],
         ncoarse=snakemake.params.coarsening,
         stagger=snakemake.params.stagger)
