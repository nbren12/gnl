import dask.array as da
import numpy as np
import xarray as xr
from xarray.core.computation import apply_ufunc

from functools import wraps


def dfun(func):
    @wraps(func)
    def f(x, *args, **kwargs):
        if isinstance(x, xr.DataArray):
            return func(x, *args, **kwargs)
        elif isinstance(x, xr.Dataset):
            return x.apply(lambda x: func(x, *args, **kwargs))
    return f


def coarsen_destagger_dask(x, blocks, stagger=None, mode='wrap'):
    """


    Examples
    --------
    >>> x = da.arange(6, chunks=6)
    >>> xc = coarsen_destagger_dask(x, {0: 2}, stagger=0)
    >>> xc.compute()
    array([ 1. ,  3. ,  3.5])
    >>> x = da.from_array(x, chunks=x.shape)
    >>> xc = coarsen_destagger_dask(x, {0: 2}, stagger=0)
    >>> xc.compute()
    array([ 1. ,  3. ,  3.5])
    """
    output_numpy = False

    try:
        x._keys
    except AttributeError:
        output_numpy = True
        x = da.from_array(x, x.shape)

    xcoarse = da.coarsen(np.sum, x, blocks)
    # TODO refactor this code to another function
    if stagger is not None:
        blk = {key: val
               for key, val in blocks.items()
               if key != stagger}

        left_inds = np.arange(0, x.shape[stagger], blocks[stagger])
        left = da.coarsen(np.sum, da.take(x, left_inds, stagger), blk)
        n = left.shape[stagger]
        # handle boundary conditions
        if mode == 'wrap':
            bc = da.take(left, [0], axis=stagger)
        if mode == 'clip':
            bc = da.take(left, [-1], axis=stagger)

        right = da.take(left, np.arange(1, n), axis=stagger)
        right = da.concatenate((right, bc), axis=stagger)
        xcoarse = xcoarse + (right - left)/2

    n = np.prod(list(blocks.values()))
    ans = xcoarse/n

    if output_numpy:
        return ans.compute()
    else:
        return ans


@dfun
def coarsen(A, blocks, stagger_dim=None, mode='wrap'):
    """coarsen and potentially destagger a 
    """
    blocks = {k:blocks[k] for k in blocks
              if k in A.dims}

    if len(blocks) == 0:
        return A

    kwargs = {'mode': mode}
    if stagger_dim is not None:
        kwargs['stagger'] = A.get_axis_num(stagger_dim)

    np_blocks = {A.get_axis_num(dim): val for dim, val in blocks.items()}
    vals = coarsen_destagger_dask(A.data, np_blocks, **kwargs)

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

@dfun
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
    coarsen(ds, blocks, **kwargs).to_netcdf(output)


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
