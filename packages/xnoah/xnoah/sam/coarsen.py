import numpy as np
import xarray as xr
from xarray.core.computation import apply_ufunc
import dask.array as da

def dataset2dfun(coarsen):
    def f(arr, **kwargs):
        if isinstance(arr, xr.DataArray):
            return coarsen(arr, **kwargs)
        elif isinstance(arr, xr.Dataset):
            def g(x):
                if set(x.dims) >= set(['x', 'y']):
                    return coarsen(x, **kwargs)
                else:
                    return x
            return arr.apply(g)

    return f


@dataset2dfun
def coarsen_xy(A, fun=np.mean, **kwargs):
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

    print(f"Coarsening data to {kwargs}")
    # this function needs a dask array to work
    if A.chunks is None:
        A = A.chunk()

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


def destagger_dask(darr, mode=None):
    ind = np.arange(darr.shape[-1])
    r = (np.take(darr, ind, axis=-1, mode=mode) +darr)/2
    return r


@dataset2dfun
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

    """
    print(f"Destaggering {xarr.name} along dim {dim}")

    return apply_ufunc(destagger_dask, xarr,
                       input_core_dims=[[dim]],
                       output_core_dims=[[dim]],
                       dask_array='forbidden',
                       kwargs=kwargs)


def main(input, output, ncoarse=40, stagger=None, mode=None):
    ds = xr.open_dataset(input)
    if stagger is not None:
        print("Destaggering variable")
        ds = destagger(ds, dim=stagger, mode=mode)

    coarsen_xy(ds, x=ncoarse, y=ncoarse)\
        .transpose("time", "z", "y", "x")\
        .to_netcdf(output)


def test_coarsen():
    import os
    file = "NG_5120x2560x34_4km_10s_QOBS_EQX_1280_0000173880_TABS.nc"
    folder = "~/.datasets/id/1381d73c091f2ea34ef8ea303c94e998/"
    path = os.path.join(folder, file)
    main(path, "coarse.nc", stagger='x', mode=None)


try:
    snakemake
except NameError:
    # import sys
    # main(sys.argv[1], sys.argv[2])
    test_coarsen()
else:
    main(snakemake.input[0], snakemake.output[0],
         ncoarse=snakemake.params.coarsening,
         stagger=snakemake.params.stagger)
