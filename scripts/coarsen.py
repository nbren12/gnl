import xarray as xr
import dask
from xnoah.xarray import coarsen
import joblib

def f(x, ncoarse=40):
    if set(x.dims) >= set(['x', 'y']):
        return coarsen(x, x=ncoarse, y=ncoarse)
    else:
        return x


def destagger(x, dim):
    """Destagger input data array along dimension"""
    pass
    # TODO implement
    # assert that data not be chunked along x
    # or write a shift function that has different boundary conditions

def main(input, output, ncoarse=40):
    ds = xr.open_dataset(input)


    ds_coarse = ds.apply(f)

    ds_coarse.to_netcdf(output)
try:
    snakemake
except NameError:
    pass
else:
    main(snakemake.input[0], snakemake.output[0])
