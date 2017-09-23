import xarray as xr
import dask
from xnoah.xarray import coarsen
import joblib

def f(x, ncoarse=40):
    if set(x.dims) >= set(['x', 'y']):
        return coarsen(x, x=ncoarse, y=ncoarse)
    else:
        return x


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
