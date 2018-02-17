import xarray as xr
from xnoah.sam.coarsen import destagger

d = xr.open_dataset(snakemake.input[0])
destagger(d, "z", mode="extrap").to_netcdf(snakemake.output[0])
