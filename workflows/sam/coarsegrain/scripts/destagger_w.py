import xarray as xr
from xnoah.sam.coarsen import destagger

d = xr.open_dataset(snakemake.input[0])
destagger(d, "z", mode="clip").to_netcdf(snakemake.output[0])
