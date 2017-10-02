"""
Concatenate many many netcdf files

The combined dataset should fit in memory
"""
import xarray as xr


batchsize = snakemake.params.get('batchsize', 100)
files = list(snakemake.input)
datasets = []
counter = 1
while True:
    print(f"Processing batch {counter}")
    counter += 1

    batchsize = min(batchsize, len(files))
    if batchsize == 0:
        break
    da = xr.open_mfdataset(files[:batchsize], concat_dim='time').compute()
    del files[:batchsize]
    datasets.append(da)

xr.auto_combine(datasets, concat_dim='time').to_netcdf(snakemake.output[0])
