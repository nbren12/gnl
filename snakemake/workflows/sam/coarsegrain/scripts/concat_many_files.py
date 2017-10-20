"""
Concatenate many many netcdf files

The combined dataset should fit in memory
"""
import xarray as xr
import rx


batchsize = snakemake.params.get('batchsize', 100)
files = list(snakemake.input)
datasets = []


def open_files(xs):
    d =  xr.open_mfdataset(xs, concat_dim='time').compute()
    print("Opened data from", float(d.time.min()),
          "to", float(d.time.max()))
    return d
     
def concat(xs):
    print("concatenating")
    len(xs)
    xr.auto_combine(xs, concat_dim='time').to_netcdf(snakemake.output[0])


rx.Observable.from_(files)\
        .buffer_with_count(batchsize)\
        .map(open_files)\
        .to_list()\
        .subscribe(concat)
