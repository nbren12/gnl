import shutil
import os
import xarray as xr
import numpy as np

dt = 3/24

def create_grid(statfile):

    dy = 160e3
    dx = 160e3

    nx,ny,nt = 128,16,320

    # get z from statfile
    z = xr.open_dataset(statfile).z


    grid = xr.Dataset({
        'x': np.r_[:nx] * dx,
        'y': np.r_[:ny] * dy,
        'time': np.r_[:ny] * dt,
        'z': z,
    }).set_coords(["x", "y", "time", "z"])


    grid.x.attrs['units'] = 'm'
    grid.y.attrs['units'] = 'm'
    grid.time.attrs['units'] = 'd'


    return grid




def apply_grid(ds, grid):



    if 't' in ds.dims:
        ds = ds.rename({'t': 'time'})


    nt = len(ds['time'])
    if nt != len(grid['time']):
        grid['time'] = np.r_[:nt] * dt

    # set values
    for k in grid:
        if k in ds.dims:
            ds[k] = grid[k]

    return ds


def main():

    grid = create_grid(snakemake.input.stat)

    data_paths = snakemake.input



    if not os.path.isdir("ave160km"):
        os.mkdir("ave160km")

    for key, val in data_paths.items():
        if key !='stat':
            print(f"Fixing {key}")
            ds = xr.open_mfdataset(val, concat_dim='t')
            ds = apply_grid(ds, grid)
            ds.to_netcdf(f"ave160km/{key}.nc")
    shutil.copy(snakemake.input.stat, "ave160km/stat.nc")

if __name__ == '__main__':
    main()
