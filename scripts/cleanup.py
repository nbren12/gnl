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
        ds[k] = grid[k]


    return ds


def main():

    grid = create_grid("OUT_STAT/NG_5120x2560x34_4km_10s_QOBS_EQX.nc")

    data_paths = {
        'q1': 'nc3hrlydata/q1.nc',
        'q2': 'nc3hrlydata/q2.nc',
        'qt': 'nc3hrlydata/qt.nc',
        'sl': 'nc3hrlydata/sl.nc',
        'qrad': ["dataintp160km3hr_inst_trop/QRAD_nopert.nc", "dataintp160km3hr_inst_trop/QRAD_nopert_ext.nc"]
    }



    if not os.path.isdir("ave160km"):
        os.mkdir("ave160km")

    for key, val in data_paths.items():
        print(f"Fixing {key}")
        ds = xr.open_mfdataset(val, concat_dim='t')
        apply_grid(ds, grid).\
            to_netcdf(f"ave160km/{key}.nc")

if __name__ == '__main__':
    main()
