import xarray as xr
import numpy as np

dy = 160e3
dx = 160e3
dt = 3/24

nx,ny,nt = 128,16,320

# get z from statfile
z = xr.open_dataset(snakemake.input[0]).z


grid = xr.Dataset({
    'x': np.r_[:nx] * dx,
    'y': np.r_[:ny] * dy,
    'time': np.r_[:ny] * dt,
    'z': z,
}).set_coords(["x", "y", "time", "z"])


grid.x.attrs['units'] = 'm'
grid.y.attrs['units'] = 'm'
grid.time.attrs['units'] = 'd'

grid.to_netcdf(snakemake.output[0])



