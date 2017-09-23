import xarray as xr
import numpy as np

dy = 160e3
dx = 160e3
dt = 3/24

statfile = "OUT_STAT/NG_5120x2560x34_4km_10s_QOBS_EQX.nc"

nx,ny,nt = 128,16,320

# get z from statfile
z = xr.open_dataset(statfile).z


grid = xr.Dataset({
    'x': np.r_[:nx] * dx,
    'y': np.r_[:ny] * dy,
    'time': np.r_[:ny] * dt,
    'z': z,
}).set_coords(["x", "y", "time", "z"])

from IPython import embed; embed()


