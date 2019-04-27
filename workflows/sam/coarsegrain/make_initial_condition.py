import xarray as xr
import yaml
from ngaqua import *
from gnl.xarray.sam import regrid
from os.path import join

# open config
with open("config.yaml") as f:
    config = yaml.load(f)


# get files
run = "NG_5120x2560x34_4km_10s_QOBS_EQX"
steps = get_time_steps(config, run)
first_step = steps[0]
step_nc_files = get_files_for_time_step(config, run, first_step)

# open data
data = xr.open_mfdataset(step_nc_files)

# stat files
stat = xr.open_mfdataset(join(config['stat_root'], run + '.nc'))

# regrid the data
blocks = {'x': 10, 'y': 10}
out = regrid.regrid_3d(data, blocks)
out = out.compute()

out['RHO'] = stat.RHO[0]
out['Ps'] = stat.Ps[0]

out.to_netcdf(f"{run}-x{blocks['x']}-y{blocks['y']}-{first_step}.nc")
