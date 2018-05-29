import xarray as xr
from xnoah.sam import regrid as rg


def average_east(u, blocks):
    w = rg.staggered_to_left(u, blocks['x'], 'x')
    x = u['x'][::blocks['x']]
    w['x'] = x
    # average fluxes over the interface
    return rg.coarsen_dim(w, blocks['y'], 'y')\
        .rename({'x': 'xs', 'y': 'yc'})


def average_south(v, blocks):
    s = rg.staggered_to_left(v, blocks['y'], 'y')
    y = v['y'][::blocks['y']]
    s['y'] = y
    # average fluxes over the interface
    return rg.coarsen_dim(s, blocks['x'], 'x')\
        .rename({'x': 'xc', 'y': 'ys'})

ds = xr.open_mfdataset(snakemake.input).load()

out = xr.Dataset({
    'U': average_east(ds.U, snakemake.params.blocks),
    'V': average_south(ds.V, snakemake.params.blocks)
})

out.to_netcdf(snakemake.output[0])
