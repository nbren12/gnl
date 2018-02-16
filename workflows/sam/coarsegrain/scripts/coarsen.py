import xarray as xr
from xnoah import sam


def coarse_grain_with_tendencies(ds, blocks):
    variables = {}

    for scalar in ['QN', 'QP', 'TABS', 'QV']:
        variables[f'div{scalar}'] = sam.advect_scalar(ds.U, ds.V, ds[scalar], blocks=blocks)

    for variable in ['QV', 'QN', 'QP', 'TABS', 'PP', 'QRAD', 'W']:
        variables[variable] = sam.coarsen(ds[variable], blocks=blocks)

    variables['U'] = sam.coarsen(ds.U, blocks, stagger_dim='x')
    variables['V'] = sam.coarsen(ds.V, blocks, stagger_dim='y', mode='clip')
    return xr.Dataset(variables)


def main():
    ds = xr.open_mfdataset(snakemake.input).load()
    out = coarse_grain_with_tendencies(ds, blocks=snakemake.params.blocks)
    out.to_netcdf(snakemake.output[0])


main()
