"""
This snakemake file defines three operations

  D - centered derivatives
  M - material derivatives

TODO make these rules work for three derivative data
"""
import os
import xarray as xr
from gnl.xarray import coarsen
import gnl.xcalc


def xopen(name, nt=20):
    return xr.open_dataset(name, chunks=dict(time=nt))\
             .apply(lambda x: x.squeeze())

def firstvar(d):
    return d[next(iter(d.data_vars))]

def wrap_xarray_calculation(f):
    def fun(*args, **kwargs):
        args = (firstvar(xopen(a)) for a in args)
        return f(*args, **kwargs)

    return fun


def calc_der(f, dim):
    bdy = {'x': 'periodic', 'z':'extrap', 'time':'extrap'}
    return f.centderiv(dim, boundary=bdy[dim])


@wrap_xarray_calculation
def calc_matder(dt, dx, dz, u, w):
    return (dt + dx*u + dz*w)

wildcard_constraints:
    dim="(x|time|z)", # used for arguments to operators
    field="[^/]+"    # no path seperators


rule derivative:
    input: f="{f}.nc"
    output:temp("D{dim}/{f}.nc")
    run:
        dim = wildcards.dim
        f = xopen(input[0])\
            .apply(lambda x: calc_der(x, dim) if dim in x.dims else x)\
            .to_netcdf(output[0])

rule material_derivative:
    input: u="{d}/U.nc", w="{d}/W.nc",
           dx="Dx/{d}/{field}.nc",
           dz="Dz/{d}/{field}.nc",
           dt="Dtime/{d}/{field}.nc"
    output: temp("M/{d}/{field}.nc")
    run:
        out = calc_matder(input.dt, input.dx, input.dz, input.u, input.w)
        out.to_dataset(name=wildcards.field).to_netcdf(output[0])
