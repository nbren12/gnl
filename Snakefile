import xarray as xr
import os
import glob

def get_3d_files(wildcards):
    files = glob.glob(f"OUT_3D/*EQX*{wildcards.field}.nc")

    return [os.path.join("tmp/rec/tmp/coarse", x) for x in files]


rule tmpcoarsen:
    input: "{f}.nc"
    output: "tmp/coarse/{f}.nc"
    script: "coarsen.py"


rule makerecdim:
    input: "{f}.nc"
    output: temp("tmp/rec/{f}.nc")
    shell: "ncks --mk_rec_dmn time {input} {output}"


rule coarsen:
    input: get_3d_files
    output: "coarse/{field}.nc"
    # shell: "ncrcat -o {output} {input}"
    run:
        xr.open_mfdataset(input, concat_dim='time').to_netcdf(output[0])



rule all_coarse_vars:
    input: expand("coarse/{field}.nc", field='U V TABS QRAD QV QN QP'.split(' '))
