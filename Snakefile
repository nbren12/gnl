import xarray as xr
import os
import glob

statfile = "OUT_STAT/NG_5120x2560x34_4km_10s_QOBS_EQX.nc"


def get_3d_files(wildcards):
    files = glob.glob(f"OUT_3D/*EQX*{wildcards.field}.nc")

    return [os.path.join("tmp/rec/tmp/coarse", x) for x in files]


rule tmpcoarsen:
    input: "{f}.nc"
    output: "tmp/coarse/{f}.nc"
    script: "scripts/coarsen.py"


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


# This is a very large job. 
rule all_coarse_vars:
    input: expand("coarse/{field}.nc", field='U V TABS QRAD QV QN QP'.split(' '))

rule all_pingping_processed:
    input: 'nc3hrlydata/q1.nc',  'nc3hrlydata/q2.nc',
           'nc3hrlydata/qt.nc', 'nc3hrlydata/sl.nc',
           "dataintp160km3hr_inst_trop/QRAD_nopert.nc",
           "dataintp160km3hr_inst_trop/QRAD_nopert_ext.nc",
           statfile

    output: expand("ave160km/{field}.nc", field='q1 q2 qt sl qrad'.split(' '))
    script: "scripts/cleanup.py"
