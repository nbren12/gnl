import xarray as xr
import os
import glob
import xnoah.sam.coarsen


destination = "/home/disk/eos4/nbren12/Data/id/63929c31188b4c0eff9f9c36f8f648397cadcfa6"

statfile = "OUT_STAT/NG_5120x2560x34_4km_10s_QOBS_EQX.nc"

# data_paths = {
#     'q1': 'nc3hrlydata/q1.nc',
#     'q2': 'nc3hrlydata/q2.nc',
#     'qt': 'nc3hrlydata/qt.nc',
#     'sl': 'nc3hrlydata/sl.nc',
# }

data_paths = {}

for key in 'QRAD LHF Prec Q1 Q2 QN QP QT QV SHF SL TABS U V W'.split(' '):
    if key in 'Prec SHF LHF'.split(' '):
        path_key = f'2d/{key}'
    else:
        path_key = f'3d/{key}'

    data_paths[path_key] = [f"dataintp160km3hr_inst_trop/{key}_nopert.nc",
                            f"dataintp160km3hr_inst_trop/{key}_nopert_ext.nc"]

# This is a very large job.
rule all_coarse_vars:
    input: expand("coarse/{field}.nc", field='U V TABS QRAD QV QN QP'.split(' '))


def get_stagger_dim_mode(wildcards, input):
    d = xr.open_dataset(input[0], cache=False)
    name = [d for d in d.data_vars if d != 'p'][0]
    dim = {'U':'x', 'V':'y', 'W':'z'}.get(name, None)
    mode = {'x': 'wrap'}.get(dim, 'clip')
    return dim, mode

rule coarsen_one_file:
    input: "{f}.nc"
    output: "tmp/coarse/{f}.nc"
    params: blocks={'x': 40, 'y':40}
    run:
        params = dict(params)
        params['stagger_dim']=get_stagger_dim_mode
        xnoah.sam.coarsen.snakemake(input, output, params)


rule make_record_dim:
    input: "{f}.nc"
    output: temp("tmp/rec/{f}.nc")
    shell: "ncks --mk_rec_dmn time {input} {output}"


def get_3d_files(wildcards):
    files = glob.glob(f"OUT_3D/*EQX*{wildcards.field}.nc")

    return [os.path.join("tmp/rec/tmp/coarse", x) for x in files]

rule all_record_vars:
    input: get_3d_files
    output: "coarse/{field}.nc"
    # shell: "ncrcat -o {output} {input}"
    run:
        xr.open_mfdataset(input, concat_dim='time').to_netcdf(output[0])




p2_vars = expand("ave160km/{field}.nc", field=data_paths.keys())

rule all_pingping_processed:
    input: stat=statfile, **data_paths

    output: p2_vars
    params: data_paths=data_paths
    script: "scripts/cleanup.py"

rule move:
    input: p2_vars
    output: destination
    shell: "mv ave160km {output}"

