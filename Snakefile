import xarray as xr
import os
import glob
import xnoah.sam.coarsen

dataroot = "/home/disk/eos17/guest/SAM6.10.6_NG"
workdir: "/home/disk/eos4/nbren12/eos8/Data/id/726a6fd3430d51d5a2af277fb1ace0c464b1dc48"


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
    input: expand("coarse/3d/{field}.nc", field='U V TABS QRAD QV QN QP'.split(' '))


def get_stagger_dim_mode(input):
    d = xr.open_dataset(input[0], cache=False)
    name = [d for d in d.data_vars if d != 'p'][0]
    dim = {'U':'x', 'V':'y', 'W':'z'}.get(name, None)
    mode = {'x': 'wrap'}.get(dim, 'clip')
    return dim, mode

rule coarsen_one_file:
    input: "{f}.nc"
    output: temp("tmp/coarse{f}.nc")
    params: blocks={'x': 40, 'y':40}
    run:
        params = dict(params)
        params['stagger_dim']=get_stagger_dim_mode(input)
        xnoah.sam.coarsen.snakemake(input, output, params)


rule make_record_dim:
    input: "{f}.nc"
    output: temp("tmp/rec/{f}.nc")
    shell: "ncks --mk_rec_dmn time {input} {output}"


def get_3d_files(wildcards):
    pattern = f"{dataroot}/OUT_3D/*EQX*{wildcards.field}.nc"
    files = glob.glob(pattern)
    if len(files) == 0:
        raise ValueError("No files detected")
    return [os.path.normpath(f"tmp/rec/tmp/coarse/{x}") for x in files]

rule all_record_vars:
    input: get_3d_files
    output: "coarse/3d/{field}.nc"
    # shell: "ncrcat -o {output} {input}"
    run:
        xr.open_mfdataset(input, concat_dim='time').to_netcdf(output[0])



destination = "/home/disk/eos4/nbren12/Data/id/63929c31188b4c0eff9f9c36f8f648397cadcfa6"
statfile = "OUT_STAT/NG_5120x2560x34_4km_10s_QOBS_EQX.nc"

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

