import glob
import re

def get_files_for_time_step(config, run, time):
    dataroot = config['dataroot']
    fields = ['QV', 'QN', 'QP', 'U', 'V', 'W', 'QRAD', 'TABS', 'PP']
    files = [f"{dataroot}/{run}_1280_{time}_{field}.nc" for field in fields]
    return files


def get_time_steps(config, run):
    dataroot = config['dataroot']
    pattern = f"{dataroot}/{run}_*U.nc"
    files = glob.glob(pattern)
    pat = re.compile(f".*_(\d+)_U.nc")

    timesteps = [pat.search(file).group(1)
                 for file in files]

    return timesteps
