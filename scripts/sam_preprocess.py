#!/usr/bin/env python
"""Pre-process SAM outputs into netCDF format

The output directory structure will look the same as the typical SAM directory,
but with all the weird binary formats converted to netCDFs.

"""
import argparse
import glob
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

subprocess_queue = []


def process_binary_output_dir(src, dest):
    os.mkdir(dest)
    # link the files
    for file in os.listdir(src):
        os.symlink(os.path.join(src, file), os.path.join(dest, file))
    # run the SAM UTILS conversion command
    for file in os.listdir(dest):
        ext = os.path.splitext(file)[1][1:]
        conversion_cmd = ext + '2nc'
        p = subprocess.Popen(
            [conversion_cmd, file], stdout=subprocess.DEVNULL, cwd=dest)

        symlink = os.path.join(dest, file)
        subprocess_queue.append((symlink, p))


def copy(src, dest):
    try:
        fname = os.path.basename(src)
        shutil.copytree(src, os.path.join(dest, fname))
    except NotADirectoryError:
        shutil.copy(src, dest)


parser = argparse.ArgumentParser()
parser.add_argument('input', help="SAM output directory")
parser.add_argument("output", help="output directory")
args = parser.parse_args()

input_dir = os.path.abspath(args.input)
file_list = os.listdir(input_dir)
data_dirs = 'OUT_2D OUT_3D OUT_ISO OUT_MOMENTS OUT_MOVIES OUT_STAT'.split(' ')

output_dir = os.path.abspath(args.output)
os.mkdir(output_dir)

logger.info(f"Preprocessing {input_dir}")
for obj in file_list:
    if obj not in data_dirs:
        logging.info(f"Copying {obj} to {output_dir}")
        src = os.path.join(input_dir, obj)
        copy(src, output_dir)

# Process the binary outputs
for data_dir in data_dirs:
    logging.info(f"Processing binary outputs in {data_dir}")
    src = os.path.join(input_dir, data_dir)
    dest = os.path.join(output_dir, data_dir)
    process_binary_output_dir(src, dest)

while subprocess_queue:
    symlink, p = subprocess_queue.pop()
    p.wait()
    logging.info(f"Done processing {symlink}. Removing symlink.")
    os.unlink(symlink)
