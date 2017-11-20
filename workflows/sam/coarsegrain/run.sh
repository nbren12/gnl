#!/bin/sh

snakemake all \
    --directory  /home/disk/eos4/nbren12/eos8/Data/id/2 \
    --configfile config.yaml \
    -j 4
