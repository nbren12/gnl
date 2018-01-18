#!/bin/sh

file=/home/disk/eos13/guest/SAM6.10.6_NG/OUT_2D/NG_5120x2560x34_4km_10s_QOBS_EQX_4K_1280_0000861480.2Dcom_001.nc

# tgt=NG_5120x2560x34_4km_10s_QOBS_EQX_4K/coarse/2d/all.nc
tgt=NG_5120x2560x34_4km_10s_QOBS_EQX/coarse/2d/all.nc

snakemake $tgt \
    --directory  /home/disk/eos4/nbren12/eos8/Data/id/2 \
    --configfile config.yaml \
    -j 4 \
    
    


