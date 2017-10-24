#!/bin/sh

snakemake all \
    --config \
       runid=EQX_1280 \
       data=/home/disk/eos17/guest/SAM6.10.6_NG \
       stat=/home/disk/eos13/guest/SAM6.10.6_NG/OUT_STAT/NG_5120x2560x34_4km_10s_QOBS_EQX.nc \
       output=/home/disk/eos4/nbren12/eos8/Data/id/726a6fd3430d51d5a2af277fb1ace0c464b1dc48
