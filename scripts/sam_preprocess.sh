#!/bin/sh
# Preprocess model output from the system for atmospheric modeling (SAM) into
# netcdf files.  this file calls com2nc and the other functions in the SAM/UTIL
# folder and generates a nice directory structure for the output in a
# one-variable-per-file setup.

function splitvar() {
    output=$3/$1.nc
    if [ ! -f $output ]; then 
        ncrcat -v $1 -o $output  $2/*.nc
    fi
}
export -f splitvar

function process3d() {
    vars3d="U V W PP TABS QV QN QP"

    mkdir -p data/3d

    if [ -f OUT_3D/*.nc ]; then 
        echo "com2D2nc output already exists"
    else
        echo "running com2D2nc"
        pushd OUT_3D/
        com2D2nc *.com2D > /dev/null
        popd
    fi
    echo "Splitting all output variables"
    parallel splitvar ::: $vars3d ::: OUT_3D ::: data/3d/
}

function process2d() {

	vars2d="x time Prec SHF LHF CWP IWP CLD PW USFC U200 VSFC V200 W500 PSFC SWVP U850 V850 ZC TB ZE "
    if [ -e OUT_2D/*_1.nc ]; then 
        echo "2D output already exists"
    else
        echo "running com2D2nc"
        pushd OUT_2D/
        2Dcom2nc *.2Dcom > /dev/null
        popd
    fi

    echo "Splitting all output variables"
    mkdir -p data/2d/
    parallel splitvar ::: $vars2d ::: OUT_2D ::: data/2d/
}

function processStat() {
    output=data/stat.nc
    if [ ! -f $output ]; then
        echo "Processing Stats"
        pushd OUT_STAT
        stat2nc *.stat > /dev/null
        mv *.nc ../$output
        popd
    fi
}

if [ -d $1 ] ; then
    cd $1
fi
    process3d
    process2d
    processStat



