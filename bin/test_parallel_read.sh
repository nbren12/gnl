#!/bin/bash
# test for parallel read performance using dd and gnu parallel
# I have hardcoded OUT_3D which is a directory with many large netcdf files
# tags: netcdf io profile performance


NCDIR=$1

ntest=$2
nproc=$ntest

find $NCDIR -name '*.nc' | shuf | head -n $ntest  > test_list


function test_read {
  echo $1,`du -m $1 | cut -f1`, `dd if=$1 of=/dev/null bs=8k |& tail -n 1 | awk -F ',' '{print $3}'`
}

export -f test_read


parallel -j  $nproc -a test_list test_read
