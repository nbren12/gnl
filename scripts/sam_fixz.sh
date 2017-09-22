#!/bin/sh
# take z axis from stats.nc and apply it to all 3d netcdf files
# works for SAM output

find data/3d -name '*.nc' |grep -v 'stat.nc' | parallel -j 4 ncks -A -v z data/stat.nc {}
