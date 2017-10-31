#!/bin/sh
git clone https://github.com/singularityware/singularity.git
cd singularity
./autogen.sh
./configure --prefix=/usr/local
make
make install
cd ..
rm -rf singularity
