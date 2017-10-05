#!/bin/sh
file=Miniconda3-latest-MacOSX-x86_64.sh
url=https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

anaconda=$1

wget $url
mv $anaconda $anaconda.bak
echo "Installing miniconda...enter $anaconda as the installation path"
bash $file

rsync -av $anaconda/ $anaconda.bak/
mv $anaconda $anaconda.deleteme
mv $anaconda.bak $anaconda

echo "Go ahead and delete $anaconda.deleteme"
