#!/bin/sh

name=$1
dest=$2
category=$3

url=nbren12@bitbucket.org:nbren12/${name}.git

# enter git repository
pushd $dest
git remote add $name $url

# checkout new branch
# and move all files to subfolder
git fetch $name
git checkout -b $name-branch $name/master

contents=`ls`
mkdir tmp111343 
git mv $contents tmp111343
git mv .gitignore tmp111343

mkdir $category
git mv tmp111343 $category/$name
git commit -m "moved all files to $name"

# go back to old branch
git checkout master

# perform merge
git merge --allow-unrelated-histories --commit $name-branch # or whichever branch
git branch -D $name-branch
git remote remove $name

echo "All work merged in...ready to remove"
echo "$url"
echo https://bitbucket.org/nbren12/$1/admin
open https://bitbucket.org/nbren12/$1/admin





