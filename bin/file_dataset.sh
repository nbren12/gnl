#!/usr/bin/env bash
# Put file into data bucket on amazon

file=$1
bucket=gs://nbren12-data/data/id
dataroot=~/.datasets/id

echo "Calculating hash of file"
hash=$(cat $* | md5)
path=$dataroot/$hash

read -p "Move data to $path? [yn]" yn
case $yn in
    y )  break;;
    * ) exit;;
esac

mkdir $hash
mv $file $hash

[ ! -d $dataroot ] && mkdir -p  $dataroot
mv $hash $dataroot

# upload to google cloud
echo "file moved to path $path"
echo "To upload to google cloud run the following command:"
echo gsutil rsync -r $path/ $bucket/$hash
read -p "Would you like to run this command now [yn]" yn
case $yn in
    y ) gsutil rsync -r $path/ $bucket/$hash; break;;
    * ) exit;;
esac
