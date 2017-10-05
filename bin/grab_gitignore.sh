#!/bin/sh
# grab git ignore from the convinient list of gitignore files maintained by
# github.
# 
# The available options include:
# - Python
# - TeX
# - many more
# tag: git 

file=$1
curl https://raw.githubusercontent.com/github/gitignore/master/$file.gitignore
