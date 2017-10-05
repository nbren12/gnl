#!/usr/bin/env python3
"""
Save images from latex log file into a single folder

Usage:
    extract_graphics.py LOG DIR

Arguments:
    LOG      input log
    DIR      directory to save to

"""
import sys
import os, shutil
import re


def main(name, d):

    s = open(name, "r").readlines()
    regex = re.compile(r"<use (.*)>")
    img_matches =[regex.search(ss) for ss in s if ss is not None]
    imgs = [a.group(1) for a in img_matches if a]

    if not os.path.exists(d):
        os.mkdir(d)

    for img in imgs:
        print("Copying {} to {}".format(img, d))
        shutil.copy(img, d)



if __name__ == '__main__':
    from docopt import docopt

    args = docopt(__doc__)
    main(args['LOG'], args['DIR'])
