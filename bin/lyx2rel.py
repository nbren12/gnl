#!/usr/bin/env python3
"""
Convert absolute paths to relative paths in a lyx document.

Prints output to standard output

Usage:
    lyx2rel.py <file>

"""
import os, re, sys
from functools import partial
from docopt import docopt



def match2rel(match, start=None):
    fn  = match.group(1)

    if fn[0] == '/':
        path =  os.path.relpath(fn, start=None)

    else:
        path = fn


    return "filename "+ path

def lyxfile_to_relpaths(fn):

    s = open(fn, 'r').read()
    file_dir = os.path.dirname(fn)

    out = re.sub(r"filename (.*)",
                partial(match2rel, start=file_dir),
                s)
    return out

if __name__ == '__main__':
    args = docopt(__doc__)

    print(lyxfile_to_relpaths(args['<file>']))
