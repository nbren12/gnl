from os.path import join as fullfile


def findroot(path='.'):
    """Find root directory of project

    The root is assumed to contain .git

    Parameters
    ----------
    path : str, optional
        path to search backwards from
    """
    import os
    from os.path import split, join, abspath, dirname
    d = abspath(path)
    if not os.path.isdir(d): d = dirname(d)
    while not '.git' in os.listdir(d):
        d = dirname(d)

        if d == '/':
            break

    return d


def addroot2path(path="."):
    """Add root of project to path"""
    import sys
    sys.path.insert(0, findroot(path))
