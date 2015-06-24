from os.path import join as fullfile

def findroot(path='.'):
    import os
    from os.path import split, join, abspath, dirname
    d = abspath(path)
    if not os.path.isdir(d): d = dirname(d)
    while not '.git' in os.listdir(d):
        d = dirname(d)

        if d == '/':
            break

    return d

